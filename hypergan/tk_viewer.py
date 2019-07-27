"""
Opens a window that displays an image.
Usage:

    from viewer import GlobalViewer
    GlobalViewer.update(image)

"""
import numpy as np
import os
import contextlib


class TkViewer:
    def __init__(self, title="HyperGAN", viewer_size=1, enabled=True):
        self.screen = None
        self.title = title
        self.viewer_size = viewer_size
        self.enabled = enabled
        self.enable_menu = True

    def update(self, gan, image):
        if not self.enabled: return

        original_image = image
        if len(np.shape(image)) == 2:
            s = np.shape(image)
            image = np.reshape(image, [s[0], s[1], 1])
            image = np.tile(image, [1,1,3])
        image = np.transpose(image, [1, 0,2])

        if not self.screen:

            with contextlib.redirect_stdout(None):
                import pygame

            import tkinter as tk
            import tkinter.ttk
            class ResizableFrame(tk.Frame):
                def __init__(self,parent,tkviewer=None,**kwargs):
                    tk.Frame.__init__(self,parent,**kwargs)
                    self.bind("<Configure>", self.on_resize)
                    self.height = kwargs['height']
                    self.width = kwargs['width']
                    self.tkviewer = tkviewer
                    self.aspect_ratio = float(self.width)/float(self.height)

                def on_resize(self,event):
                    wscale = float(event.width)/self.width
                    hscale = float(event.height)/self.height
                    self.width = event.width
                    self.height = event.height
                    self.config(width=self.width, height=self.height)
                    self.tkviewer.size = [self.width, self.height]
                    self.tkviewer.screen = self.tkviewer.pg.display.set_mode(self.tkviewer.size,self.tkviewer.pg.RESIZABLE)   
                    self.enforce_aspect_ratio(event)
                    surface = self.tkviewer.pg.Surface([image.shape[0],image.shape[1]])
                    self.tkviewer.pg.surfarray.blit_array(surface, image[:,:,:3])
                    self.tkviewer.screen.blit(self.tkviewer.pg.transform.scale(surface,self.tkviewer.size),(0,0))
                    self.tkviewer.pg.display.flip()


                def enforce_aspect_ratio(self, event):
                    desired_width = event.width
                    desired_height = int(event.width / self.aspect_ratio)

                    if desired_height > event.height:
                        desired_height = event.height
                        desired_width = int(event.height * self.aspect_ratio)

                    self.config(width=desired_width, height=desired_height)
                    self.tkviewer.size = [desired_width, desired_height]
                    self.tkviewer.screen = self.tkviewer.pg.display.set_mode(self.tkviewer.size,self.tkviewer.pg.RESIZABLE)   


            self.size = [int(image.shape[0] * self.viewer_size), int(image.shape[1] * self.viewer_size)]

            self.pg = pygame
            self.tk = tk
            root = tk.Tk(className=self.title)
            embed = ResizableFrame(root, width=self.size[0], height=self.size[1], tkviewer=self)
            root.rowconfigure(0,weight=1)
            root.rowconfigure(1,weight=1)
            root.columnconfigure(0,weight=1)
            root.columnconfigure(1,weight=1)
            embed.pack(expand=tk.YES, fill=tk.BOTH)

            def _save_model(*args):
                gan.save(gan.save_file)

            def _exit(*args):
                gan.exit()

            def _refresh_sample(*args):
                gan.cli.sample(False)

            def _select_sampler(gan, name, value, submenu):
                def _select_sampler_proc():
                    gan.cli.sampler = gan.cli.sampler_for(name)(gan)
                    gan.cli.sample(False)
                    _refresh_sampler_submenu(submenu)
                return _select_sampler_proc

            def _refresh_sampler_submenu(submenu):
                if submenu.count > 0:
                    submenu.delete(0, submenu.count)

                for (k, v) in gan.get_registered_samplers().items():
                    showall = tk.BooleanVar()
                    showall.set(gan.selected_sampler == k)
                    if v.compatible_with(gan):
                        state = tk.NORMAL
                    else:
                        state = tk.DISABLED

                    print("Selected", gan.selected_sampler, k, gan.selected_sampler == k)
                    submenu.add_checkbutton(label=k, onvalue=True, offvalue=False, variable=showall, command=_select_sampler(gan, k, showall, submenu), state=state)
                num_samplers = len(gan.get_registered_samplers())

                submenu.count = num_samplers


            def _create_status_bar(root):
                statusbar = tk.Frame(root, height=24)
                statusbar.pack(side=tk.BOTTOM, fill=tk.X)

                label_training = tk.Label(statusbar, text="Training", font=12)
                label_training.grid(row=0,column=0) 
                sep = tkinter.ttk.Separator(statusbar, orient=tk.VERTICAL).grid(column=1, row=0, sticky='ns')
                label = tk.Label(statusbar, text="Starting", font=12)
                label.grid(row=0, column=2) 
                def __update_step():
                    if hasattr(gan, 'step_count'):
                        label['text']=("Step " + str(gan.step_count))
                    root.after(1000, __update_step)


                __update_step()
                return statusbar

            menubar = tk.Menu(root)
            filemenu = tk.Menu(menubar, tearoff=0)
            filemenu.add_command(label="Save", command=_save_model, underline=0, accelerator="Ctrl+s")

            filemenu.add_separator()

            samplemenu = tk.Menu(menubar, tearoff=0)
            samplemenu.add_command(label="Refresh", command=_refresh_sample, underline=0, accelerator="Ctrl+r")

            filemenu.add_command(label="Save and Exit", command=_exit, underline=10, accelerator="Ctrl+q")
            menubar.add_cascade(label="File", menu=filemenu, underline=0)
            menubar.add_cascade(label="Sample", menu=samplemenu, underline=0)
            samplermenu = tk.Menu(samplemenu)
            samplemenu.add_cascade(label="Sampler", menu=samplermenu, underline=0)
            samplermenu.count = 0
            _refresh_sampler_submenu(samplermenu)

            root.bind_all("<Control-q>", _exit)
            root.bind_all("<Control-r>", _refresh_sample)
            root.bind_all("<Control-s>", _save_model)


            if self.enable_menu:
                root.config(menu=menubar)
                _create_status_bar(root)

            # Tell pygame's SDL window which window ID to use
            os.environ['SDL_WINDOWID'] = str(embed.winfo_id())
            # Show the window so it's assigned an ID.
            root.update()
            self.root = root

            # Usual pygame initialization
            if self.viewer_size <= 0:
                self.viewer_size = 0.1
            self.aspect_w = image.shape[1] / image.shape[0]
            self.aspect_h = image.shape[0] / image.shape[1]
            self.temp_size = self.size
            self.screen = self.pg.display.set_mode(self.size,self.pg.RESIZABLE)
            self.pg.display.set_caption(self.title)

            root.title(self.title)
            root.wm_title(self.title)
            embed.winfo_toplevel().title(self.title)

        padw = 0
        padh = 0
        if original_image.shape[0] > original_image.shape[1]:
            padh = (original_image.shape[0] - original_image.shape[1])//2
        if original_image.shape[1] > original_image.shape[0]:
            padw = (original_image.shape[1] - original_image.shape[0])//2
        pad_image = np.pad(original_image, [(padw, padw), (padh,padh), (0,0)], 'constant')
        w = pad_image.shape[0]
        h = pad_image.shape[1]
        #xdata = b'P6 ' + str(w).encode() + b' ' + str(h).encode() + b' 255 ' + pad_image.tobytes()
        #tk_image = self.tk.PhotoImage(data=xdata, format="PPM", width=w, height=h)
        #self.root.tk.call('wm', 'iconphoto', self.root._w, tk_image.subsample(max(1, w//256), max(1, h//256)))

        surface = self.pg.Surface([image.shape[0],image.shape[1]])
        self.pg.surfarray.blit_array(surface, image[:,:,:3])
        self.screen.blit(self.pg.transform.scale(surface,self.size),(0,0))
        self.pg.display.flip()

    def tick(self):
        """
            Called repeatedly regardless of gan state.
        """
        if hasattr(self, 'root'):
            self.root.update()

       
