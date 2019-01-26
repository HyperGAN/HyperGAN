"""
Opens a window that displays an image.
Usage:

    from viewer import GlobalViewer
    GlobalViewer.update(image)

"""
import numpy as np
import os
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

    def enforce_aspect_ratio(self, event):
        desired_width = event.width
        desired_height = int(event.width / self.aspect_ratio)

        if desired_height > event.height:
            desired_height = event.height
            desired_width = int(event.height * self.aspect_ratio)

        self.config(width=desired_width, height=desired_height)
        self.tkviewer.size = [desired_width, desired_height]
        self.tkviewer.screen = self.tkviewer.pg.display.set_mode(self.tkviewer.size,self.tkviewer.pg.RESIZABLE)   

class TkViewer:
    def __init__(self, title="HyperGAN", viewer_size=1, enabled=True):
        self.screen = None
        self.title = title
        self.viewer_size = viewer_size
        self.enabled = enabled
        self.enable_menu = True

    def update(self, gan, image):
        if not self.enabled: return

        if len(np.shape(image)) == 2:
            s = np.shape(image)
            image = np.reshape(image, [s[0], s[1], 1])
            image = np.tile(image, [1,1,3])
        image = np.transpose(image, [1, 0,2])

        if not self.screen:
            import pygame

            self.size = [int(image.shape[0] * self.viewer_size), int(image.shape[1] * self.viewer_size)]

            self.pg = pygame
            root = tk.Tk()
            embed = ResizableFrame(root, width=self.size[0], height=self.size[1], tkviewer=self)
            embed.winfo_toplevel().title(self.title)
            root.rowconfigure(0,weight=1)
            root.rowconfigure(1,weight=1)
            root.columnconfigure(0,weight=1)
            root.columnconfigure(1,weight=1)
            embed.pack(expand=tk.YES, fill=tk.BOTH)

            def _save_model(*args):
                gan.save(gan.save_file)

            def _exit(*args):
                gan.exit()

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
            filemenu.add_command(label="Save", command=_save_model, underline=0, accelerator="Ctrl+S")

            filemenu.add_separator()

            filemenu.add_command(label="Save and Exit", command=_exit, underline=10, accelerator="Ctrl+Q")
            menubar.add_cascade(label="File", menu=filemenu, underline=0)

            root.bind_all("<Control-q>", _exit)
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

        for event in self.pg.event.get():
            if event.type == self.pg.VIDEORESIZE:
                if self.size[0] != event.size[0]:
                    self.temp_size = [event.size[0], int(event.size[0] * self.aspect_w)]
                elif self.size[1] != event.size[1]:
                    self.temp_size = [int(event.size[1] * self.aspect_h), event.size[1]]

            elif event.type == self.pg.ACTIVEEVENT and event.state == 2 and event.gain == 1:
                self.size = self.temp_size
                self.screen = self.pg.display.set_mode(self.size,self.pg.RESIZABLE)   


        surface = self.pg.Surface([image.shape[0],image.shape[1]])
        self.pg.surfarray.blit_array(surface, image)
        self.screen.blit(self.pg.transform.scale(surface,self.size),(0,0))
        self.pg.display.flip()

    def tick(self):
        """
            Called repeatedly regardless of gan state.
        """
        if hasattr(self, 'root'):
            self.root.update()

       
