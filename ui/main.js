const electron = require('electron');
const {app, BrowserWindow, Menu} = require('electron');
const isDev = require('electron-is-dev');

let mainWindow;

app.on('window-all-closed', () => {
  if (process.platform != 'darwin')
    app.quit();
});

app.on('ready', function() {
  mainWindow = new BrowserWindow({width: 800, height: 600, webPreferences: {nodeIntegration: true}});

  const startURL = isDev ? 'http://localhost:3000' : ('file://' + __dirname + '/build/index.html');
  mainWindow.loadURL(startURL);

  var application_menu = [
    {
      label: 'File',
      submenu: [
        {
          label: 'Save',
          accelerator: 'CmdOrCtrl+S',
          click: () => {
            mainWindow.closeDevTools();
          }
        },
        {
          label: 'Save and exit',
          accelerator: 'CmdOrCtrl+Q',
          click: () => {
            mainWindow.closeDevTools();
          }
        }
      ]
    },
    {
      label: 'Sample',
      submenu: [
        {
          label: 'Refresh',
          accelerator: 'CmdOrCtrl+R',
          click: () => {
            mainWindow.closeDevTools();
          }
        },
        {
          label: 'Sampler',
          submenu: [
            {
              label: 'Batch Walk',
              click: () => {
                mainWindow.openDevTools();
              }
            }
          ]
        }
      ]
    },

  ];

  menu = Menu.buildFromTemplate(application_menu);
  Menu.setApplicationMenu(menu);

  mainWindow.on('closed', function() {
    mainWindow = null;
  });
});
