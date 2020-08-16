import React from 'react';
import logo from './logo.svg';
import ServerDiscovery from './ServerDiscovery';
import './App.css';
//import HgApiClient from './HgApiClient.js';

function App() {
  const [messages, setMessages] = React.useState([]);
  const [clients, setClients] = React.useState([]);
  const serverDiscovery = new ServerDiscovery();
  const addMessage = (msg) => {
    console.log(messages);
    setMessages([...messages, msg])
  }
  serverDiscovery.onConnect = (client) => {
    console.log("New client: ", client);
    addMessage(`Connected to client ${client.socket.url}`);
    setClients([...clients, client]);
    client.sample();
  }
  return (
    <div className="App">
    <header className="App-header">
    <img src={logo} className="App-logo" alt="logo" />
    <p>
      <button onClick={_=>{
          serverDiscovery.discover();
      }}>
          Discover
          </button>
          {messages.map((msg,index) => <li>{msg}</li>)}
        </p>
        Clients connected:
          {clients.length}
      </header>
    </div>
  );
}

export default App;
