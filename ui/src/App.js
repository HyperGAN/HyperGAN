import React from 'react';
import logo from './logo.svg';
import ServerDiscovery from './ServerDiscovery';
import './App.css';
//import HgApiClient from './HgApiClient.js';

function App() {
  const [messages, setMessages] = React.useState([]);
  const [clients, setClients] = React.useState([]);
  const [samples, setSamples] = React.useState([]);
  const serverDiscovery = new ServerDiscovery();
  const refresh = () => {
    serverDiscovery.discover();
  }
  serverDiscovery.onClientsDiscovered = (clients) => {
    console.log("New client: ", clients);
    setClients(clients);
    clients.map( (client) => {
      client.sample();
      client.onSamples = (samples) => {
        console.log("Got a bunch of fresh samples", samples[0]);
        setSamples(samples);
      }
    });
  }
  return (
    <div className="App">
    <header className="App-header">
    <img src={logo} className="App-logo" alt="logo" />
    {
      samples.map((sample,index) => <img src={URL.createObjectURL(sample)}/>)
    }
    <p>

      <button onClick={_=>{
        refresh();
      }}>
          Refresh
          </button>
        </p>
        Clients connected:
          {clients.length}
      </header>
    </div>
  );
}

export default App;
