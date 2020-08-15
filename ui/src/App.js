import React from 'react';
import logo from './logo.svg';
import './App.css';
const socket = new WebSocket("ws://localhost:9999")
//import HgApiClient from './HgApiClient.js';

function App() {
  const [messages, setMessages] = React.useState([]);
  socket.onopen = () => {
    setMessages([...messages, "connected"])
  };
  socket.onmessage = (message) => {
    setMessages([...messages, message.data])
  };
  return (
    <div className="App">
      <header className="App-header">
        <img src={logo} className="App-logo" alt="logo" />
        <p>
          Listening for HyperGAN
          <button onClick={_=>{
            socket.send("ping")
          }}>
          Ping
          </button>
          {messages.map((msg,index) => <li>{msg}</li>)}
        </p>
      </header>
    </div>
  );
}

export default App;
