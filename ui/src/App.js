import React from 'react';
import logo from './logo.svg';
import './App.css';
var socket;
//import HgApiClient from './HgApiClient.js';

function App() {
  const [messages, setMessages] = React.useState([]);
  function addMessage() {
    setMessages([...messages.slice(-3), "reconnecting"])
  }
  function connect() {
    socket = new WebSocket("ws://localhost:9999")
    socket.onclose = () => {
      addMessage("reconnecting");
      setTimeout(() => {
        connect();
      }, 10000);
    };
    socket.onopen = () => {
      addMessage("connected");
    };
    socket.onmessage = (message) => {
      addMessage(message.data);
    };
  }
  connect();
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
