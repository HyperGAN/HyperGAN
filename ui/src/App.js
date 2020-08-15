import React from 'react';
import logo from './logo.svg';
import './App.css';
var socket;
//import HgApiClient from './HgApiClient.js';

function App() {
  const [messages, setMessages] = React.useState([]);
  const addMessage = (msg) => {
    console.log(messages);
    setMessages([...messages, msg])
  }
  return (
    <div className="App">
    <header className="App-header">
    <img src={logo} className="App-logo" alt="logo" />
    <p>
    Listening for HyperGAN
      <button onClick={_=>{
        socket = new WebSocket("ws://localhost:9999")

        socket.onclose = () => {
          addMessage("reconnecting");
        };
        socket.onmessage = (message) => {
          addMessage(message.data);
        };
        socket.onopen = () => {
          addMessage("connected");
        };
        }}>
            Connect
          </button>
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
