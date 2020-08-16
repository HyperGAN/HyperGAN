export default class Client {
  constructor(socket) {
    this.onConnect = _ => {}
    this.socket = socket;
    this.socket.onmessage = (message) => {
      let json = JSON.parse(message.data);
      console.log("Got message");
      if(json["action"] == "connect") {
        this.onConnect();
      }
    }
  }

  connect() {
    this.socket.send(JSON.stringify({ "action": "connect" }));
  }

  sample() {
    this.socket.send(JSON.stringify({ "action": "get_samples" }));
  }
}
