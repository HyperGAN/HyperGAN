import Client from './api/Client';
export default class ServerDiscovery {

  constructor() {
    this.clients = [];
    this.onConnect = (client) => {};
  }

  emitConnect(client) {
    this.onConnect(client);
  }

  discover(port=9999) {
    let socket = new WebSocket(`ws://localhost:${port}`);
    let client = new Client(socket);

    socket.onclose = () => {
      console.log(`No server found on port ${port}, stopping scan`)
    }

    socket.onopen = () => {
      client.connect();
      this.discover(port+1);
    }

    client.onConnect = () => {
      this.emitConnect(client);
    }
  }
}
