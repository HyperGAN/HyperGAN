export default class Client {
  constructor(socket) {
    this.onConnect = _ => {}
    this.onSamples = _ => {}
    this.samples = [];
    this.socket = socket;
    this.socket.onmessage = (message) => {
      console.log(message);
      if(message.data instanceof Blob) {
        this.sample_data = message.data;
        return
      }
      let json = JSON.parse(message.data);
      if(json["action"] == "connect") {
        this.onConnect();
      }
      if(json["action"] == "sample_start") {
      }
      if(json["action"] == "sample_end") {
        this.samples.push(this.sample_data.slice(0, this.sample_data.size, "image/png")); //slice sets the mime type
      }
      if(json["action"] == "samples_start") {
        this.samples = [];
      }
      if(json["action"] == "samples_end") {
        if(this.samples.length > 0) {
          console.log("Sending samples", this.samples)
          this.onSamples(this.samples);
        }
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
