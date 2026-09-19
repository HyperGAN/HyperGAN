/* Dedicated-worker owner: a deadline terminates work without blocking the UI. */
export class ReducerWorker {
  constructor({timeoutMs = 10000, workerURL = new URL('./worker.js', import.meta.url)} = {}) {
    if (!Number.isInteger(timeoutMs) || timeoutMs < 1 || timeoutMs > 60000) throw new Error('Invalid worker timeout');
    this.worker = new Worker(workerURL, {type:'module'});
    this.timeoutMs = timeoutMs; this.nextID = 0; this.pending = null; this.closed = false;
    this.worker.onmessage = ({data}) => {
      if (!this.pending || data.id !== this.pending.id) return;
      const {resolve,reject,timer} = this.pending;
      clearTimeout(timer); this.pending = null;
      if ('error' in data) reject(new Error(data.error)); else resolve(data.ok);
    };
    this.worker.onerror = event => this.close(new Error(event.message || 'Reducer worker failed'));
  }
  call(message) {
    if (this.closed) return Promise.reject(new Error('Reducer worker is closed'));
    if (this.pending) return Promise.reject(new Error('Await the current reducer request before submitting another'));
    const id = ++this.nextID;
    return new Promise((resolve,reject) => {
      const timer = setTimeout(() => this.close(new Error('Reducer worker deadline exceeded')), this.timeoutMs);
      this.pending = {id,resolve,reject,timer};
      try { this.worker.postMessage({...message,id}); }
      catch (error) { this.close(error); }
    });
  }
  close(error = new Error('Reducer worker closed')) {
    this.closed = true; this.worker.terminate();
    if (this.pending) { clearTimeout(this.pending.timer); this.pending.reject(error); this.pending = null; }
  }
}
