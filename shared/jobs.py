import websocket

# Sets up a websocket hc.io client to run in the background, listening for job requests
def create_connection():
    ws.connect("ws://example.com/websocket", http_proxy_host="proxy_host_name", http_proxy_port=3128)
    ws = websocket.WebSocket()

def process(sess):
    job = pop()
    while(job):
        if job.name == 'sample':
            sample()
        else:
            print("Unknown job")

        job = pop()


    
