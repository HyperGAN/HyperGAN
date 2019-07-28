# An example of how to run all the configuration files in the current directory, in parallel.
# Usage: python3 run_all.py

import logging
import random
import threading
import time
import queue
import glob
import os
import json

logging.basicConfig(level=logging.DEBUG,
                    format='(%(threadName)-10s) %(message)s',
                    )
class Trainer(object):
    def __init__(self, card=0):
        self.lock = threading.Lock()
        self.card = card
        self.steps = 100000
        #number_samples = 60*25 # one minute video
        number_samples = 10*25 # ten second video
        self.sample_every = self.steps//number_samples
        self.format = 'jpg'
        self.dataset = "/ml/datasets/faces/128x128_sketched/all"
        self.batch_size = 8
    def run(self, json_file):
        obj = json.loads(open(json_file+".json", 'r').read())
        if "width" in obj["runtime"]:
            size = " --size "+str(obj["runtime"]["width"])+"x"+ str(obj["runtime"]["height"])+"x"+str(obj["runtime"]["channels"])
        else:
            print("NULL SIZE", obj)
            size = ""
        command = obj["runtime"]["train"]
        
        logging.debug('Run %s on card %d', json_file, self.card)
        if os.path.exists("samples/"+json_file):
            logging.info('Skipping '+json_file+', samples exist')
        else:
            command = ("CUDA_VISIBLE_DEVICES=%d " % self.card) + command
            command = command.replace("[dataset]", self.dataset)
            command = command.replace("[hypergan]", "/ml/dev/hypergan/")
            command += " --sample_every %d -b %d --format %s -c %s --save_every %d --steps %d --save_samples --resize" % (self.sample_every, self.batch_size, self.format, json_file, self.steps-1, self.steps)
            command += size
            logging.debug(command)
            os.system(command)

def worker(c):
    thread = threading.currentThread()
    if queue.empty():
        return
    json_file = queue.get()
    config = json_file.replace(".json","")
    c.run(config)
    worker(c)
    queue.task_done()
    logging.debug('Done')


gpus = []
gpu_count = 2
gpu_offset = 0
per_gpu = 2
queue = queue.Queue()


files = glob.glob("*.json")

for f in files:
    queue.put(f)

for j in range(per_gpu):
    for i in range(gpu_count):
        gpu = Trainer(i+gpu_offset)

        t=threading.Thread(target=worker, args=(gpu,))
        gpus.append(t)
        t.start()

logging.debug('Waiting for worker threads')
main_thread = threading.currentThread()

queue.join()

for t in threading.enumerate():
    if t is not main_thread:
        t.join()

logging.debug("Goodbye")
