from subprocess import call

import glob

filenames = glob.glob("*.json")

cmd="rm -rf samples"
result = call(cmd, shell=True)

for f in filenames:
    config = f.split(".")[0]

    cmd="CUDA_VISIBLE_DEVICES=0 hypergan train /ml/datasets/faces/128x128/all --sample_every 8000 --sampler debug --format jpg --size 64x64x3 -b 8 -c ./"+config+" --resize --save_samples --steps 16001 --save_every 1000000"
    print(cmd)
    result = call(cmd, shell=True)

    cmd="mkdir samples/"+config
    result = call(cmd, shell=True)

    cmd="mv samples/*.png samples/"+config
    result = call(cmd, shell=True)


