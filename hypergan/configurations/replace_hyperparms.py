from subprocess import call

import glob

filenames = glob.glob("*.json")

for f in filenames:
    cmd = "jq '.loss.gradient_locally_stable=0.01' "+f+" | awk 'BEGIN{RS=\"\";getline<\"-\";print>ARGV[1]}' "+f
    print(cmd)
    
    result = call(cmd, shell=True)



