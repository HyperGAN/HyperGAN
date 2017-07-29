from subprocess import call

import glob

dirnames = glob.glob("samples/*")

for d in dirnames:
    images = glob.glob(d+"/*.png")

    print("")
    print("### "+d)
    images.sort()
    for image in images:
        print("")
        print("!["+image+"]("+image+")")


