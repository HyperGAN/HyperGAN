import inflection
import glob
import os
import hypergan as hg

layers = {}
for k, v in hg.layers.__dict__.items():
    if isinstance(v, type) and issubclass(v, hg.layer.Layer):
        layers[v] = v.__doc__

dir_path = os.path.dirname(os.path.realpath(__file__))
layer_defn_list = []
for key in sorted(layers, key=lambda x: str(x)):
    name = inflection.underscore(key.__name__)
    fname = name+".md"
    path = "components/layers/"+fname
    layer_defn_list.append((name, path))
    with open(dir_path + "/" + path, "w") as f:
        if layers[key]:
            f.write(layers[key])
        else:
            f.write("No documentation exists\n")

with open(dir_path+"/SUMMARY.md", "w") as f:
    with open(dir_path+"/SUMMARY.md.template") as read:
        for line in read.readlines():
            if "LAYER_DEFINITION_LIST" in line:
                for name, path in layer_defn_list:
                    f.write("  *  ["+name+"]("+path+")\n")
            else:
                f.write(line)
print("Done")
