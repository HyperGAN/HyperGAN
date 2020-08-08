import hyperchamber as hc
import os
import glob

class Configuration:
    def all_paths(prepackaged):
        paths = [ os.path.abspath(os.path.relpath("."))+"/" ]
        if prepackaged:
            paths = [
                     os.path.dirname(os.path.realpath(__file__)) + "/configurations/",
                     os.path.abspath(os.path.expanduser('~/.hypergan/configs/'))+'/'
                    ]
        return paths
    def find(configuration, verbose=True, config_format='.json', prepackaged=False):
        def _find_file():
            paths = Configuration.all_paths(prepackaged)
            Configuration.paths = paths
            for path in paths:
                file_path = path + configuration
                file_path = os.path.realpath(file_path)
                if os.path.exists(file_path):
                    return file_path

        if not configuration.endswith(config_format):
            configuration += config_format
        config_filename = _find_file()
        if config_filename is None:
            message = "Could not find configuration " + configuration
            raise Exception(message)
        if verbose:
            print("Loading configuration", config_filename)
        return config_filename

    def load(configuration, verbose=True, use_toml=False, prepackaged=False):
        config_file = Configuration.find(configuration, verbose=verbose, prepackaged=prepackaged)
        if config_file is None:
            print("[hypergan] Could not find config named:", configuration, "checked paths", Configuration.paths)
        if verbose:
          print("[hypergan] Loading config", config_file)
        return hc.Selector().load(config_file, load_toml=use_toml)
    def default():
        return Configuration.load('default.json')
    def list(config_format='.json', prepackaged=False):
        paths = Configuration.all_paths(prepackaged)
        return sorted(sum([[x.split("/")[-1].split(".")[0] for x in glob.glob(path+"/*"+config_format)] for path in paths], []))

