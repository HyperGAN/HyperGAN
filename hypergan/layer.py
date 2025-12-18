"""Layer base class with a resilient import of torch.nn.

Some CI/dev environments have `torch` installed but missing CUDA system
libraries which cause `import torch` to raise during module import.
To allow running tests in such environments (where the code paths that
actually use CUDA won't be exercised), we provide a lightweight fallback
`nn.Module` so the package can import successfully.
"""

try:
    import torch.nn as nn
except Exception:
    # Fallback minimal implementation so importing hypergan doesn't fail
    # when system CUDA libs are missing (e.g., in lightweight test envs).
    class _DummyModule(object):
        def __init__(self, *args, **kwargs):
            pass

    class _DummyNN:
        Module = _DummyModule

    nn = _DummyNN()


class Layer(nn.Module):
    def __init__(self, component, args, options):
        super(Layer, self).__init__()
        self.args = args
        self.options = options

    def forward(self, input, context):
        pass

    def output_size(self):
        pass

    def latent_parameters(self):
        return []
