# Compatibility shim: expose the EWC optimizer at
# `hypergan.optimizers.elastic_weight_consolidation_optimizer` so configs
# that reference that module (older configs) continue to work.
#
# Delay importing the heavy TensorFlow-backed implementation until the
# optimizer is instantiated so importing this module doesn't require
# TensorFlow to be installed (helps in test environments / CPU-only setups).

class ElasticWeightConsolidationOptimizer:
    """Lazy proxy for the TF-backed ElasticWeightConsolidationOptimizer.

    On instantiation, this will import the real implementation from
    `hypergan.optimizers.needs_pytorch.elastic_weight_consolidation_optimizer`.
    If the import fails (e.g., TensorFlow not installed), a clear
    ModuleNotFoundError is raised with guidance.
    """
    def __init__(self, *args, **kwargs):
        try:
            from .needs_pytorch.elastic_weight_consolidation_optimizer import (
                ElasticWeightConsolidationOptimizer as _RealEWC,
            )
        except ModuleNotFoundError as e:
            raise ModuleNotFoundError(
                "ElasticWeightConsolidationOptimizer requires TensorFlow (or other optional deps). "
                "Install the optional dependencies or use a different optimizer. Original error: {}".format(e)
            )
        self._wrapped = _RealEWC(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self._wrapped, name)


__all__ = ["ElasticWeightConsolidationOptimizer"]
