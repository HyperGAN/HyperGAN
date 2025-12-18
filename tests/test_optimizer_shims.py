def test_ewc_shim_importable():
    import importlib

    m = importlib.import_module('hypergan.optimizers.elastic_weight_consolidation_optimizer')
    assert hasattr(m, 'ElasticWeightConsolidationOptimizer')

    cls = m.ElasticWeightConsolidationOptimizer
    # Instantiation should raise ModuleNotFoundError in environments without TF;
    # ensure the error message is helpful.
    try:
        cls()
    except ModuleNotFoundError as e:
        msg = str(e)
        assert 'TensorFlow' in msg or 'tensorflow' in msg
