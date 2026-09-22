# Algorithm contract

Add one local Python module per algorithm version. A tuning solution references
it by a path relative to the solution JSON. Required entry point:

```python
def propose(context):
    config = context['config']       # copied resolved training configuration
    options = context['options']     # solution-specific options
    probes = context['evidence']     # explicitly referenced probe JSON reports
    # Derive a bounded proposal; verify the assumptions using probe evidence.
    return {'schema_version': 1, 'evidence': {'decision': 'abstain'}}
```

An empty intervention explicitly leaves the configuration unchanged. Record why
the algorithm abstained. Do not hide negative curvature or unresolved evidence
with an arbitrary LR floor. Accepted intervention fields are `g_lr`, `d_lr`,
`init_scales`, and `layer_lr_multipliers`; the latter two contain explicit
pattern/multiplier rules. The shared evaluator validates ownership and applies
the proposal once.

Keep probe creation separate from final outcome evaluation. Include probe
budget and elapsed time in evidence; loading precomputed probes does not make
their cost free. The current contract accepts offline probes, not live nested
training or an unbounded search. Record actual formula, assumptions and sanity
checks in the returned `evidence`. Do not consume the benchmark monitor bank
to fit the proposal and then present it as independent validation.

`grouped_first_g.py` implements [the grouped first-generator contract](grouped-first-g-contract.md).
It abstains unless every predeclared gate passes. It does not choose a replacement
group or a rate floor.
