"""Every optional extra must install versions that the runtime version checks accept."""
from pathlib import Path
import tomllib

from packaging.requirements import Requirement

from hypergan.image_metrics import TORCH_FIDELITY_VERSION

PYPROJECT = Path(__file__).resolve().parents[2] / 'pyproject.toml'
# Distributions whose version the runtime checks, mapped to the versions it accepts
# and to every release published so far, so a range cannot hide a refused one.
RUNTIME_CHECKS = {
    'torch-fidelity': ({TORCH_FIDELITY_VERSION}, ['0.1.0', '0.2.0', '0.3.0', '0.4.0']),
}


def test_every_extra_pin_satisfies_the_runtime_version_checks():
    extras = tomllib.loads(PYPROJECT.read_text())['project']['optional-dependencies']
    checked = set()
    for extra, requirements in extras.items():
        for text in requirements:
            requirement = Requirement(text)
            if requirement.name not in RUNTIME_CHECKS:
                continue
            accepted, releases = RUNTIME_CHECKS[requirement.name]
            installable = set(requirement.specifier.filter(releases + sorted(accepted)))
            # pip may pick any installable release; the runtime must accept each.
            assert installable and installable <= accepted, (extra, text, installable - accepted)
            checked.add(requirement.name)
    assert checked == set(RUNTIME_CHECKS)
