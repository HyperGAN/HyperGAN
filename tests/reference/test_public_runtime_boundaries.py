"""Numerical API/CLI integration belongs to the installed training-extra suite."""
import json
from pathlib import Path
import subprocess
import sys

import pytest

import hypergan.api as hg

ROOT = Path(__file__).resolve().parents[2]
EXAMPLES = ROOT / 'examples' / 'api' / 'config_first'

@pytest.fixture(autouse=True)
def project_on_path(monkeypatch):
    monkeypatch.syspath_prepend(str(EXAMPLES))

def small(model, steps=3):
    return hg.override(model, {'train.steps': steps})

def run_cli(tmp_path, *args, isolated=True):
    cmd = [sys.executable]
    if isolated:
        cmd.append('-I')
    cmd.extend(['-m', 'hypergan', *map(str, args)])
    return subprocess.run(cmd, cwd=tmp_path, text=True, capture_output=True, timeout=30)


def test_lift_all_examples():
    example_files = list((ROOT / 'examples').glob('*.toml'))
    assert len(example_files) >= 20
    for path in example_files:
        recipe = hg.load_recipe(path)
        lifted = hg.lift(recipe, base=path.parent)
        assert lifted.spec['format'] == 'hypergan-model/1'
        warnings = hg.validate(lifted)
        assert isinstance(warnings, list)


def test_api_checkpoints_and_export(tmp_path):
    import json
    import torch
    run_dir = tmp_path / "run"
    ckpt_dir = run_dir / "checkpoints" / "att-1-step-00000100-uuid1"
    ckpt_dir.mkdir(parents=True)
    state = {
        "step": 100,
        "ema_graph": {"models.generator.weight": torch.randn(2, 2)},
        "ema_prior": {"prior": torch.randn(2)},
        "buffers": {"ema_graph": {}, "ema_prior": {}},
        "last_batch": {},
    }
    torch.save(state, ckpt_dir / "state.pt")
    (ckpt_dir / "manifest.json").write_text(json.dumps({
        "step": 100,
        "config": {"components": {"generator": {"factory": "linear", "inputs": {}}}},
    }), encoding="utf-8")

    # 1. resolve_checkpoint
    resolved = hg.resolve_checkpoint(run_dir, "latest")
    assert resolved == ckpt_dir

    # 2. extract_ema_bundle
    bundle = hg.extract_ema_bundle(ckpt_dir)
    assert bundle["kind"] == "ema-inference"
    assert bundle["step"] == 100

    # 3. export
    out_pt = hg.export(run_dir, checkpoint="latest")
    assert out_pt.is_file()
    assert out_pt.name == "model.pt"
    assert (out_pt.parent / "model.sha256").is_file()

    # 4. prune_checkpoints
    report = hg.prune_checkpoints(run_dir, dry_run=True)
    assert report["dry_run"] is True
    assert report["retained_count"] == 1


def test_api_load_inference(tmp_path):
    import json
    import torch
    from hypergan.config import resolve_config
    from hypergan.recipes import ComponentGraph, make_prior

    run_dir = tmp_path / "run"
    ckpt_dir = run_dir / "checkpoints" / "att-1-step-00000100-uuid1"
    ckpt_dir.mkdir(parents=True)
    cfg = resolve_config({
        "components": {
            "generator": {"factory": "linear", "args": {"in_features": 2, "out_features": 2}, "inputs": {"z": "latent"}},
            "discriminator": {"factory": "linear", "args": {"in_features": 2, "out_features": 1}, "inputs": {"x": "candidate"}},
        }
    })
    cg = ComponentGraph(cfg["components"])
    gen_sd = {f"models.generator.{k}": v for k, v in cg.models["generator"].state_dict().items()}
    prior = make_prior(cfg["prior"], device="cpu")
    state = {
        "step": 100,
        "ema_graph": gen_sd,
        "ema_prior": prior.state_dict(),
        "buffers": {"ema_graph": {}, "ema_prior": {}},
        "last_batch": {},
    }
    torch.save(state, ckpt_dir / "state.pt")
    (ckpt_dir / "manifest.json").write_text(json.dumps({
        "step": 100,
        "config": cfg,
    }), encoding="utf-8")

    out_pt = hg.export(run_dir, checkpoint="latest")
    loaded_state, loaded_cfg, graph, loaded_prior, sha = hg.load_inference(out_pt.parent)
    assert loaded_state["kind"] == "ema-inference"
    assert loaded_state["step"] == 100
    assert graph is not None
    assert loaded_prior is not None
    assert sha is not None


def test_item_data_is_seeded_resumable_and_rolls_back():
    import torch
    from hypergan.item_data import ItemData
    data = ItemData('toy_project:GridPoints', {'side': 3, 'size': 10})
    rng = torch.Generator().manual_seed(1)
    first = data(4, generator=rng)['real']
    state, rng_state = data.state_dict(), rng.get_state()
    expected = torch.cat([data(4, generator=rng)['real'] for _ in range(3)])   # crosses an epoch boundary
    again = ItemData('toy_project:GridPoints', {'side': 3, 'size': 10})
    again.load_state_dict(state)
    rng.set_state(rng_state)
    assert torch.equal(torch.cat([again(4, generator=rng)['real'] for _ in range(3)]), expected)
    assert first.shape == (4, 2) and data.state_dict()['epoch'] == 2
    other = ItemData('toy_project:GridPoints', {'side': 3, 'size': 11})
    with pytest.raises(ValueError, match='identity'):
        other.load_state_dict(state)

    class Broken:
        def __len__(self):
            return 4

        def __getitem__(self, index):
            if index == 3:
                raise OSError('unreadable')
            return {'real': torch.zeros(2), 'meta': {'id': index, 'pair': [torch.ones(1), 2.0]}}
    import toy_project
    toy_project.Broken = Broken
    broken = ItemData('toy_project:Broken', shuffle=False)
    batch = broken(2, generator=rng)
    assert batch['meta']['id'].tolist() == [0, 1] and batch['meta']['pair'][0].shape == (2, 1)
    before = broken.state_dict()
    with pytest.raises(OSError):
        broken(2, generator=rng)
    assert broken.state_dict() == before


def test_plain_function_metric_and_evaluation_adapters():
    import torch
    from hypergan.plugin_functions import ScalarFunction, SnapshotFunction
    metric = ScalarFunction('toy_project:d_over_g', direction='none')
    assert metric.describe()['kind'] == 'scalar' and 'source_sha256=' in metric.describe()['description']
    assert metric.evaluate(context={}, d=1.0, g=4.0) == 0.25
    evaluation = SnapshotFunction('toy_project:modes_covered', args={'side': 2}, direction='maximize')
    corners = torch.tensor([[-1.0, -1.0], [1.0, 1.0]])
    batches = iter([{'generated': corners[:1], 'reference': corners[:1]}, {'generated': corners[1:], 'reference': corners[1:]}])
    assert evaluation.evaluate(batches=batches, context={}) == 0.5


def test_train_simple_model_and_read_metrics(tmp_path):
    model = small(hg.load(EXAMPLES / 'simple.toml'))
    run = hg.train(model, tmp_path / 'run')
    assert (run.status, run.step, run.fingerprint) == ('complete', 3, hg.fingerprint(model))
    series = hg.metrics(run)
    assert [step for step, _ in series['loss/d_total']] == [1, 2, 3]
    assert set(hg.metrics(run, ['loss/g_total'])) == {'loss/g_total'}
    fresh = hg.sample(run, count=5, seed=1)
    assert tuple(fresh.data.shape) == (5, 2) and fresh.step == 3
    image = hg.view(fresh, 'toy_project:scatter', run=run)
    assert (image.width, image.height, image.channels) == (64, 64, 1)
    assert image.png().startswith(b'\x89PNG')
    assert not any(p.suffix in ('.py', '.hndl') for p in run.path.rglob('*'))   # no user code copied


def test_item_data_run_stops_and_resumes_at_the_same_position(tmp_path):
    model = small(hg.override(hg.load(EXAMPLES / 'simple.toml'),
                              {'data': {'dataset': 'toy_project:GridPoints', 'args': {'size': 100}}}), steps=4)
    straight = hg.train(model, tmp_path / 'straight')
    hg.train(model, tmp_path / 'split', stop_after_steps=2)
    split = hg.train(model, tmp_path / 'split')          # same model file, same run dir: continue
    assert split.step == 4
    left, right = hg.metrics(straight)['loss/d_total'], hg.metrics(split)['loss/d_total']
    assert [v for s, v in left if s > 2] == pytest.approx([v for s, v in right if s > 2][-2:])


def test_multi_network_model_trains(tmp_path):
    run = hg.train(small(hg.load(EXAMPLES / 'two_critics.toml')), tmp_path / 'run')
    series = hg.metrics(run)
    assert run.step == 3 and 'loss/objectives/reconstruction' in series
    assert run.config['adversarial_terms'][0]['component'] == 'marginal_critic'


@pytest.mark.heavy
def test_extensions_run_metrics_evaluations_previews_and_samplers(tmp_path):
    import demo_3_extensions
    run = demo_3_extensions.main(tmp_path / 'run')
    assert 'd_over_g' in hg.metrics(run)
    results = hg.evaluations(run)
    assert results['nearest'][0]['status'] == 'complete' and results['modes'][0]['step'] == 10
    previews = hg.samples(run)
    assert previews and tuple(previews[0].data.shape) == (16, 2)
    assert hg.view(previews[0], 'scatter', run=run).width == 64


def test_cli_resume_without_config_applies_set_overrides(tmp_path):
    config = tmp_path / "model.toml"
    run_dir = tmp_path / "run"
    run_cli(tmp_path, "new", config, "--device", "cpu")
    train_res = run_cli(tmp_path, "train", config, "--run-dir", run_dir, "--steps", "1", "--set", "train.lr_floor=1.0", "--no-server", isolated=False)
    assert train_res.returncode == 0, train_res.stderr

    resume_res = run_cli(tmp_path, "resume", run_dir, "--set", "train.steps=2", "--no-server", isolated=False)
    assert resume_res.returncode == 0, resume_res.stderr
    data = json.loads(resume_res.stdout)
    assert data["status"] == "complete"
    assert data["steps"] == 2


def test_cli_resume_with_malformed_set_fails_cleanly(tmp_path):
    config = tmp_path / "model.toml"
    run_dir = tmp_path / "run"
    run_cli(tmp_path, "new", config, "--device", "cpu")
    run_cli(tmp_path, "train", config, "--run-dir", run_dir, "--steps", "1", "--set", "train.lr_floor=1.0", "--no-server", isolated=False)

    result = run_cli(tmp_path, "resume", run_dir, "--set", "bad_set", "--no-server", isolated=False)
    assert result.returncode != 0
    assert "error:" in result.stderr
    assert "Invalid --set override 'bad_set'; expected KEY=VALUE" in result.stderr
    assert "Traceback" not in result.stderr


def test_cli_train_optional_run_dir_and_provenance(tmp_path):
    ws_file = tmp_path / "hypergan.toml"
    runs_dir = tmp_path / "custom_runs"
    ws_file.write_text(f"""
[workspace]
root = "{tmp_path}"
runs_root = "{runs_dir}"
""")
    config = tmp_path / "my_experiment.toml"
    run_cli(tmp_path, "new", config, "--device", "cpu")

    # Train without --run-dir; should default to {runs_root}/{stem}-{adj}-{noun}
    train_res = run_cli(tmp_path, "train", config, "--steps", "1", "--no-server", isolated=False)
    assert train_res.returncode == 0, train_res.stderr

    created_runs = [d for d in runs_dir.iterdir() if d.is_dir() and d.name.startswith("my_experiment-")]
    assert len(created_runs) == 1
    expected_run = created_runs[0]
    assert len(expected_run.name.split("-")) >= 3
    assert (expected_run / "manifest.json").is_file()

    # Verify launch provenance and config snapshot were written
    launch_json = expected_run / "launch" / "0001.json"
    assert launch_json.is_file()
    launch_data = json.loads(launch_json.read_text(encoding="utf-8"))
    assert launch_data["attempt"] == 1
    assert isinstance(launch_data.get("pid"), int)
    assert "hypergan" in launch_data["git"]

    config_source = expected_run / "config-source"
    assert config_source.is_dir()
    assert (config_source / "my_experiment.toml").is_file()
    assert (config_source / "hypergan.toml").is_file()


def test_resume_alias_of_train_with_run_dir(tmp_path):
    config = tmp_path / "model.toml"
    run_dir = tmp_path / "run_resume_alias"

    run_cli(tmp_path, "new", config, "--device", "cpu")
    t1 = run_cli(tmp_path, "train", config, "--run-dir", run_dir, "--steps", "1",
                 "--set", "train.lr_floor=1.0", "--no-server", isolated=False)
    assert t1.returncode == 0, t1.stderr

    # Resume using `hypergan train RUN_DIR` with --progress-json
    t2 = run_cli(tmp_path, "train", run_dir, "--steps", "2", "--progress-json", "--no-server", isolated=False)
    assert t2.returncode == 0, t2.stderr
    last_line = t2.stdout.strip().splitlines()[-1]
    res_event = json.loads(last_line)
    d2 = res_event.get("manifest", res_event)
    assert d2["steps"] == 2
    assert d2["status"] == "complete"

    # Resume using `hypergan resume RUN_DIR`
    t3 = run_cli(tmp_path, "resume", run_dir, "--steps", "3", "--no-server", isolated=False)
    assert t3.returncode == 0, t3.stderr
    d3 = json.loads(t3.stdout)
    assert d3["steps"] == 3
    assert d3["status"] == "complete"


def test_train_cli_human_summary_footer_and_progress_json(tmp_path):
    config = tmp_path / "model.toml"
    run_cli(tmp_path, "new", config, "--device", "cpu")

    # Interactive mode without --progress-json emits human summary footer on stdout
    run_dir_human = tmp_path / "run_human"
    res_human = run_cli(tmp_path, "train", config, "--run-dir", run_dir_human, "--steps", "1",
                        "--set", "train.lr_floor=1.0", "--no-server", isolated=False)
    assert res_human.returncode == 0, res_human.stderr
    assert "Training complete" in res_human.stdout
    assert "Steps:" in res_human.stdout
    assert "Checkpoint:" in res_human.stdout
    assert "Dashboard:" in res_human.stdout

    # Machine mode with --progress-json emits valid JSON result
    run_dir_json = tmp_path / "run_json"
    res_json = run_cli(tmp_path, "train", config, "--run-dir", run_dir_json, "--steps", "1",
                       "--progress-json", "--set", "train.lr_floor=1.0", "--no-server", isolated=False)
    assert res_json.returncode == 0, res_json.stderr
    last_line = res_json.stdout.strip().splitlines()[-1]
    parsed = json.loads(last_line)
    assert parsed.get("event") == "result"
    assert parsed["manifest"]["status"] == "complete"
    assert parsed["manifest"]["steps"] == 1


def test_run_train_writes_provenance_inside_run_lock(tmp_path):
    from hypergan.run_controller import run_train
    from hypergan.run_state import run_lock
    from hypergan.config import resolve_config
    import hypergan.api as hg

    # Create a simple valid configuration
    config_file = tmp_path / "model.toml"
    hg.save(hg.override(hg.load(Path(__file__).resolve().parents[2] / "examples" / "paired-linear.toml"),
                        {"training": {"device": "cpu", "steps": 1}}), config_file)
    cfg = resolve_config(hg.lower(hg.load(config_file)))
    cfg["_source_path"] = str(config_file)

    run_dir = tmp_path / "concurrent_run"
    run_dir.mkdir(parents=True)

    # Acquire lock externally simulating a concurrent training process
    with run_lock(run_dir):
        with pytest.raises(RuntimeError, match="Run is already locked"):
            from hypergan.single_execution import SingleProcessExecution
            run_train(cfg, run_dir, steps=1, execution_factory=SingleProcessExecution)

        # Confirm that no launch provenance or config snapshots were created while locked
        assert not (run_dir / "launch").exists()
        assert not (run_dir / "config-source").exists()

    # Now with lock released, run_train succeeds and creates provenance inside lock
    from hypergan.single_execution import SingleProcessExecution
    result = run_train(cfg, run_dir, steps=1, execution_factory=SingleProcessExecution)
    assert result is not None
    assert (run_dir / "launch" / "0001.json").is_file()
    assert (run_dir / "config-source" / "model.toml").is_file()
