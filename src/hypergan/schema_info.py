"""Schema introspection for hypergan-model/1 specification."""
from typing import Any, Dict, Optional


MODEL_SCHEMA: Dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "format": "hypergan-model/1",
    "version": 1,
    "title": "HyperGAN Model Specification",
    "description": "Declarative TOML specification for HyperGAN 2.0 models, networks, losses, and run policies.",
    "type": "object",
    "required": ["format", "name"],
    "sections": {
        "format": {
            "type": "string",
            "description": "Schema format identifier, must be 'hypergan-model/1'.",
            "default": "hypergan-model/1",
        },
        "name": {
            "type": "string",
            "description": "Unique model identifier or slug, e.g. 'cifar10-dcgan' or 'demo/gaussian-grid'.",
        },
        "description": {
            "type": "string",
            "description": "Human-readable description of the model architecture and intended dataset.",
            "default": "",
        },
        "defaults": {
            "type": "string",
            "description": "Numerical formulation preset ('particlegan' or 'hypergan').",
            "choices": ["particlegan", "hypergan"],
            "default": "particlegan",
        },
        "extends": {
            "type": "string",
            "description": "Path or packaged model name this configuration inherits from.",
        },
        "data": {
            "type": "object",
            "description": "Dataset provider configuration ([data] or alias [dataset]).",
            "fields": {
                "builtin": {
                    "type": "string",
                    "description": "Built-in dataset loader identifier.",
                    "choices": ["gaussian_grid", "cifar10", "paired_linear", "image_folder"],
                },
                "factory": {
                    "type": "string",
                    "description": "Custom Python batch generator callable in module:Class format.",
                },
                "dataset": {
                    "type": "string",
                    "description": "Custom item-level Dataset class in module:Class format.",
                },
                "root": {
                    "type": "string",
                    "description": "Root filesystem path to dataset directory.",
                },
                "width": {
                    "type": "integer",
                    "description": "Target image width in pixels for image_folder.",
                    "default": 64,
                },
                "height": {
                    "type": "integer",
                    "description": "Target image height in pixels for image_folder.",
                    "default": 64,
                },
                "channels": {
                    "type": "integer",
                    "description": "Number of image color channels (1 for L, 3 for RGB).",
                    "default": 3,
                },
                "mode": {
                    "type": "string",
                    "description": "Color mode for image_folder ('RGB' or 'L').",
                    "choices": ["RGB", "L"],
                    "default": "RGB",
                },
                "resize": {
                    "type": "string",
                    "description": "Spatial resize policy for image_folder.",
                    "choices": ["center_crop", "pad", "stretch", "none"],
                    "default": "center_crop",
                },
                "alpha": {
                    "type": "string",
                    "description": "Alpha compositing mode for RGBA/palette images.",
                    "choices": ["composite_white", "composite_black", "error"],
                    "default": "composite_white",
                },
                "follow_symlinks": {
                    "type": "boolean",
                    "description": "Whether image_folder should traverse directory symlinks.",
                    "default": False,
                },
                "batch_size": {
                    "type": "integer",
                    "description": "Global batch size per training step (alias for train.batch_size).",
                },
                "shuffle": {
                    "type": "boolean",
                    "description": "Whether item-level datasets shuffle items each epoch.",
                    "default": True,
                },
                "workers": {
                    "type": "integer",
                    "description": "Number of background worker threads for item-level datasets.",
                    "default": 0,
                },
                "args": {
                    "type": "object",
                    "description": "Optional subtable of constructor arguments passed to the dataset or factory.",
                },
            },
        },
        "prior": {
            "type": "object",
            "description": "Latent distribution or ParticleGAN learned particle table.",
            "fields": {
                "kind": {
                    "type": "string",
                    "description": "Prior distribution family (alias: 'type').",
                    "choices": ["particles", "mog", "gaussian"],
                    "default": "particles",
                },
                "z_dim": {
                    "type": "integer",
                    "description": "Dimensionality of latent code vector.",
                    "default": 4,
                },
                "num_particles": {
                    "type": "integer",
                    "description": "Number of learned latent particles or mixture components.",
                    "default": 10000,
                },
                "init_std": {
                    "type": "number",
                    "description": "Initial standard deviation for latent draws or particle centers.",
                    "default": 1.0,
                },
                "fixed_sigma": {
                    "type": "number",
                    "description": "Fixed isotropic Gaussian noise scale for 'mog' priors.",
                },
                "sigma_rel": {
                    "type": "number",
                    "description": "Relative nearest-neighbor sigma calibration scale for 'mog' priors.",
                    "default": 0.025,
                },
                "learnable": {
                    "type": "boolean",
                    "description": "Whether particle centers are updated by the prior optimizer.",
                    "default": True,
                },
                "standardize": {
                    "type": "boolean",
                    "description": "Whether to standardize learned 'mog' particle centers across rows.",
                },
                "initialization_seed": {
                    "type": "integer",
                    "description": "Dedicated deterministic RNG seed for initializing the particle table.",
                },
                "initialization_device": {
                    "type": "string",
                    "description": "Device used to draw initial particle table ('execution' or 'cpu').",
                    "choices": ["execution", "cpu"],
                    "default": "execution",
                },
            },
        },
        "networks": {
            "type": "object",
            "description": "Declared neural network components (generator, critics, encoders, auxiliary).",
            "properties": {
                "role": {
                    "type": "string",
                    "description": "Functional role in the GAN training graph.",
                    "choices": ["generator", "critic", "encoder", "auxiliary"],
                },
                "hndl": {
                    "type": "string",
                    "description": "Relative path to native HNDL (.hndl) network source file.",
                },
                "file": {
                    "type": "string",
                    "description": "Alternative key for .hndl file path.",
                },
                "source": {
                    "type": "string",
                    "description": "Inline HNDL architecture DSL string.",
                },
                "module": {
                    "type": "string",
                    "description": "Custom PyTorch nn.Module class or factory in module:Class format.",
                },
                "reuse": {
                    "type": "string",
                    "description": "Name of another declared network whose weights are shared for this pass.",
                },
                "freeze_parameters": {
                    "type": "boolean",
                    "description": "When true on a reused network, blocks parameter gradients while keeping input gradients.",
                    "default": False,
                },
                "inputs": {
                    "type": "object",
                    "description": "Mapping from network input argument ports to graph bindings ('latent', 'candidate', 'batch.real', '<network>').",
                },
                "input_shape": {
                    "type": "array",
                    "description": "Input tensor shape contract for HNDL networks, e.g. ['B', 128] or { z = ['B', 128] }.",
                },
                "output_shape": {
                    "type": "array",
                    "description": "Output tensor shape contract for HNDL networks, e.g. ['B', 1] or ['B', 3, 64, 64].",
                },
                "input_dtype": {
                    "type": "string",
                    "description": "Optional input tensor dtype for HNDL networks (e.g. 'float32' or 'int64').",
                },
                "trainable": {
                    "type": "boolean",
                    "description": "Whether optimizer updates parameters for this network.",
                    "default": True,
                },
                "args": {
                    "type": "object",
                    "description": "Constructor arguments for module networks or HNDL template parameters/providers.",
                },
            },
        },
        "losses": {
            "type": "array",
            "description": "Array of objective loss tables ([[losses]]) minimized during training.",
            "items": {
                "type": "object",
                "fields": {
                    "type": {
                        "type": "string",
                        "description": "Objective loss family.",
                        "choices": ["adversarial", "prior", "reconstruction", "objective"],
                    },
                    "id": {
                        "type": "string",
                        "description": "Unique identifier for metrics publication (required for 'objective' and secondary 'adversarial' terms).",
                    },
                    "critic": {
                        "type": "string",
                        "description": "Name of the critic network scoring real and fake samples (for type='adversarial').",
                    },
                    "penalty": {
                        "type": "number",
                        "description": "Critic gradient penalty coefficient (for type='adversarial'; 0 disables).",
                        "default": 1.0,
                    },
                    "weight": {
                        "type": "number",
                        "description": "Scalar multiplier on this loss term.",
                        "default": 1.0,
                    },
                    "fn": {
                        "type": "string",
                        "description": "Loss function ('mse', 'l1', or 'module:callable') for 'reconstruction'/'objective' losses.",
                        "default": "mse",
                    },
                    "input": {
                        "type": "string",
                        "description": "Input binding for 'reconstruction'/'objective' losses (defaults to 'generated' for reconstruction).",
                    },
                    "target": {
                        "type": "string",
                        "description": "Target binding for 'reconstruction'/'objective' losses (defaults to 'batch.real' for reconstruction).",
                    },
                    "inputs": {
                        "type": "object",
                        "description": "Named argument bindings table for custom multi-input losses or conditional critics.",
                    },
                    "detach": {
                        "type": "array",
                        "description": "List of input port names to detach before gradient computation.",
                        "default": ["target"],
                    },
                    "rows": {
                        "type": "string",
                        "description": "Particle table subset regularized by type='prior' loss.",
                        "choices": ["sampled_unique", "full"],
                        "default": "sampled_unique",
                    },
                },
            },
        },
        "penalty": {
            "type": "object",
            "description": "Shared critic gradient-penalty schedule and anchor settings.",
            "fields": {
                "kappa": {
                    "type": "number",
                    "description": "Scale factor for ParticleGAN critic gradient penalty.",
                    "default": 1.0,
                },
                "lazy_k": {
                    "type": "integer",
                    "description": "Lazy regularization interval in training steps.",
                    "default": 1,
                },
                "anchor_weight": {
                    "type": "number",
                    "description": "Anchor stabilization weight.",
                    "default": 1.0,
                },
                "anchor_decay": {
                    "type": "number",
                    "description": "EMA decay for critic anchor stabilization in [0, 1).",
                    "default": 0.999,
                },
            },
        },
        "train": {
            "type": "object",
            "description": "Unified training schedule, optimizer, and backend determinism settings.",
            "fields": {
                "steps": {
                    "type": "integer",
                    "description": "Total logical training steps to execute.",
                    "default": 5,
                },
                "batch_size": {
                    "type": "integer",
                    "description": "Global batch size per training step.",
                    "default": 16,
                },
                "seed": {
                    "type": "integer",
                    "description": "Master RNG seed for deterministic initialization.",
                    "default": 42,
                },
                "device": {
                    "type": "string",
                    "description": "Target compute device ('cuda', 'cuda:0', 'cuda:1', 'cpu').",
                    "default": "cuda",
                },
                "initialization": {
                    "type": "string",
                    "description": "Network weight initialization policy.",
                    "choices": ["constructor", "deterministic_orthogonal"],
                    "default": "constructor",
                },
                "lr": {
                    "type": "number",
                    "description": "Base generator/encoder Adam learning rate.",
                    "default": 0.0006,
                },
                "d_lr_mult": {
                    "type": "number",
                    "description": "Critic learning-rate multiplier relative to lr.",
                    "default": 1.5,
                },
                "prior_lr_mult": {
                    "type": "number",
                    "description": "Learnable prior learning-rate multiplier relative to lr.",
                    "default": 10.0,
                },
                "betas": {
                    "type": "array",
                    "description": "Adam [beta1, beta2] coefficients for network parameters.",
                    "default": [0.0, 0.999],
                },
                "prior_betas": {
                    "type": "array",
                    "description": "Adam [beta1, beta2] coefficients for learnable prior parameters.",
                    "default": [0.0, 0.999],
                },
                "implementation": {
                    "type": "string",
                    "description": "Adam optimizer kernel implementation.",
                    "choices": ["recipe", "device_adam", "torch_fused_adam"],
                    "default": "device_adam",
                },
                "ema": {
                    "type": "number",
                    "description": "Exponential moving average decay for inference generator weights in [0, 1).",
                    "default": 0.995,
                },
                "lr_anneal_start": {
                    "type": "number",
                    "description": "Fraction of total steps where learning-rate annealing begins.",
                    "default": 0.6,
                },
                "lr_floor": {
                    "type": "number",
                    "description": "Final learning-rate floor multiplier (1.0 keeps LR constant for step extension).",
                    "default": 0.05,
                },
                "every_steps": {
                    "type": "integer",
                    "description": "Cadence in steps for writing scalar training metrics.",
                    "default": 10,
                },
                "checkpoint_every": {
                    "type": "integer",
                    "description": "Default step interval between durable checkpoints.",
                    "default": 1000,
                },
                "preview_every": {
                    "type": "integer",
                    "description": "Default step interval between preview grid renders (0 disables).",
                    "default": 0,
                },
                "progress_every": {
                    "type": "integer",
                    "description": "Default step interval between CLI progress lines.",
                    "default": 100,
                },
                "backend": {
                    "type": "object",
                    "description": "CUDA/cuDNN determinism and TF32 backend flags.",
                },
            },
        },
        "sampling": {
            "type": "object",
            "description": "Periodic preview and inference sampling configuration.",
            "fields": {
                "count": {
                    "type": "integer",
                    "description": "Number of samples rendered in periodic preview grids.",
                    "default": 256,
                },
                "seed": {
                    "type": "integer",
                    "description": "Deterministic RNG seed for preview sampling.",
                    "default": 123,
                },
                "output": {
                    "type": "string",
                    "description": "Primary generator output binding to sample.",
                    "default": "generator",
                },
                "views": {
                    "type": "object",
                    "description": "Up to 4 named output bindings rendered as extra preview grids.",
                },
                "comparison": {
                    "type": "array",
                    "description": "List of 2 to 4 {label, binding} columns for side-by-side comparison grids.",
                },
            },
        },
        "samplers": {
            "type": "object",
            "description": "Named read-side visualization samplers ([samplers.<name>]).",
        },
        "checkpoints": {
            "type": "object",
            "description": "Model checkpoint creation and retention policies.",
            "fields": {
                "keep_last": {
                    "type": "integer",
                    "description": "Number of most recent checkpoints to retain.",
                    "default": 3,
                },
                "keep_best": {
                    "type": "object",
                    "description": "Best-checkpoint retention criteria ({metric, mode, count} or integer count).",
                    "default": {"metric": "fid50k_train", "mode": "min", "count": 2},
                },
                "keep_every": {
                    "type": "integer",
                    "description": "Step interval to retain perpetual milestone checkpoints.",
                    "default": 50000,
                },
                "export_bundle": {
                    "type": "string",
                    "description": "When to automatically export standalone EMA generator bundles.",
                    "choices": ["on_eval", "on_checkpoint", "never"],
                    "default": "on_eval",
                },
            },
        },
        "run": {
            "type": "object",
            "description": "Runtime device topology, viewer autostart, and output cadence settings.",
            "fields": {
                "checkpoint_every": {
                    "type": "integer",
                    "description": "Step interval between atomic checkpoint commits.",
                    "default": 1000,
                },
                "preview_every": {
                    "type": "integer",
                    "description": "Step interval between preview PNG image tile generations.",
                    "default": 0,
                },
                "progress_every": {
                    "type": "integer",
                    "description": "Step interval for emitting CLI progress lines.",
                    "default": 100,
                },
                "devices": {
                    "type": "string",
                    "description": "Target GPU device specification, e.g. '0', '0,1', 'gpu0,gpu1', 'cuda:0', or 'cpu'.",
                },
                "viewer": {
                    "type": "boolean",
                    "description": "Automatically start the background Web Observatory during training.",
                },
                "cpu_threads": {
                    "type": "integer",
                    "description": "PyTorch intra-op CPU thread limit per worker.",
                },
                "nodes": {
                    "type": "integer",
                    "description": "Number of cluster nodes participating in distributed training.",
                    "default": 1,
                },
                "strategy": {
                    "type": "string",
                    "description": "Distributed replica aggregation strategy.",
                    "choices": ["gradient_allreduce", "parameter_average", "pseudo_gradient"],
                    "default": "gradient_allreduce",
                },
            },
        },
        "aggregation": {
            "type": "object",
            "description": "Distributed replica synchronization and outer-optimizer settings.",
            "fields": {
                "strategy": {
                    "type": "string",
                    "description": "Replica aggregation strategy.",
                    "choices": ["gradient_allreduce", "parameter_average", "pseudo_gradient"],
                    "default": "gradient_allreduce",
                },
                "sync_every_steps": {
                    "type": "integer",
                    "description": "Synchronization cadence in local steps (alias: 'sync_every').",
                    "default": 1,
                },
                "outer_lr": {
                    "type": "number",
                    "description": "Outer optimizer learning rate for pseudo_gradient / DiLoCo.",
                    "default": 1.0,
                },
                "outer_momentum": {
                    "type": "number",
                    "description": "Outer Nesterov momentum factor in [0.0, 1.0).",
                    "default": 0.0,
                },
            },
        },
        "metrics": {
            "type": "object",
            "description": "Scalar training metric calculation and logging configuration.",
            "fields": {
                "preset": {
                    "type": "string",
                    "description": "Built-in scalar metric preset bundle.",
                    "choices": ["standard", "none"],
                    "default": "standard",
                },
                "every_steps": {
                    "type": "integer",
                    "description": "Frequency in training steps for scalar metric logging (alias: 'every').",
                    "default": 10,
                },
                "disable": {
                    "type": "array",
                    "description": "List of standard metric IDs to disable.",
                },
                "overrides": {
                    "type": "object",
                    "description": "Per-metric enable/disable overrides table.",
                },
            },
        },
        "evaluations": {
            "type": "object",
            "description": "Declared snapshot evaluation tasks ([evaluations.<id>], e.g. InceptionFID or SampleDiversity).",
            "fields": {
                "factory": {
                    "type": "string",
                    "description": "Evaluator class import path ('module:Class').",
                },
                "fn": {
                    "type": "string",
                    "description": "Evaluator function import path ('module:callable').",
                },
                "every": {
                    "type": "integer",
                    "description": "Interval in training steps for automatic evaluation (omit for manual trigger).",
                },
                "samples": {
                    "type": "integer",
                    "description": "Total number of generated and reference examples to evaluate.",
                },
                "batch_size": {
                    "type": "integer",
                    "description": "Batch size per evaluation forward pass.",
                },
                "seed": {
                    "type": "integer",
                    "description": "Deterministic RNG seed for evaluation draws.",
                },
                "device": {
                    "type": "string",
                    "description": "Execution device for the evaluation worker ('cuda', 'cuda:0', 'cpu').",
                },
                "data": {
                    "type": "object",
                    "description": "Evaluation dataset table ({builtin/dataset/factory, ...}) or 'train' to reuse [data] with shuffle=false.",
                },
            },
        },
    },
}


def get_schema(section: Optional[str] = None) -> Dict[str, Any]:
    """Return the hypergan-model/1 specification, optionally filtered to a section.

    Args:
        section: Optional section name, e.g. 'train', 'data', 'networks', 'losses'.

    Returns:
        Full schema dictionary or section-specific subschema.
    """
    if section is None:
        return MODEL_SCHEMA
    sections = MODEL_SCHEMA["sections"]
    if section in sections:
        return {"format": MODEL_SCHEMA["format"], "section": section, **sections[section]}
    valid = sorted(sections.keys())
    raise KeyError(f"Unknown schema section '{section}'; available sections: {', '.join(valid)}")
