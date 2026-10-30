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
        "extends": {
            "type": "string",
            "description": "Name of packaged starter recipe or template this model inherits from.",
        },
        "data": {
            "type": "object",
            "description": "Dataset provider configuration.",
            "fields": {
                "builtin": {
                    "type": "string",
                    "description": "Built-in synthetic dataset generator.",
                    "choices": ["gaussian_grid", "paired_linear", "image_folder"],
                },
                "factory": {
                    "type": "string",
                    "description": "Custom Python batch generator class in module:Class format.",
                },
                "dataset": {
                    "type": "string",
                    "description": "Custom PyTorch Dataset class in module:Class format.",
                },
                "root": {
                    "type": "string",
                    "description": "Root filesystem path to image directory for ImageFolder.",
                },
                "resolution": {
                    "type": "integer",
                    "description": "Image target spatial resolution (square H=W).",
                    "default": 32,
                },
                "channels": {
                    "type": "integer",
                    "description": "Number of image color channels (typically 1 or 3).",
                    "default": 3,
                },
                "alpha": {
                    "type": "string",
                    "description": "Alpha compositing mode for RGBA/palette images.",
                    "choices": ["composite_white", "composite_black", "error"],
                    "default": "composite_white",
                },
                "follow_symlinks": {
                    "type": "boolean",
                    "description": "Whether ImageFolder should traverse directory symlinks.",
                    "default": False,
                },
            },
        },
        "prior": {
            "type": "object",
            "description": "Latent distribution or ParticleGAN learned particle table.",
            "fields": {
                "kind": {
                    "type": "string",
                    "description": "Prior distribution type.",
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
                    "description": "Number of learned latent particles when kind='particles'.",
                    "default": 10000,
                },
                "fixed_sigma": {
                    "type": "boolean",
                    "description": "Freeze particle variance scales during training.",
                    "default": False,
                },
            },
        },
        "networks": {
            "type": "object",
            "description": "Declared neural network components (generator, critics, encoders).",
            "properties": {
                "role": {
                    "type": "string",
                    "description": "Functional role in GAN game.",
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
                    "description": "Custom PyTorch nn.Module class in module:Class format.",
                },
                "reuse": {
                    "type": "string",
                    "description": "Name of another network whose weights are shared with this component.",
                },
                "inputs": {
                    "type": "object",
                    "description": "Mapping from network input argument ports to runtime context paths.",
                },
                "input_shape": {
                    "type": "array",
                    "description": "Input tensor shape contract, e.g. ['B', 4] or ['B', 3, 32, 32].",
                },
                "output_shape": {
                    "type": "array",
                    "description": "Output tensor shape contract, e.g. ['B', 1] or ['B', 3, 32, 32].",
                },
                "trainable": {
                    "type": "boolean",
                    "description": "Whether optimizer computes gradients and updates parameters for this network.",
                    "default": True,
                },
                "optimizer": {
                    "type": "object",
                    "description": "Component-specific optimizer overrides.",
                    "fields": {
                        "lr": {"type": "number", "description": "Learning rate for this network."},
                        "betas": {"type": "array", "description": "Adam beta parameters [beta1, beta2]."},
                        "d_guard_ratio": {"type": "number", "description": "Discriminator update guard threshold."},
                    },
                },
            },
        },
        "losses": {
            "type": "array",
            "description": "List of objective loss terms minimized during training.",
            "items": {
                "type": "object",
                "fields": {
                    "type": {
                        "type": "string",
                        "description": "Objective loss type.",
                        "choices": ["adversarial", "prior", "reconstruction", "gradient_penalty"],
                    },
                    "critic": {
                        "type": "string",
                        "description": "Name of the critic network scoring real and fake samples.",
                    },
                    "penalty": {
                        "type": "number",
                        "description": "R1/gradient penalty coefficient applied to critic.",
                        "default": 1.0,
                    },
                    "weight": {
                        "type": "number",
                        "description": "Loss term multiplier.",
                        "default": 1.0,
                    },
                },
            },
        },
        "train": {
            "type": "object",
            "description": "Global training execution loop parameters.",
            "fields": {
                "steps": {
                    "type": "integer",
                    "description": "Total generator/critic training step budget.",
                    "default": 1000,
                },
                "batch_size": {
                    "type": "integer",
                    "description": "Batch size per device or worker.",
                    "default": 64,
                },
                "seed": {
                    "type": "integer",
                    "description": "Master RNG seed for deterministic initialization.",
                    "default": 0,
                },
                "device": {
                    "type": "string",
                    "description": "Primary compute device ('cuda', 'cuda:0', 'cpu', 'rocm').",
                    "default": "cuda",
                },
                "ema": {
                    "type": "number",
                    "description": "Exponential moving average decay for shadow generator weights.",
                    "default": 0.999,
                },
                "every_steps": {
                    "type": "integer",
                    "description": "Cadence in steps for evaluating standard metrics.",
                    "default": 10,
                },
            },
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
                    "description": "Best-checkpoint retention criteria based on metric evaluation.",
                    "default": {"metric": "fid50k_train", "mode": "min", "count": 2},
                },
                "keep_every": {
                    "type": "integer",
                    "description": "Step interval to retain perpetual milestone checkpoints.",
                    "default": 50000,
                },
                "export_bundle": {
                    "type": "string",
                    "description": "When to automatically export standalone generator bundles.",
                    "choices": ["on_eval", "on_checkpoint", "never"],
                    "default": "on_eval",
                },
            },
        },
        "run": {
            "type": "object",
            "description": "Runtime cluster topology and output cadence settings.",
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
                    "default": 1,
                },
                "devices": {
                    "type": "array",
                    "description": "Explicit list of GPU device indices, e.g. [0, 1].",
                },
                "nodes": {
                    "type": "integer",
                    "description": "Number of cluster nodes participating in distributed training.",
                    "default": 1,
                },
                "strategy": {
                    "type": "string",
                    "description": "Distributed data-parallel strategy.",
                    "choices": ["ddp", "diloco"],
                    "default": "ddp",
                },
            },
        },
        "metrics": {
            "type": "object",
            "description": "Metric calculation and logging configuration.",
            "fields": {
                "preset": {
                    "type": "string",
                    "description": "Metric preset bundle.",
                    "choices": ["standard", "all", "minimal"],
                    "default": "standard",
                },
                "every_steps": {
                    "type": "integer",
                    "description": "Frequency in training steps for metric logging.",
                    "default": 10,
                },
            },
        },
        "evaluations": {
            "type": "object",
            "description": "Declared offline evaluation tasks (e.g. InceptionFID).",
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
