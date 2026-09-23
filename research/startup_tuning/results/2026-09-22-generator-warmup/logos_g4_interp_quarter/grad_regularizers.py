"""
Gradient regularizers for GAN discriminators.

The repo baseline (examples/100gaussians.py) uses R1+R2, i.e. zero-centered
penalties on ||grad_x D(x)|| at the reals and the fakes. That drives D toward
flatness: the unique global minimum of the penalty alone is a constant D.

This module collects a family of *re-centered* alternatives so they can be
swapped in a controlled experiment. The interesting one is the "eikonal"
penalty (n - 1)^2, borrowed from implicit-surface fitting, which pins the slope
of D at 1 instead of 0 and therefore *forbids* a flat discriminator while still
bounding its steepness.

All arms share the same coefficient units: every data-point arm is written as

    penalty = (coeff / 2) * ( E_real[phi(n_r)] + E_fake[phi(n_f)] )

so that `a_r1r2` with coeff=0.02 reproduces `r1_gamma=0.02` in the example
script exactly.

Two orthogonal knobs sit on top of that family, both off by default:

  * `norm` picks *which* norm of grad_x D(x) is penalized. Penalizing the L1
    norm of the gradient corresponds to enforcing an L-infinity margin around
    the decision boundary, and penalizing the L-infinity norm corresponds to an
    L1 margin -- the norm on the gradient and the norm on the margin are dual
    (cf. arXiv:1910.06922, "Gradient penalty from a maximum margin
    perspective"). The default L2/L2 pairing is just the self-dual case.
  * `target_anneal` moves the penalty *center* c over training. As c -> 0 the
    re-centered arms continuously morph into a zero-centered R1/R2-like
    penalty: b_cap's cap slides down to relu(n)^2 = n^2 and c_eikonal's
    (n - 1)^2 becomes (n - 0)^2 = ||g||^2. So an annealed run starts out
    forbidding a flat D and ends up with the baseline's flatness pressure,
    without ever switching arms mid-run.
"""

from typing import Dict, Tuple
import math

import torch
import torch.nn.functional as F


def _score_scalar(logits):
    """Sum of per-image logit means.

    One logit per image matches ``logits.sum()``: other batch rows do not
    change an image's gradient. Extra logits are averaged inside the image so
    a spatial map does not multiply that gradient by its number of locations.
    """
    if logits.ndim < 2:
        return logits.sum()
    return logits.flatten(1).mean(dim=1).sum()


def finite_difference_norm(critic, x, eps):
    """Central derivative along detached normalized grad_x D.

    Uses one ordinary input backward to choose the steepest direction, then
    two forwards/ordinary parameter backwards instead of double backward.
    As eps tends to zero this estimates both ||grad_x D|| and its parameter
    derivative away from activation kinks. eps is an L2 image displacement,
    not per-pixel noise. No random-direction surrogate and no pixel clamping.
    """
    x = x.detach().requires_grad_(True)
    grad = torch.autograd.grad(_score_scalar(critic(x)), x, create_graph=False)[0]
    norm = (grad.square().flatten(1).sum(1) + 1e-12).sqrt()
    direction = (grad / norm.reshape((-1,) + (1,) * (x.ndim - 1))).detach()
    center = x.detach()
    return (critic(center + eps * direction) - critic(center - eps * direction)) / (2 * eps)


class GradRegularizer:
    """
    Discriminator gradient penalty with a selectable centering scheme.

    Args:
        arm (str): one of ARMS.
            - 'a_r1r2':    phi(n) = n^2          (baseline R1+R2, zero-centered)
            - 'b_cap':     phi(n) = relu(n - kappa)^2   (one-sided cap, free below kappa)
            - 'c_eikonal': phi(n) = (n - 1)^2    (two-sided, slope pinned at 1)
            - 'd_asym':    phi(n) = relu(n - 1)^2 + 0.25 * relu(1 - n)^2
                           (eikonal, but 4x cheaper to be too flat than too steep)
            - 'e_interp':  (n_i - 1)^2 on real/fake interpolates (WGAN-GP geometry,
                           centered at 1 rather than at 0)
            - 'f_none':    no penalty at all.
            - 'g_interp_cap': relu(n_i - kappa)^2 on real/fake interpolates: the
                           one-sided cap of 'b_cap', but enforced along the
                           path between the samples rather than at them. In
                           high-dimensional data the sample-point cap leaves
                           D free to be arbitrarily steep *between* reals and
                           fakes (which is where the fakes have to travel), so
                           this is the cap's natural high-dim form.
        coeff (float): penalty strength (`r1_gamma` in the example script).
        kappa (float): the cap for 'b_cap'; ignored by the other arms.
        lazy_k (int): apply the penalty only every k-th step and multiply the
            coefficient by k ("lazy regularization", StyleGAN2). k=1 means
            every step.
        norm (str): which norm of grad_x D(x) the arms b_cap / c_eikonal /
            d_asym / e_interp penalize.
            - 'l2':   n = sqrt(sum_i g_i^2 + 1e-12)   (default, self-dual)
            - 'l1':   n = sum_i |g_i|
            - 'linf': n = max_i |g_i|
            l1 and linf need no epsilon: there is no sqrt to differentiate at
            g = 0. Because the norm on the gradient is dual to the norm in
            which the margin is measured, penalizing ||g||_1 enforces an
            L-infinity margin and penalizing ||g||_inf enforces an L1 margin
            (arXiv:1910.06922). `a_r1r2` is defined as the squared-L2 R1/R2
            form and keeps that path regardless of this setting, so asking it
            for any other norm is a ValueError rather than a silent no-op.
        target_anneal (str): schedule for the penalty *center* c, i.e. the
            kappa in b_cap and the literal 1.0 in c_eikonal / d_asym /
            e_interp. With c0 = kappa for b_cap and 1.0 for the others:
            - 'none':    c = c0 at every step (the historical behavior).
            - 'linear':  c(step) = c0 * max(0, 1 - step / total_steps).
            - 'delayed': c = c0 while step < 0.6 * total_steps, then linear
                         from c0 down to 0 at total_steps.
            As c -> 0 the re-centered arms degrade continuously into a
            zero-centered R1/R2-like penalty (relu(n)^2 = n^2 for b_cap,
            (n - 0)^2 for c_eikonal), so an anneal is a smooth handover from
            "D may not be flat" to "D should be flat". `a_r1r2` has no center
            and is unaffected.
        method (str): autograd (exact double backward) or finite_difference
            (gradient-aligned central difference, L2 b_cap only).
        fd_eps (float): L2 input displacement for central differences.
        total_steps (int): run length the schedule is expressed against.
            Required (> 0) whenever `target_anneal` is not 'none'.
    """

    ARMS = ("a_r1r2", "b_cap", "c_eikonal", "d_asym", "e_interp", "f_none", "g_interp_cap")
    NORMS = ("l2", "l1", "linf")
    ANNEALS = ("none", "linear", "delayed")

    # Fraction of the run that 'delayed' holds the center at c0 before the ramp.
    DELAY_FRAC = 0.6

    def __init__(
        self,
        arm: str = "b_cap",
        coeff: float = 1.0,
        kappa: float = 1.0,
        lazy_k: int = 1,
        norm: str = "l2",
        target_anneal: str = "none",
        total_steps: int = 0,
        method: str = "autograd",
        fd_eps: float = 0.05,
    ) -> None:
        if arm not in self.ARMS:
            raise ValueError(f"Unknown grad regularizer arm: {arm} (expected one of {self.ARMS})")

        if method not in ("autograd", "finite_difference"):
            raise ValueError(f"Unknown gradient method: {method}")
        if not math.isfinite(fd_eps) or fd_eps <= 0:
            raise ValueError("fd_eps must be finite and positive")
        if method == "finite_difference" and (arm != "b_cap" or norm != "l2"):
            raise ValueError("finite differences currently support L2 b_cap only")
        self.method, self.fd_eps = method, float(fd_eps)
        self.arm = arm
        self.coeff = float(coeff)
        self.kappa = float(kappa)
        self.lazy_k = int(lazy_k)
        self.norm = str(norm)
        self.target_anneal = str(target_anneal)
        self.total_steps = int(total_steps)
        if not math.isfinite(self.coeff) or self.coeff < 0:
            raise ValueError("coeff must be finite and nonnegative")
        if not math.isfinite(self.kappa) or self.kappa < 0:
            raise ValueError("kappa must be finite and nonnegative")

        if self.lazy_k < 1:
            raise ValueError(f"lazy_k must be >= 1, got {lazy_k}")

        if self.norm not in self.NORMS:
            raise ValueError(f"Unknown grad norm: {norm} (expected one of {self.NORMS})")
        if self.arm == "a_r1r2" and self.norm != "l2":
            raise ValueError(
                "arm 'a_r1r2' is the squared-L2 R1/R2 penalty and supports only "
                f"norm='l2', got norm={norm!r}"
            )

        if self.target_anneal not in self.ANNEALS:
            raise ValueError(
                f"Unknown target_anneal: {target_anneal} (expected one of {self.ANNEALS})"
            )
        if self.target_anneal != "none" and self.total_steps <= 0:
            raise ValueError(
                f"target_anneal={target_anneal!r} needs total_steps > 0, got {total_steps}"
            )

    # -------------------------
    #  Public API
    # -------------------------

    def penalty(
        self,
        D: torch.nn.Module,
        x_real: torch.Tensor,
        x_fake: torch.Tensor,
        step: int = 1,
        generator: torch.Generator = None,
        collect_stats: bool = True,
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Compute the penalty term to add to the discriminator loss.

        collect_stats=False returns empty stats and avoids a GPU scalar sync.

        Returns:
            (penalty_loss, stats)
            penalty_loss: scalar tensor attached to D's graph, or a detached
                zero on the right device/dtype when the arm is 'f_none' or the
                lazy schedule skips this step.
            stats: {'applied': bool, 'pen': float} plus 'center' (the penalty
                center actually used) on the steps where it is applied.
        """
        skip = self.arm == "f_none" or (self.lazy_k > 1 and step % self.lazy_k != 0)
        if skip:
            zero = torch.zeros((), device=x_real.device, dtype=x_real.dtype)
            return zero, ({"applied": False, "pen": 0.0} if collect_stats else {})

        # Lazy regularization: fewer applications, proportionally bigger hits,
        # so the time-averaged pressure on D is unchanged.
        coeff_eff = self.coeff * self.lazy_k if self.lazy_k > 1 else self.coeff

        center = self.center(step)

        if self.arm in ("e_interp", "g_interp_cap"):
            pen = coeff_eff * self._interp_term(D, x_real, x_fake, center, generator)
        else:
            n_r = self._grad_norm(D, x_real, squared=(self.arm == "a_r1r2"))
            n_f = self._grad_norm(D, x_fake, squared=(self.arm == "a_r1r2"))
            phi = self._phi
            pen = (coeff_eff / 2.0) * (phi(n_r, center).mean() + phi(n_f, center).mean())

        return pen, ({"applied": True, "pen": float(pen.detach()), "center": float(center)} if collect_stats else {})

    def __call__(self, D, x_real, x_fake, step=1, generator=None):
        """Return only the penalty tensor, without collecting synchronized stats."""
        return self.penalty(D, x_real, x_fake, step, generator, collect_stats=False)[0]

    def center(self, step: int) -> float:
        """
        The penalty center c in force at `step` (see `target_anneal`).

        c0 is `kappa` for b_cap and 1.0 for c_eikonal / d_asym / e_interp.
        `a_r1r2` is zero-centered by construction and always reports 0.0.
        """
        if self.arm == "a_r1r2":
            return 0.0
        c0 = self.kappa if self.arm in ("b_cap", "g_interp_cap") else 1.0

        if self.target_anneal == "none":
            return c0
        if self.target_anneal == "linear":
            return c0 * max(0.0, 1.0 - step / self.total_steps)
        if self.target_anneal == "delayed":
            hold = self.DELAY_FRAC * self.total_steps
            if step < hold:
                return c0
            frac = (step - hold) / max(1e-12, self.total_steps - hold)
            return c0 * max(0.0, 1.0 - frac)
        raise ValueError(f"Unknown target_anneal: {self.target_anneal}")

    # -------------------------
    #  Penalty kernels
    # -------------------------

    def _phi(self, n: torch.Tensor, center: float) -> torch.Tensor:
        """Per-sample penalty on the gradient norm (or its square, for a_r1r2)."""
        if self.arm == "a_r1r2":
            # `n` is already ||g||^2 here: no sqrt, no epsilon, so this is
            # bit-for-bit the inline R1/R2 formula in examples/100gaussians.py.
            return n
        if self.arm == "b_cap":
            return F.relu(n - center).pow(2)
        if self.arm == "c_eikonal":
            return (n - center).pow(2)
        if self.arm == "d_asym":
            # Steepness costs full price; flatness costs a quarter. Keeps the
            # "no flat D" property while letting D relax between the modes.
            return F.relu(n - center).pow(2) + 0.25 * F.relu(center - n).pow(2)
        raise ValueError(f"Unknown grad regularizer arm: {self.arm}")

    def _interp_term(
        self,
        D: torch.nn.Module,
        x_real: torch.Tensor,
        x_fake: torch.Tensor,
        center: float,
        generator: torch.Generator = None,
    ) -> torch.Tensor:
        """E[(||grad D(x_i)|| - c)^2] on per-sample real/fake interpolates."""
        eps = torch.rand(
            (x_real.shape[0],) + (1,) * (x_real.ndim - 1),
            device=x_real.device, dtype=x_real.dtype, generator=generator,
        )
        x_i = eps * x_real.detach() + (1.0 - eps) * x_fake.detach()
        n_i = self._grad_norm(D, x_i, squared=False)
        if self.arm == "g_interp_cap":
            return F.relu(n_i - center).pow(2).mean()
        return (n_i - center).pow(2).mean()

    def _grad_norm(
        self,
        D: torch.nn.Module,
        x: torch.Tensor,
        squared: bool = False,
    ) -> torch.Tensor:
        """
        ||grad_x D(x)||, per sample, in the norm selected by `self.norm`, with
        create_graph=True so the penalty is differentiable w.r.t. D's params.

        `squared` returns the squared L2 norm ||g||^2 without the sqrt/epsilon
        (the R1/R2 form) and ignores `self.norm`, which the constructor has
        already pinned to 'l2' for the one arm that asks for it.

        The differentiated scalar is the sum of per-image logit means. A
        critic that returns one logit per image is unchanged.
        """
        if self.method == "finite_difference":
            return finite_difference_norm(D, x, self.fd_eps)
        x = x.detach().clone().requires_grad_(True)
        logits = D(x)
        g = torch.autograd.grad(_score_scalar(logits), x, create_graph=True)[0]
        if squared:
            return g.pow(2).flatten(1).sum(dim=1)
        if self.norm == "l1":
            # No epsilon: |.| is already differentiable a.e. and there is no
            # sqrt whose derivative blows up at g = 0.
            return g.abs().flatten(1).sum(dim=1)
        if self.norm == "linf":
            return g.abs().flatten(1).max(dim=1).values
        sq = g.pow(2).flatten(1).sum(dim=1)
        # Epsilon keeps the sqrt differentiable at g = 0.
        return torch.sqrt(sq + 1e-12)


GradientPenalty = GradRegularizer


def grad_norm_stats(
    D: torch.nn.Module,
    x_real: torch.Tensor,
    x_fake: torch.Tensor,
    generator: torch.Generator = None,
) -> Dict[str, float]:
    """
    Measurement-only summary of D's gradient-norm field (no create_graph, so
    nothing here is backpropagated).

    Reports the median and the 10/90 percentiles of ||grad_x D(x)|| at the
    reals (nr), at the fakes (nf), and at per-sample uniform interpolates
    x_i = eps * x_real + (1 - eps) * x_fake with eps ~ U(0, 1) (ni).

    Safe to call inside a `torch.no_grad()` eval block: it re-enables grad
    locally and works on detached clones.
    """
    with torch.enable_grad():
        eps = torch.rand(
            (x_real.shape[0],) + (1,) * (x_real.ndim - 1),
            device=x_real.device, dtype=x_real.dtype, generator=generator,
        )
        x_interp = eps * x_real.detach() + (1.0 - eps) * x_fake.detach()

        norms = {}
        for key, x in (("r", x_real), ("f", x_fake), ("i", x_interp)):
            xd = x.detach().clone().requires_grad_(True)
            logits = D(xd)
            g = torch.autograd.grad(_score_scalar(logits), xd, create_graph=False)[0]
            norms[key] = torch.sqrt(g.pow(2).flatten(1).sum(dim=1) + 1e-12).detach()

    stats = {}
    for key, n in norms.items():
        q = torch.quantile(
            n.float(), torch.tensor([0.1, 0.5, 0.9], device=n.device)
        )
        stats[f"q10_n{key}"] = float(q[0])
        stats[f"med_n{key}"] = float(q[1])
        stats[f"q90_n{key}"] = float(q[2])
    return stats
