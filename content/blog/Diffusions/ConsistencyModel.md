+++
date = '2026-10-18T14:00:00-07:00'
draft = false
title = 'Diffusion-Series 3 - One Step: Consistency Models, Flow Maps, MeanFlow, and Distillation'
summary = "How to generate in a single network evaluation: the flow map that every 1-NFE method learns, the from-scratch line from consistency models to MeanFlow and drifting, the three families of distillation, and where the ImageNet numbers stand."
categories = ["Diffusion Models"]
tags = ["Diffusion", "Flow Matching", "Consistency Models", "MeanFlow", "Distillation", "1-NFE"]
+++

The [flow matching post](../flowmatching/) ended with a problem. A network trained by flow matching outputs the **velocity** at a point, and one Euler step with that velocity lands on $\mathbb{E}[x_0 \mid x_t]$: the average of all the data the point could lead to. From pure noise, that is the data mean. To generate in one step, we need the network to output the **endpoint of the curved path** instead.

This post covers the two ways to get there:

- **Train a one-step model from scratch** (consistency models, flow maps, MeanFlow, and the 2026 distribution-level methods).
- **Distill a many-step teacher into a one-step student** (trajectory, distribution-matching, and adversarial distillation).

"NFE" is the number of function evaluations, i.e. forward passes of the network per sample. "1-NFE" means one forward pass. Notation follows the previous post: $x_t = (1-t)\,x_0 + t\,\epsilon$, $t=0$ is data, $t=1$ is noise, and $v(x,t)$ is the marginal velocity.

# 1. The object everyone learns: the flow map

Let $\Phi_{t \to s}(x)$ be the result of solving the ODE $\frac{dx}{d\tau} = v(x,\tau)$ from time $t$ to time $s$, starting at $x$. Sampling in one step means evaluating $\Phi_{1 \to 0}$ directly. The flow map has three defining properties:

\begin{align}
&\text{Boundary:} && \Phi_{t \to t}(x) = x \\
&\text{Composition:} && \Phi_{s \to r} \circ \Phi_{t \to s} = \Phi_{t \to r} \\
&\text{Tangent:} && \partial_s \Phi_{t \to s}(x)\big|_{s = t} = v(x, t)
\end{align}

Almost every 1-NFE method learns some slice of $\Phi$ and enforces one of these properties as its training signal. They differ in which slice and which property:

| Learned quantity | Definition | Methods |
|---|---|---|
| Consistency function | $f(x_t, t) = \Phi_{t \to 0}(x_t)$ | CM, iCT, ECT, sCM |
| Two-time flow map | $\Phi_{t \to s}(x_t)$ for any $s < t$ | CTM, Flow Map Matching, IMM, TVM, SoFlow, AYF |
| Average velocity | $u(x_t, r, t) = \dfrac{x_t - \Phi_{t \to r}(x_t)}{t - r} = \dfrac{1}{t-r}\displaystyle\int_r^t v(x_\tau, \tau)\, d\tau$ | MeanFlow, AlphaFlow, iMF, pixel MeanFlow |
| Step-size-conditioned velocity | the average velocity over a step of size $d$ | Shortcut models |

The average velocity is just the flow map in velocity units: $\Phi_{t \to r}(x_t) = x_t - (t - r)\,u(x_t, r, t)$. When $r \to t$ it becomes the ordinary velocity, so **one network does both flow matching ($r = t$) and one-step generation ($r = 0, t = 1$)**.

The question is where the supervision for $\Phi$ comes from. We don't have the true ODE solution. Either a **teacher** solves it (distillation), or the model **supervises itself** through one of the properties above (training from scratch).

# 2. From scratch: self-consistency

## 2.1 Consistency models

Consistency models (Song et al., 2023) learn $f_\theta(x_t, t) \approx \Phi_{t \to 0}(x_t)$, parameterized so the boundary condition holds exactly: $f_\theta(x, t) = c_{\text{skip}}(t)\,x + c_{\text{out}}(t)\,F_\theta(x, t)$ with $c_{\text{skip}} = 1$, $c_{\text{out}} = 0$ at the smallest noise level.

The training signal is **consistency along one trajectory**: two points on the same ODE path must map to the same endpoint.

\begin{equation}
\mathcal{L}_{\text{CM}} = \mathbb{E}\Big[\lambda(t_n)\; d\big(f_\theta(x_{t_{n+1}}, t_{n+1}),\; f_{\theta^-}(\hat x_{t_n}, t_n)\big)\Big]
\end{equation}

where $\theta^-$ is a stop-gradient (or EMA) copy, and $\hat x_{t_n}$ is one step back along the trajectory. In **consistency distillation** (CD) that step uses a teacher's ODE solver. In **consistency training** (CT) it reuses the same $(x_0, \epsilon)$ at a lower noise level, which is an unbiased estimate in the limit of small steps.

The original CT reached CIFAR-10 1-step FID 8.70. **iCT** (Song & Dhariwal) made it competitive with no teacher: drop the EMA target ($\theta^- = \text{sg}(\theta)$), Pseudo-Huber loss instead of LPIPS, a lognormal noise schedule, and a curriculum that doubles the number of discretization steps. CIFAR-10 2.51, ImageNet-64 3.25 in one step.

## 2.2 Continuous time: ECT and sCM

Shrink the gap between the two points to zero and the loss becomes a differential condition: $\frac{d}{dt} f_\theta(x_t, t) = 0$ along the ODE. Its gradient needs the total derivative

\begin{equation}
\frac{d f_\theta}{dt} = \nabla_x f_\theta \cdot \frac{dx_t}{dt} + \partial_t f_\theta,
\end{equation}

a **Jacobian-vector product (JVP)**: one forward-mode pass, about the cost of a second forward.

- **ECT** (Geng et al.) writes consistency as this differential equation and tightens the gap on a curriculum, starting from a pretrained diffusion model (at gap $= t$ the loss is ordinary denoising). ImageNet-64 2-step 1.67.
- **sCM** (Lu & Song) made continuous-time CMs train stably at scale with a TrigFlow parameterization ($x_t = \cos t\, x_0 + \sin t\, z$), identity time embeddings, adaptive double normalization, tangent normalization $g / (\|g\| + 0.1)$, and a learned loss weight. At 1.5B parameters on ImageNet 512: consistency training (sCT) gets 4.29 / 3.76 at 1 / 2 steps, consistency distillation (sCD) 2.28 / 1.88.

## 2.3 Two-time flow maps

A consistency model only jumps to $t = 0$. Its quality does not reliably improve with more steps, because a 2-step CM sampler jumps to 0, re-noises, and jumps again. Learning $\Phi_{t \to s}$ for any $s$ fixes that: more steps simply means shorter jumps along the same flow.

**CTM** (Kim et al.) learned $\Phi_{t \to s}$ first, but its headline results are distilled and add a GAN loss (ImageNet-64 1-step 1.92). **Flow Map Matching** and **"How to build a consistency model"** (Boffi, Albergo, Vanden-Eijnden) gave the clean taxonomy. You can train $\Phi_{t\to s}$ by enforcing any of three characterizations:

\begin{align}
&\text{Lagrangian (vary the end time):} && \partial_s \Phi_{t\to s}(x) = v\big(\Phi_{t\to s}(x), s\big) \\
&\text{Eulerian (vary the start time):} && \partial_t \Phi_{t\to s}(x) + \nabla_x \Phi_{t\to s}(x)\, v(x,t) = 0 \\
&\text{Progressive (composition):} && \Phi_{t\to r} = \Phi_{s\to r}\circ\Phi_{t\to s}
\end{align}

with $v$ supplied by the model's own diagonal ($s = t$), trained by flow matching. Consistency models are the Eulerian condition at $s = 0$. The methods below are all one of these three.

## 2.4 Shortcut models

**Shortcut models** (Frans et al.) condition the network on a step size $d$ and use the composition property: one step of size $2d$ must equal two steps of size $d$, the second taken from where the first lands. At $d = 0$ the network is trained with plain flow matching. No JVP, no teacher. ImageNet 256, DiT-XL: 1-step FID 10.6, 4-step 7.8.

## 2.5 MeanFlow

**MeanFlow** (Geng, Deng, Bai, Kolter, He, 2025) learns the average velocity $u(x_t, r, t)$. Start from its definition:

\begin{equation}
(t - r)\, u(x_t, r, t) = \int_r^t v(x_\tau, \tau)\, d\tau
\end{equation}

Differentiate both sides with respect to $t$ (holding $r$ fixed):

\begin{equation}
u(x_t, r, t) + (t - r)\, \frac{d}{dt} u(x_t, r, t) = v(x_t, t)
\end{equation}

which gives the **MeanFlow identity**:

\begin{equation}
\boxed{\, u(x_t, r, t) = v(x_t, t) - (t - r)\, \frac{d}{dt} u(x_t, r, t) \,}, \qquad \frac{du}{dt} = v\, \partial_x u + \partial_t u
\end{equation}

The right-hand side becomes a regression target. Replace the marginal $v$ by the conditional velocity $\epsilon - x_0$ (the same trick that makes flow matching work), compute $\frac{du}{dt}$ with a JVP, and stop the gradient through the target:

\begin{equation}
\mathcal{L} = \Big\| u_\theta(x_t, r, t) - \text{sg}\big( (\epsilon - x_0) - (t - r)\,(\,(\epsilon - x_0)\,\partial_x u_\theta + \partial_t u_\theta\,) \big) \Big\|^2
\end{equation}

In JAX this is a few lines, since `jax.jvp` gives the forward pass and the tangent together:

```python
def meanflow_loss(params, x0, eps, r, t, c=1e-3):
    # x0, eps: [B, H, W, C]; r, t: [B, 1, 1, 1] with r <= t
    xt = (1 - t) * x0 + t * eps
    v = eps - x0                                   # conditional velocity
    u_fn = lambda x, r_, t_: model(params, x, r_, t_)
    # tangent direction (dx/dt, dr/dt, dt/dt) = (v, 0, 1)
    u, dudt = jax.jvp(u_fn, (xt, r, t), (v, jnp.zeros_like(r), jnp.ones_like(t)))
    target = jax.lax.stop_gradient(v - (t - r) * dudt)
    err = jnp.sum((u - target) ** 2, axis=(1, 2, 3))
    w = jax.lax.stop_gradient(1.0 / (err + c))     # adaptive weight, p = 1, c a small constant
    return jnp.mean(w * err)
```

Practical details from the paper: sample $(r, t)$ from a logit-normal distribution with $r = t$ (pure flow matching) for 75% of samples; use the adaptive weight $1/(\|\Delta\|^2 + c)^p$; and **bake classifier-free guidance into the target** (mix conditional and unconditional velocities in $v$ at a fixed scale), so sampling stays at 1 NFE. ImageNet 256, DiT-XL/2 (676M): **1-NFE FID 3.43**, 2-NFE 2.20 with longer training.

MeanFlow started a busy line of follow-ups, all on ImageNet 256 with DiT-style backbones:

- **AlphaFlow** splits the MeanFlow loss into a trajectory flow-matching term and a trajectory consistency term whose gradients conflict, and interpolates with a parameter $\alpha$: $\alpha = 1$ is flow matching, $\alpha = \tfrac12$ is the Shortcut target, $\alpha \to 0$ recovers MeanFlow. A curriculum from 1 to 0: 1-NFE 2.58.
- **SplitMeanFlow** uses the algebraic version of the identity, $(t-r)\,u(x_t,r,t) = (t-s)\,u(x_t,s,t) + (s-r)\,u(x_s,r,s)$, which needs no JVP. MeanFlow's identity is its infinitesimal limit.
- **Decoupled MeanFlow** turns a trained flow model into a flow map by conditioning only the last blocks on the target time: 1-NFE 2.16, 2-NFE 1.64.
- **Terminal Velocity Matching** enforces the Lagrangian condition at the end time instead, with a Wasserstein bound and a fused attention kernel that supports the JVP: 1-NFE 3.29, 4-NFE 1.99.
- **SoFlow** (with Zhuang Liu) replaces the JVP with a consistency loss at a shifted point: 2.96 vs MeanFlow's 3.43 at equal training.
- **iMF (improved MeanFlow)** rewrites the self-referential target as a loss on velocity, and makes the guidance scale and interval **network inputs**, so CFG stays adjustable at test time at 1 NFE: **1-NFE 1.72** (XL/2, 610M), 2-NFE 1.54.
- **Pixel MeanFlow** drops the VAE and generates pixels directly, predicting $x$ and converting to $u$, with added LPIPS and ConvNeXt perceptual losses: 2.22 at 256, 2.48 at 512.

## 2.6 Matching distributions instead of trajectories

Every method above matches individual trajectories. A second branch matches **distributions** of jumps.

**Inductive Moment Matching** (IMM; Zhou, Ermon, Song) learns $\Phi_{t\to s}$ but trains it with a kernel distance (MMD, maximum mean discrepancy) between groups of samples pushed from $t$ and from a nearby $r$. The CM loss is its single-sample special case, which the authors give as the reason CMs are unstable. ImageNet 256: 1 / 2 / 8 steps = 8.05 / 3.99 / 1.99.

The current ImageNet records come from this branch, and drop the ODE altogether:

- **Drifting Models** (Deng, ..., He, 2026) train a one-step generator by moving its samples along a "drift field": attraction toward data, repulsion from other generated samples, zero when the two distributions match. **1-NFE FID 1.54.**
- **W-Flow** uses a Wasserstein (Sinkhorn) gradient flow as the drift: 1.29.
- **Priming Models** (posted October 2026) combine a fixed-noise denoising stage with one-step optimal-transport post-training: 1.19 in pixel space.

<div class="callout callout-warn">

"From scratch" here means **no diffusion teacher**, not no pretrained network. Drifting reports that ImageNet did not work without computing its drift in a pretrained self-supervised (MAE) feature space; W-Flow and Priming reuse those features, and Pixel MeanFlow and Priming add perceptual losses built on pretrained networks. These are real gains, but they are not the same setting as MeanFlow or iMF.

</div>

# 3. Distillation: learning from a teacher

If a good many-step model already exists (and for text-to-image and video, one always does), distilling it is usually cheaper than training a one-step model from scratch. There are three families, distinguished by the loss.

## 3.1 Trajectory distillation: regress onto the teacher's ODE

The student imitates the teacher's ODE solution from the same noise.

- **Knowledge distillation** (Luhman & Luhman, 2021): a one-step student regresses the endpoint of a full DDIM solve. It works, but samples are blurry, which the authors attribute to the regression objective.
- **Progressive distillation** (Salimans & Ho, 2022): a student learns to do two teacher steps in one; repeat, halving the steps each round. It also introduced v-prediction. **Guided distillation** (Meng et al.) first folds the CFG pair into one guidance-conditioned model, then distills progressively.
- **Consistency distillation** and its latent-space version **LCM** / **LCM-LoRA** (2-4-step Stable Diffusion, the distillation shipped as a plug-in LoRA), with refinements like PCM, TCD and Hyper-SD.
- **Reflow** (rectified flow, from the previous post) and **InstaFlow**: retrain on teacher pairs until the flow is straight, then one Euler step works.
- **Flow map distillation**: **Align Your Flow** (Sabour, Fidler, Kreis) distills the two-time map with Eulerian and Lagrangian objectives (ImageNet 512, 2 NFE 1.87); **Flow Map Distillation Without Data** reaches ImageNet 256 1-NFE 1.45.

Strengths: stable, keeps the teacher's noise-to-image coupling, so diversity is preserved. Weakness: at 1-2 steps the student's errors show up as blur, and errors along the trajectory accumulate.

## 3.2 Distribution matching: reverse KL through a score difference

Instead of matching the teacher's *trajectory*, match its *distribution*. Let $G_\theta(z)$ be the one-step generator, and diffuse its output to $x_t$. The gradient of the KL divergence between the generator's noised distribution and the teacher's is a **difference of two scores**:

\begin{equation}
\nabla_\theta\, D_{\text{KL}}\big(p^{\text{fake}}_t \,\|\, p^{\text{real}}_t\big) = \mathbb{E}_{z,\, t,\, x_t}\Big[ w_t\, \big(s_{\text{fake}}(x_t, t) - s_{\text{real}}(x_t, t)\big)^{\top} \frac{\partial G_\theta(z)}{\partial \theta} \Big]
\end{equation}

$s_{\text{real}}$ is the frozen teacher. $s_{\text{fake}}$ is a second diffusion model trained online on the generator's own samples. No ODE solves needed.

- The idea goes back to **VSD** (ProlificDreamer, text-to-3D) and **Diff-Instruct**.
- **DMD** (Yin et al., CVPR 2024) used it for one-step image generation, plus a regression loss on teacher pairs to stop mode collapse.
- **DMD2** removed the regression (and its expensive pair dataset), stabilized training by updating the fake-score network more often than the generator, and added a GAN loss on real images: ImageNet-64 1-step **1.28**, also 4-step SDXL.
- **SiD** (score identity distillation) derives a different loss from score identities, data-free: ImageNet-64 1.52. **SiDA** adds an adversarial term and real data: 1.11, and 1.37 on ImageNet 512, one step, no CFG.
- **f-distill** generalizes reverse KL to any f-divergence (forward KL and Jensen-Shannon cover modes better): ImageNet-64 1.16.
- **Decoupled DMD** found that for text-to-image, most of DMD's effect comes from the CFG term folded into the target, and the distribution-matching part acts as a regularizer.

The known weakness: reverse KL is **mode-seeking**. The student can collapse onto a subset of what the teacher generates, and loses diversity.

## 3.3 Adversarial distillation

Train the student with a discriminator, usually alongside a distillation loss.

- **ADD** (SDXL-Turbo): a GAN loss with a frozen DINOv2 feature discriminator on real images, plus a score-distillation loss. 1-4 steps.
- **LADD** (SD3-Turbo): the discriminator uses the teacher's own latent features instead of pixel features, so it scales to an 8B model.
- **SDXL-Lightning**: progressive distillation with an adversarial loss in place of MSE.
- **Seaweed-APT**: adversarial post-training against real data with no distillation loss at all, for **one-step 2-second 720p video in real time**.

GANs give the sharpest one-step samples and can exceed the teacher because they see real data. The price is the usual GAN instability and discriminator engineering.

## 3.4 Hybrids, and what is used for video

Modern recipes combine families so each covers another's failure: DMD2 is distribution matching + GAN; CTM is trajectory + score matching + GAN; SANA-Sprint is sCM + LADD. For large video models two threads stand out:

- **rCM** (NVIDIA): continuous-time consistency distillation scaled past 10B parameters with a FlashAttention-2 JVP kernel. sCM alone lost fine detail (the authors attribute it to sCM's mode-covering divergence), so they add DMD-style reverse KL as a regularizer. On Wan2.1-14B: VBench 84.92 / 85.05 / 83.02 at 4 / 2 / 1 steps against the teacher's 83.58 at 100 NFE.
- **Causal video**: **CausVid** distills a bidirectional teacher into a causal, autoregressive student with DMD; **Self Forcing** trains the student on its own generated frames (with a KV cache during training), which removes the mismatch between training on real frames and generating from its own.

Newer directions include **π-Flow**, where the student outputs a small network-free "policy" that produces velocities for several sub-steps (ImageNet 256 1-NFE 2.85; at 4 NFE on FLUX, better diversity than DMD), and **TwinFlow**, a self-adversarial flow with no frozen teacher, no fake-score network and no discriminator.

# 4. Where the numbers stand

ImageNet 256×256, class-conditional, FID-50K at **1 NFE** (lower is better). For scale: the original DiT-XL/2 needed 250 steps with guidance (500 NFE) to reach 2.27.

| Method | Date | Teacher? | 1-NFE FID |
|---|---|---|---|
| Shortcut models (XL) | 2024-10 | no | 10.6 |
| IMM (XL/2) | 2025-03 | no | 7.77 |
| MeanFlow (XL/2) | 2025-05 | no | 3.43 |
| Terminal Velocity Matching (XL/2) | 2025-11 | no | 3.29 |
| SoFlow (XL/2) | 2025-12 | no | 2.96 |
| π-Flow | 2025-10 | yes | 2.85 |
| AlphaFlow (XL/2+) | 2025-10 | no | 2.58 |
| Pixel MeanFlow (H/16, pixels) | 2026-01 | no (perceptual losses) | 2.22 |
| Decoupled MeanFlow (XL/2+) | 2025-10 | own flow model first | 2.16 |
| iMF (XL/2) | 2025-12 | no | 1.72 |
| FACM | 2025-07 | yes | 1.70 |
| Drifting (L/2) | 2026-02 | no (MAE features) | 1.54 |
| Flow Map Distillation Without Data | 2025-11 | yes | 1.45 |
| W-Flow (XL/2) | 2026-05 | no (MAE features) | 1.29 |
| Priming (L, pixels) | 2026-10 | no (features + perceptual) | 1.19 |

ImageNet 64×64 at 1 step, mostly distillation: CD 6.20, DMD 2.62, CTM 1.92, SiD 1.52, DMD2 1.28, f-distill 1.16, SiDA 1.11. From scratch: iCT 3.25, sCT 2.04.

Treat these as a map, not a leaderboard: models differ in size, training epochs, guidance handling, and which pretrained networks they use, and papers sometimes re-evaluate each other's baselines differently.

# 5. Takeaways

1. **Every one-step method learns the flow map** $\Phi_{t\to s}$ or a slice of it. The differences are which slice (endpoint, two-time map, average velocity) and which property supervises it (boundary, composition, Eulerian or Lagrangian derivative).
2. **Velocity regression alone can't give one step**: its one-step answer is a conditional mean. Something must supply the endpoint: a teacher's ODE, self-consistency, or a distribution-level loss.
3. **Two-time maps beat endpoint-only models** when you want quality to improve with more steps.
4. **From scratch has caught up.** In 2023 one-step ImageNet 256 from scratch was far behind distillation; by late 2025 iMF reached 1.72 with no teacher and no extra networks, and 2026 methods with pretrained features reach 1.2-1.5.
5. **Distillation still rules large text-to-image and video models**, where a strong teacher exists and training from scratch is expensive. The recipe there is a hybrid: continuous-time consistency or flow maps for coverage, plus a reverse-KL or adversarial term for sharpness.
6. **Mind the JVP.** Continuous-time objectives (sCM, MeanFlow, TVM, rCM) need forward-mode derivatives through the whole network. In JAX that is one `jax.jvp` call; with custom attention kernels it means writing the kernel's JVP rule too (rCM and TVM both did). JVP-free alternatives (Shortcut, SplitMeanFlow, SoFlow, TiM's finite differences) exist for this reason.

# References

**Consistency models and flow maps**
- Song, Dhariwal, Chen, Sutskever. *Consistency Models.* ICML 2023. [arXiv:2303.01469](https://arxiv.org/abs/2303.01469)
- Song, Dhariwal. *Improved Techniques for Training Consistency Models.* [arXiv:2310.14189](https://arxiv.org/abs/2310.14189)
- Kim et al. *Consistency Trajectory Models.* ICLR 2024. [arXiv:2310.02279](https://arxiv.org/abs/2310.02279)
- Boffi, Albergo, Vanden-Eijnden. *Flow Map Matching with Stochastic Interpolants.* [arXiv:2406.07507](https://arxiv.org/abs/2406.07507)
- Geng et al. *Consistency Models Made Easy.* ICLR 2025. [arXiv:2406.14548](https://arxiv.org/abs/2406.14548)
- Lu, Song. *Simplifying, Stabilizing and Scaling Continuous-Time Consistency Models.* ICLR 2025. [arXiv:2410.11081](https://arxiv.org/abs/2410.11081)
- Frans, Hafner, Levine, Abbeel. *One Step Diffusion via Shortcut Models.* ICLR 2025. [arXiv:2410.12557](https://arxiv.org/abs/2410.12557)
- Zhou, Ermon, Song. *Inductive Moment Matching.* ICML 2025. [arXiv:2503.07565](https://arxiv.org/abs/2503.07565)
- Boffi, Albergo, Vanden-Eijnden. *How to Build a Consistency Model: Learning Flow Maps via Self-Distillation.* [arXiv:2505.18825](https://arxiv.org/abs/2505.18825)
- Wang et al. *Transition Models: Rethinking the Generative Learning Objective.* [arXiv:2509.04394](https://arxiv.org/abs/2509.04394)

**MeanFlow and follow-ups**
- Geng, Deng, Bai, Kolter, He. *Mean Flows for One-step Generative Modeling.* NeurIPS 2025. [arXiv:2505.13447](https://arxiv.org/abs/2505.13447)
- Guo et al. *SplitMeanFlow: Interval Splitting Consistency in Few-Step Generative Modeling.* [arXiv:2507.16884](https://arxiv.org/abs/2507.16884)
- Zhang et al. *AlphaFlow: Understanding and Improving MeanFlow Models.* [arXiv:2510.20771](https://arxiv.org/abs/2510.20771)
- Lee, Yu, Shin. *Decoupled MeanFlow: Turning Flow Models into Flow Maps for Accelerated Sampling.* [arXiv:2510.24474](https://arxiv.org/abs/2510.24474)
- Zhou et al. *Terminal Velocity Matching.* [arXiv:2511.19797](https://arxiv.org/abs/2511.19797)
- Geng et al. *Improved Mean Flows: On the Challenges of Fastforward Generative Models.* [arXiv:2512.02012](https://arxiv.org/abs/2512.02012)
- Luo, Yuan, Liu. *SoFlow: Solution Flow Models for One-Step Generative Modeling.* [arXiv:2512.15657](https://arxiv.org/abs/2512.15657)
- Lu et al. *One-step Latent-free Image Generation with Pixel Mean Flows.* [arXiv:2601.22158](https://arxiv.org/abs/2601.22158)

**Distribution-level one-step generators**
- Deng et al. *Generative Modeling via Drifting.* [arXiv:2602.04770](https://arxiv.org/abs/2602.04770)
- Han et al. *One-Step Generative Modeling via Wasserstein Gradient Flows.* [arXiv:2605.11755](https://arxiv.org/abs/2605.11755)
- Park et al. *Are We Overthinking One-Step Generation?* [arXiv:2610.20331](https://arxiv.org/abs/2610.20331)

**Trajectory distillation**
- Luhman, Luhman. *Knowledge Distillation in Iterative Generative Models for Improved Sampling Speed.* [arXiv:2101.02388](https://arxiv.org/abs/2101.02388)
- Salimans, Ho. *Progressive Distillation for Fast Sampling of Diffusion Models.* ICLR 2022. [arXiv:2202.00512](https://arxiv.org/abs/2202.00512)
- Meng et al. *On Distillation of Guided Diffusion Models.* CVPR 2023. [arXiv:2210.03142](https://arxiv.org/abs/2210.03142)
- Luo et al. *Latent Consistency Models.* [arXiv:2310.04378](https://arxiv.org/abs/2310.04378)
- Liu et al. *InstaFlow: One Step is Enough for High-Quality Diffusion-Based Text-to-Image Generation.* ICLR 2024. [arXiv:2309.06380](https://arxiv.org/abs/2309.06380)
- Sabour, Fidler, Kreis. *Align Your Flow: Scaling Continuous-Time Flow Map Distillation.* NeurIPS 2025. [arXiv:2506.14603](https://arxiv.org/abs/2506.14603)
- Peng et al. *FACM: Flow-Anchored Consistency Models.* [arXiv:2507.03738](https://arxiv.org/abs/2507.03738)
- Tong et al. *Flow Map Distillation Without Data.* [arXiv:2511.19428](https://arxiv.org/abs/2511.19428)

**Distribution matching**
- Wang et al. *ProlificDreamer (Variational Score Distillation).* NeurIPS 2023. [arXiv:2305.16213](https://arxiv.org/abs/2305.16213)
- Luo et al. *Diff-Instruct.* [arXiv:2305.18455](https://arxiv.org/abs/2305.18455)
- Yin et al. *One-step Diffusion with Distribution Matching Distillation.* CVPR 2024. [arXiv:2311.18828](https://arxiv.org/abs/2311.18828)
- Yin et al. *Improved Distribution Matching Distillation for Fast Image Synthesis (DMD2).* NeurIPS 2024. [arXiv:2405.14867](https://arxiv.org/abs/2405.14867)
- Zhou et al. *Score identity Distillation.* ICML 2024. [arXiv:2404.04057](https://arxiv.org/abs/2404.04057)
- Zhou et al. *Adversarial Score identity Distillation.* [arXiv:2410.14919](https://arxiv.org/abs/2410.14919)
- Salimans et al. *Multistep Distillation of Diffusion Models via Moment Matching.* [arXiv:2406.04103](https://arxiv.org/abs/2406.04103)
- Xu, Nie, Vahdat. *One-step Diffusion Models with f-Divergence Distribution Matching.* [arXiv:2502.15681](https://arxiv.org/abs/2502.15681)
- Liu et al. *Decoupled DMD: CFG Augmentation as the Spear, Distribution Matching as the Shield.* [arXiv:2511.22677](https://arxiv.org/abs/2511.22677)

**Adversarial and hybrid**
- Sauer et al. *Adversarial Diffusion Distillation (SDXL-Turbo).* ECCV 2024. [arXiv:2311.17042](https://arxiv.org/abs/2311.17042)
- Sauer et al. *Fast High-Resolution Image Synthesis with Latent Adversarial Diffusion Distillation (SD3-Turbo).* [arXiv:2403.12015](https://arxiv.org/abs/2403.12015)
- Lin et al. *SDXL-Lightning: Progressive Adversarial Diffusion Distillation.* [arXiv:2402.13929](https://arxiv.org/abs/2402.13929)
- Lin et al. *Diffusion Adversarial Post-Training for One-Step Video Generation.* ICML 2025. [arXiv:2501.08316](https://arxiv.org/abs/2501.08316)
- Chen et al. *SANA-Sprint: One-Step Diffusion with Continuous-Time Consistency Distillation.* ICCV 2025. [arXiv:2503.09641](https://arxiv.org/abs/2503.09641)
- Yin et al. *From Slow Bidirectional to Fast Autoregressive Video Diffusion Models (CausVid).* CVPR 2025. [arXiv:2412.07772](https://arxiv.org/abs/2412.07772)
- Huang et al. *Self Forcing: Bridging the Train-Test Gap in Autoregressive Video Diffusion.* [arXiv:2506.08009](https://arxiv.org/abs/2506.08009)
- Zheng et al. *Large Scale Diffusion Distillation via Score-Regularized Continuous-Time Consistency (rCM).* [arXiv:2510.08431](https://arxiv.org/abs/2510.08431)
- Chen et al. *pi-Flow: Policy-Based Few-Step Generation via Imitation Distillation.* [arXiv:2510.14974](https://arxiv.org/abs/2510.14974)
- Cheng et al. *TwinFlow: Realizing One-step Generation on Large Models with Self-adversarial Flows.* [arXiv:2512.05150](https://arxiv.org/abs/2512.05150)
