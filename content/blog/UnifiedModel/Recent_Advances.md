+++
date = '2025-06-05T23:00:30+08:00'
draft = false
title = 'Advanced Unified-Models'
summary = 'Recording the reading progress of recent advanced unified-models and diffusion-LLMs'
+++

# Diffusion-LLMs
Just thinking like what diffusion model do in pure image generation
- Masking paradigm 1: Category distribution
    The core principle is simple:
    1.  Take a piece of perfect data (like a sentence).
    2.  Systematically and slowly destroy it by adding "noise" until it's unrecognizable.
    3.  Train the model to learn how to precisely reverse that destruction process.
    
    How to corrupt a discrete token? We have one additional '[MASK]' token, which is used to replace the corrupted token.

    \begin{equation}
        q(x_t | \mathbf{x}) = \text{Cat}(\alpha_t \mathbf{x} + (1 - \alpha_t)\mathbf{m})
    \end{equation}

    Where $\mathbf{x}$ is a one-hot vector of the original token, $\mathbf{m}$ is a one-hot vector of the masked token, and $\alpha_t$ is a scalar that controls the amount of noise added. $\alpha_t$ is a schedular that controls the amount of noise added at each step.

    Then how do we reverse(denoise) the corrupted token?

    \begin{equation}
    q(x_s | x_t, \mathbf{x}) =
    \begin{cases}
        \text{Cat}(x_s | x_t), & \text{if } x_t \neq \mathbf{m} \\\
        \text{Cat}\left(x_s \left| \frac{(\alpha_s - \alpha_t)\mathbf{x} + (1 - \alpha_s)\mathbf{m}}{1 - \alpha_t}\right.\right), & \text{if } x_t = \mathbf{m}
    \end{cases}
    \end{equation}


## Unified Discrete Diffusion Family

### [UniDisc](https://unidisc.github.io/): Unified Multimodal Discrete Diffusion

![Fail to load its arch](/images/blogs/UniDisc.png)

### [Muddit](https://arxiv.org/abs/2505.23606): Liberating Generation Beyond Text-to-Image with a Unified Discrete Diffusion Model 
- Using one unified discrete diffusion for image and text generation
![Fail to load its architecture](/images/blogs/Muddit.png "Muddit Architecture")

#### Related Works
- [MaskGIT](https://masked-generative-image-transformer.github.io/)
- [Meissonic](https://arxiv.org/abs/2410.08261)


### Llada Series
- [LLADA 1](https://arxiv.org/abs/2502.09992)
- [LLADA V](https://arxiv.org/abs/2505.16933)
- [LLADA 1.5](https://arxiv.org/abs/2505.19223)

## [Dual Diffusion](https://arxiv.org/abs/2501.00289) for Text-and-Image Generation


## Proprietary
- [Gemini Diffusion](https://deepmind.google/models/gemini-diffusion/)
- [GPT-4o](https://arxiv.org/abs/2504.05979)

## Diffusion-LLM Theory
- [A Convergence Theory for Diffusion Language Models: An Information-Theoretic Perspective](https://arxiv.org/abs/2505.21400)

## Diffusion-LLM Acceleration
- [Fast DLLM](https://nvlabs.github.io/Fast-dLLM/)
- [dLLM-Cache](https://github.com/maomaocun/dLLM-cache)