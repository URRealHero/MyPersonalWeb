+++
date = '2025-05-25T16:42:37+08:00'
draft = false
title = 'Diffusion-Series'
summary = "This is my Diffusion-series, including all my understanding of diffusion models"
image= "/images/blogs/DDPM_intro.png"
+++

My notes on diffusion models, from the basics to generating in a single step.

1. [DDIM](ddim/): making the DDPM sampler non-Markovian and deterministic, so it can skip steps.
2. [Flow Matching](flowmatching/): diffusion as a velocity field along a chosen path; why straight training paths still give curved sampling paths.
3. [One Step](consistencymodel/): consistency models, flow maps, MeanFlow, and distilling a many-step model into a one-step generator.