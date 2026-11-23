+++
title = 'Example thread: does EMA help one-step generators?'
date = 2026-11-20
started = 2026-11-10
status = 'active'
question = 'Is the gain from EMA larger when the model takes **one** sampling step than when it takes many?'
repo = 'https://github.com/URRealHero'
paper = '/research/memorization-3d-gen'
tags = ['meta']
draft = true
# Reference page only: viewable at its URL with `hugo server -D`, but never listed
# on the homepage, in the nav, or under Linked from.
[build]
  list = 'never'
  publishResources = false
+++

{{< callout state >}}
*This is a draft-only example of a research-log thread. It never gets published.* EMA seems to help more at 1 NFE than at 48. The effect is still within two seeds' noise, so next is a third seed and a longer run.
{{< /callout >}}

{{< callout hypothesis >}}
One-step models are more sensitive to weight noise, so EMA should help them more. If the gap closes with more seeds, the hypothesis is wrong.
{{< /callout >}}

## 2026-11-18 · third seed, and the gap narrows

The third seed brings the 1-NFE gain down from 0.6 to 0.4 FID.

{{< dumbbell from="no EMA" to="EMA 0.9999" max=8 caption="FID (lower is better), mean of three seeds." >}}
1 NFE | 6.1 | 5.7
48 NFE | 3.3 | 3.2
{{< /dumbbell >}}

{{< callout todo >}}
Run 400k steps before believing anything.
{{< /callout >}}

## 2026-11-14 · first numbers

Two seeds. EMA helps at 1 NFE (−0.6 FID) and barely at 48 NFE (−0.1).{{< sidenote >}}Same data and model, same 200k steps. Only the EMA decay changes.{{< /sidenote >}}

## 2026-11-10 · setup

Toy setup: a small DiT on CIFAR-10, flow-matching loss, Euler sampler. Compare EMA decay 0 vs 0.9999 at 1 and 48 steps.
