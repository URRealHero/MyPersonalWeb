+++
title = 'Style guide: everything a post can do'
date = 2026-11-20
summary = 'A draft-only page that shows every writing tool on this site. It never gets published; run `hugo server -D` to see it, and copy what you need.'
kind = 'note'
status = 'evolving'
tags = ['meta']
draft = true
# Reference page only: viewable at its URL with `hugo server -D`, but never listed
# on the homepage, in the nav, or under Linked from.
[build]
  list = 'never'
  publishResources = false
+++

This page is a **draft**, so it only exists when you run `hugo server -D`. Copy any block below into a real post. Start a new post with `hugo new blog/my-post` (or `hugo new log/my-thread` for a research log thread). Either command creates a folder; put images next to `index.md`.

## Links between notes

Link to other pages by their site path: [the log](/log/), or [a paper](/research/memorization-3d-gen/) (paper links go straight to the publication). A post or log thread you link to lists this one under **Linked from** automatically. A typo like `/log/my-thred/` makes `hugo` print a *broken internal link* warning, so mistakes don't slip through.

## Callouts

{{< callout hypothesis >}}
Longer latent sets reduce memorization because each shape gets more capacity to be *encoded* rather than *indexed*.
{{< /callout >}}

{{< callout type="result" title="EMA 0.9999 helps at 1 NFE" >}}
FID 4.1 → 3.6 at 200k steps. Seeds: 3. Variance across seeds ±0.1.
{{< /callout >}}

{{< callout question >}}
Does the effect survive at 4× the data?
{{< /callout >}}

{{< callout todo >}}
Rerun with rotation augmentation off, same seed.
{{< /callout >}}

{{< callout caveat >}}
Evaluated on 10k samples only; FID is biased at that size.
{{< /callout >}}

Also available: `takeaway` and `note`.

## Sidenotes and footnotes

A sidenote sits in the margin on wide screens{{< sidenote >}}And opens on tap on phones. *Markdown* and $\LaTeX$ both work here.{{< /sidenote >}} and doesn't interrupt the sentence. A footnote collects at the end instead.[^fn]

[^fn]: Use footnotes for references and long asides, sidenotes for short ones you want read in place.

## Math

Inline: the flow-matching path is $x_t = (1-t)\,x_0 + t\,\epsilon$. Display math is numbered and referenceable:

\begin{equation}
\mathcal{L}_{\text{FM}}(\theta) = \E_{t,\,x_0,\,\epsilon}\big\| v_\theta(x_t, t) - (\epsilon - x_0) \big\|^2
\label{eq:fm}
\end{equation}

Then refer to it as \eqref{eq:fm}. The shortcuts `\E`, `\R`, `\N` and `\KL` are predefined.

## Figures

A Markdown image with a title becomes a numbered figure:

![Straight vs curved paths](toy-paths.svg "Training paths are straight between paired points (left), but the marginal velocity field makes sampling paths curve (right).")

Several images side by side, or a video, using `fig`:

{{< fig src="toy-paths.svg,toy-paths.svg" caption="Two panels in one figure. Separate sources with commas." >}}

## Before → after

{{< dumbbell from="Baseline" to="Ours" max=10 caption="`dumbbell` takes one row per metric: label | before | after." >}}
FID ↓ | 6.2 | 3.6
IS ↑ (÷10) | 7.1 | 8.4
{{< /dumbbell >}}

## Code

```python
def euler_sample(v, x, steps=48):
    for i in range(steps, 0, -1):
        t, dt = i / steps, 1 / steps
        x = x - dt * v(x, t)   # integrate t: 1 → 0
    return x
```

## Tables and quotes

| Method | NFE | FID ↓ |
|---|---:|---:|
| Euler | 48 | 3.1 |
| Ours | 1 | 3.6 |

> A good research note says what you believed, what you did, and what changed your mind.

---

That's it. Delete nothing here; it never ships.
