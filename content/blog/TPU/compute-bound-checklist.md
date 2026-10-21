+++
date = '2026-10-18T12:00:00-07:00'
draft = true
title = 'Making a Model Compute-Bound on TPU: A Stage-by-Stage Checklist'
summary = 'What to measure and what to change, from launch to the attention kernel, so a JAX model spends its time on the matrix unit. Drawn from speeding up a 33B video diffusion transformer, a Qwen3.8 captioner, and a VGGT training run on TPU v6e.'
tags = ["TPU", "JAX", "Pallas", "Performance", "MFU"]
+++

Porting a model to TPU is the easy half. The first JAX version that matches the GPU reference usually runs at 10-20% of the chip's peak. Getting it to 40-50% is a long list of small changes, each found by measuring where the time goes, fixing one thing, and checking that the output did not change.

This post is that list, written as a checklist you can walk through stage by stage (launch, input, layout across chips, the step itself, memory, the whole job) and module by module inside the step (attention, matmuls, layout glue, elementwise ops, conditioning, heads, optimizer). Every item comes from one of three projects where I kept a log of every version, its time, and whether it was kept:

| Project | Workload | Hardware | Start | End |
|---|---|---|---|---|
| **MiniMax-H3** JAX port | Inference: 33B diffusion transformer, text/image → 10 s video + audio, 74k-token sequences, batch 1 | v6e-4 (4 chips) | 14.66 s/step, 19.8% MFU | **6.14 s/step, 47.4% MFU** |
| | same, multi-host | v6e-64 | 1.36 s/step | **0.41 s/step, 44% MFU** |
| **Qwen3.8-27B** on vLLM-TPU | Inference: video captioning and prompt rewriting for the H3 pipeline | v6e-4 | 11.4 s/clip | **0.88 s/clip** |
| **VGGT encoder** (LagerNVS fine-tuning) | Training: VGGT-1B encoder + 12-block renderer, 16 views at 518 px | v6e-8 | 3.354 s/micro-step | **1.385 s/micro-step (2.42×)** |

MFU (model FLOPs utilization) is the useful FLOPs of the model divided by elapsed time × chips × peak FLOP/s. Peak on a v6e chip is 918 TFLOP/s in bf16.

The two big projects ended up teaching opposite lessons about the same module, which is the main point of this post: **a trick that pays off on one model can cost time on another, because they are bound by different units of the chip.** So the checklist below says what to *check*, not just what to *do*.

---

## 0. Before changing anything: build the measuring stick

Most wasted effort in performance work comes from comparing numbers that were not comparable. Set these up first.

### A fixed, recorded workload

- **Replay recorded inputs.** For H3, a script runs one denoising step on the inputs recorded from the reference pipeline at step 24, so every version does identical work. For training, use the real production shape (VGGT used the shape of its first long training run), not a toy shape. Small shapes hide padding waste and memory pressure.
- **Count FLOPs analytically.** Write down `2·m·n·k` per matmul and `4·Lq·Lk·D` per attention (forward). For training with full rematerialization, a step is about four forward passes (forward + recomputed forward + a backward worth two). Divide by time to get MFU. Do not trust the compiler's cost analysis for Pallas kernels: they have no cost estimate.
- **Time like an experimenter.** Run A and B back to back in the same job. Drop the first 1-3 calls (H3's first calls took 1.03-1.26 s against a steady 0.91 s). A profiler trace adds about 2%, so compare traced with traced. VGGT ran the same arm twice and saw 0.04% spread: that is your noise floor.
- **One commit per version, with the number in the commit message.** Keep rejected versions on branches. Thirty versions later you will want to know why v28 was slower.

### An accept rule written before the run

Decide what "better" means before you look: faster on *every* measured shape, fits in memory, passes the correctness check. VGGT's rule rejected changes that helped one shape and hurt another.

### A correctness gate per version

Speed changes break numerics silently. Sort every change into one of three classes:

| Class | Example | Gate |
|---|---|---|
| **Bitwise** (reordering only) | donating buffers, one-hot matmul instead of gather, scheduling hints | outputs hash-identical |
| **Rounding** (fewer or different roundings) | new kernel tiles, f32 accumulation in a new place | error vs float32 truth no worse than the reference's own bf16 error |
| **Approximate** (drops computation) | sparse attention, lower precision | opt-in only, never the default |

For H3 the gate was teacher-forced velocities against the reference at 5 steps (mean relative L2 went from 2.03% at v0 to 1.87% at the end, so it got *more* accurate). For VGGT it was step-0 and step-1 losses (final kernel: 6.0e-6 relative to the reference run) plus a kernel test against exact float32 attention at all five attention sites, forward and gradients.

<div class="callout">

**Calibrate tolerances before you trust them.** VGGT once failed a 1e-5 loss limit with a 1.9e-4 drift. A CPU experiment showed that changing *only splash tile sizes* moves the step-1 loss by 4.2e-4. The limit was wrong, not the kernel. Measure how much harmless reorderings move your metric, then set the limit.

</div>

### Know the roofline of your chip

For TPU v6e (Trillium): 918 TFLOP/s bf16, about 1.64 TB/s HBM, so the ridge point is about 560 FLOP per byte. An op below that arithmetic intensity is memory-bound however good the kernel is. Inside the chip, the work runs on:

- the **MXU** (matrix unit): 256×256 systolic arrays doing the matmuls,
- the **VPU** (vector unit): elementwise math, exp, max, masks, on registers of 8 sublanes × 128 lanes,
- **VMEM**: the fast on-chip scratch memory that Pallas kernels tile into.

Arrays live in HBM in (8, 128) tiles. A last dimension of 64 is padded to 128; a slice of 48 lanes forces a relayout. Remember these two facts: they come back in almost every section below.

### The profiling toolkit

| Question | Tool |
|---|---|
| Where does time go, by module? | Perfetto/xprof trace, with ops mapped to model components through HLO source metadata. VGGT's script groups every op by component (backbone, frame, global, decoder, loss, optimizer, collectives), phase (forward, backward, recompute, update) and op class (matmul, attention kernel, layout copy, elementwise, collective). Raise `--traceback-frames` (JAX keeps 10 Python frames; remat wrappers eat 5). |
| Is an op compute- or memory-bound? | XLA's per-op FLOP and byte counts → achieved FLOP/s vs peak and GB/s vs HBM bandwidth. |
| What will a change cost before I have a TPU? | Deviceless compiles with `jax.experimental.topologies` ("v6e:2x4"), `compiled.memory_analysis()`, HLO `estimated_cycles` scaled per op kind by a measured profile. VGGT predicted every round-3 result within range this way except one. |
| Why is my Pallas kernel slow? | Static VLIW bundle dumps (`--xla_jf_dump_llo_text`): count bundles per loop trip and which unit is saturated. |
| Why did it recompile? | `JAX_EXPLAIN_CACHE_MISSES=1`. |
| Will it fit? | Deviceless AOT compile of every program; keep a few GiB under the limit (see §5 for why "just under" is bad). |

---

## 1. Launch: time to first step

On a large model, launch can take longer than the work. H3's first version spent about 18 minutes from request to MP4 for a 10-second video, a good part of it not in the denoising loop at all. VGGT-Omega's first launches took about 7 minutes before step 1.

### Compilation

- [ ] **Persistent compilation cache.** VGGT-Omega: launch to first step 430 → 42 s, identical losses.
- [ ] **On multi-host, every process writes its own cache.** JAX wrote it from process 0 only, so 15 of H3's 16 hosts recompiled every run: step 0 came 118 s after load, 26 s after the fix.
- [ ] **Cache small programs too**, not only those over 1 s to compile. H3's VAE-decode small ops had cost about 14 s of recompiles.
- [ ] **Compress the cache entry** if it is large (zstd: a 1 GiB entry in 2.4 s instead of 29.3 s) and **use relative source paths** so a fresh host can reuse a shared cache (VGGT: all programs loaded in 96 s).
- [ ] **Compile in parallel and in the background.** VGGT compiled its five programs with 5 threads in 553 s instead of 1,100 s. H3 builds the step programs in a thread while the text encoder runs: step 0 waited 14.1 → 2.3 s.
- [ ] **Bucket dynamic shapes.** Every new prompt length was a new program for H3 (~105 s to step 0). One compiled step per bucket of lengths, with the real key count passed as a runtime scalar into the kernel: 8-12 s.
- [ ] **Hunt hidden recompiles.** Importing Pallas modules *during* tracing changed cache keys, so step 1 recompiled (45 s → 8 s after moving the imports). Pallas cache keys can include the caller's Python frames, so raising traceback depth made every VGGT arm recompile (360-390 s each).
- [ ] **Keep host-side setup off NumPy-sized hot paths.** A dense padded² NumPy attention mask took 671 s and 1.67 GiB to build on CPU at VGGT's largest shape; a lazy mask took 0.57 s. Drawing H3's initial noise flat then reshaping cut its compile from 50 s to 0.7 s.

### Loading weights

- [ ] **Each host places only its own shards.** `jax.device_put` of a full array to a 16-host sharding cost about 1 s per array: over 20 minutes for H3's transformer. Placing local shards and assembling with `jax.make_array_from_single_device_arrays`: 7 s.
- [ ] **Read each array once per slice and gather over the chip interconnect (ICI)** instead of every host reading everything. Worker-0 disk reads 114.8 → 20.0 GB.
- [ ] **Know your filesystem.** On a gcsfuse bucket, `pread` read a 4.1 GB share in 3-4 s; `mmap` took over 6 minutes. Reading with 32 threads: 23 → 1.4 s.
- [ ] **Prefetch later stages into host memory** while earlier stages run (H3's VAE: load 28 → 3 s), and gather large weights one block at a time to keep the loader's peak memory down (29.6 → 14.6 GiB).

---

## 2. Input and the host loop

The goal here is simple: the TPU should never wait for Python.

- [ ] **Measure loader wait, don't assume it.** VGGT's GCS loader (16 threads, prefetch 2) read a step's data in 0.90 s p50 against a 2.1 s step, with about 1 ms wait. Gaps between steps in the trace were 16-37 ms (under 1%). Nothing to fix, and knowing that saved time.
- [ ] **Never put an eager op inside the step loop.** H3 had a stray `rows[:0]` slice running eagerly; it blocked host run-ahead (the host dispatching step N+1 while the chip runs step N). Removing it: 0.682 → 0.645 s per step on 64 chips, about 5% for one line.
- [ ] **Pass small per-step values as NumPy, slice inside the program**, instead of one-chip `jnp` arrays (small, but free: 0.539 → 0.537 s).
- [ ] **Pause Python's GC during latency-critical threads** (H3 startup: 13.6 → 12.5 s).
- [ ] **Make checkpointing asynchronous.** A blocking save took 56 s; async, 5-6.5 s of block, about 2.8% of total wall time back, bitwise-identical through resume.
- [ ] **Donate buffers.** `nnx.jit` returned a copy of the weights; `donate_argnums=0` dropped H3's peak from 19.2 to 10.5 GiB per chip, bit-identical. That headroom paid for later changes.

---

## 3. Layout across chips: mesh, sharding, collectives

Once one step is a single program, the next question is how it is split across chips and how much of the communication is hidden behind compute.

### Pick the split from the model's shape

- **Weights that don't fit on one chip** (H3: about 40 GB of transformer weights, 32 GB HBM per chip) → tensor parallelism over a "tp" axis.
- **Long sequences, batch 1** (H3: 74k tokens) → activations sequence-parallel between blocks, with Ulysses all-to-alls around attention (gather all tokens, scatter heads, and back). Multi-host added a context-parallel "sp" axis.
- **Model fits, batch > chips** (VGGT-Omega, 1.1B) → data parallel, with Adam moments sharded over chips (ZeRO-1: peak 29.1 → 26.7 GB at a cost of 0.2-1.8% step time). Data parallel over 16 chips kept 96% of single-chip throughput.

### Checklist

- [ ] **Map logical axes to physical chips.** `make_mesh(..., physical=True)` so each tp group is a physical row of 8 chips: 0.645 → 0.584 s on 64 chips. A mesh of `--tp 8 --sp 8` instead of the default: 0.941 → 0.911 s and peak 10.6 → 6.0 GiB.
- [ ] **Read the gathers in the HLO.** XLA gathered all 56 heads of K/V (1.2 GB per gather) when each host needed only its own heads. One sharding constraint: **1.36 → 1.01 s** per step, the single biggest multi-host win.
- [ ] **Pad per device, not per kernel block.** Pad the sequence to 8 rows per device instead of 128: 81,920 → 74,240 rows, 1.01 → 0.94 s. (On one host: pad to 128 rows per chip instead of whole 2048-row blocks, 6.92 → 6.86 s.)
- [ ] **Avoid all-reduces of big outputs.** Running H3's float32 video head on each chip's own rows, then all-gathering, instead of all-reducing 780 MB of rows: 0.4215 → 0.4129 s.
- [ ] **Replicate small, hot weights** if memory allows: two output projections kept whole on every chip (+10 GB each) removed their gathers, 0.590 → 0.550 s.
- [ ] **Shard weights over the axis whose links are idle.** Splitting weights over "sp" columns kept their gathers off the links the all-to-alls use: 0.4368 → 0.4215 s.

### Overlap communication with compute

- [ ] **Give Pallas calls a cost estimate** (`pl.pallas_call(..., cost_estimate=...)`). Without one, XLA's scheduler sees a 90 ms attention call as free and plans no communication under it. On H3 adding it was neutral by itself (6.70 → 6.70 s, bit-identical) but was the precondition for the collective flags below to help; on VGGT, kernels without it were why the overlap flags did nothing.
- [ ] **Pipeline by head group.** Run attention one head group at a time so head *h+1*'s K/V gather overlaps head *h*'s kernel: 0.4946 → 0.4368 s, H3's slice MFU 37% → 42%.
- [ ] **Try the async-collective and offload flags, then measure.** On H3: `--xla_tpu_enable_async_all_to_all` and `--xla_tpu_overlap_compute_collective_tc` (6.61 → 6.50 s), and SparseCore all-gather offload on 64 chips (0.899 → 0.687 s). Run each flag set in a fresh process; H3 tried 21 sets for the all-gathers alone.

<div class="callout callout-warn">

**Flags don't transfer between models.** The scoped-VMEM flag `--xla_tpu_scoped_vmem_limit_kib=98304` took H3's MLP matmuls from 71% to 90% of peak (6.40 → 6.24 s). The same flag made the VGGT run **4.0% slower** and the VGGT-Omega pre-training **16.9% slower**, because it starved other kernels. MaxText's published v6e flag set: also +4.0%. Collective-overlap flags: 0% on VGGT, where collectives were only 2.6% of the step. Measure every flag on your own program.

</div>

---

## 4. Inside the step, module by module

This is where most of the MFU comes from. Start from a breakdown of one step by op class. Two real ones:

| H3 inference, v6e-4, after 37 versions (6.35 s) | | VGGT training, v6e-8, before the kernel rewrite (2.16 s) | |
|---|---|---|---|
| attention kernel | 72.1% | attention kernel | 52% |
| MLP matmuls | 10.4% | matmul fusions | 22% |
| projections | 6.1% | elementwise | 8% |
| exposed communication | 5.6% | layout copies | 8% |
| idle | 0.0% | old XLA attention path | 6.5% |
| HBM bandwidth used | 18% | collectives | 2.6% |

Both are dominated by attention, so that is where to start.

### 4a. Attention

**Step 1: use a real kernel.** XLA's attention (the "CUDA-matched" one a port starts with) materializes scores. Switching VGGT-Omega's global cross-frame attention to the Pallas splash kernel: 0.774 → 0.565 s per step, and a shape that crashed the compiler started working.

**Step 2: tune the tiles at the real shape.** Splash defaults to 512-row blocks. H3's sweep found 2048/2048/1024: **15.35 → 10.04 s**, the largest single change in the whole project. VGGT replaced a fitted cost model with a measured sweep per attention site and pinned the tiles: −6.7%. Choose a query block that divides the rows exactly (H3: 2560 × 29) so you don't run a second small call for the remainder: attention calls per chip 100 → 50.

**Step 3: find out what bounds the kernel.** This is the step that decides everything after it, and the two projects got opposite answers.

The MXU is 256 wide. A score `q·k` with head dimension *D* uses *D* of those 256 rows.

- **H3, D = 128.** At most half of the MXU can be useful: putting two heads side by side would mix their products. (We checked: padding q and k with zeros to 256 made a call only 0.9% slower.) The kernel was limited by the **VPU**: the exp, max, and sum work around each score tile. So the wins were vector-side:
  - Softmax row sums computed on the MXU as `P @ [V, 1]`, filling the MXU's idle half: 8.74 → 7.88 s.
  - `exp2` instead of `exp` (fold `log2 e` into the scale): 7.88 → 7.80 s.
  - P·V in bf16 with f32 accumulation: 10.04 → 9.92 s.
  - "Lazy max": take the row maximum from the first key block and redo the row only if a later block overflows, instead of rescaling every block: 7.50 → 7.16 s. (Redo the overflow per head in a `while_loop`, not per call: otherwise one chip redoes a whole call while the others wait at the next all-to-all.)
  - Kernel time per call went from 236 ms to 92.8 ms, 422 TFLOP/s per chip, which is 92% of the 50% ceiling.

- **VGGT, D = 64.** The same vector tricks *added* bundles. Static bundle dumps showed why: the bound was **MXU result pops**. Each 256×256 weight tile returns its products 8 rows × 128 lanes at a time, 256 columns per row *regardless of how many are useful*. With D = 64, both `Q·Kᵀ` (contraction 64) and `P·V` (output width 64) pop as many registers as a 256-wide product would, capping the kernel near 14% of peak. The vector unit had slack.

  The fix was to **transpose the products** so the 64-wide side streams through the array and the score-sized side sits in the weight tile: `[64, 768] @ [768, 2048]` takes 1,017 bundles against 1,805 for `[2048, 768] @ [768, 64]`. A new transposed flash kernel (forward, dq, dkv, fused backward) that takes Qᵀ/Kᵀ/Vᵀ in a dense `[heads, 64, tokens]` layout, with no 64→128 lane padding, cut static cycles per micro-step by 39% and the measured step by **23-28%**.

  H3 later used the same idea in the other direction: a transposed kernel that keeps P in the MXU, so row sums cost 8 streamed rows instead of 128 columns (−14.4% per call on the slice).

<div class="callout">

**Head dimension sets your attention ceiling on TPU.** With 128-wide heads, exact attention can use at most 50% of the MXU; with 64-wide heads you are fighting pops. Know which regime you are in before you copy a kernel trick from another model. The 128-wide heads are also the main reason H3 on a v6e-4 runs about 1.6× slower than on 4×H200 for a 5-second video.

</div>

**Step 4: get padding out of the main loop.**

- [ ] Move the padding mask out of the main call: run the unpadded keys mask-free, handle the tail separately, merge with log-sum-exp (9.43 → 8.74 s). Later, folding the tail back into the main call without a mask: 6.86 → 6.70 s.
- [ ] Mask only partial key blocks (VGGT `skip_full_block_mask`: −12% of the forward kernel's bundles, bitwise).
- [ ] Keep the waste bounded at any length: whole 4096-key blocks plus a tail in 256-key windows. Before this, H3's speed depended on prompt length.
- [ ] Apply the softmax scale to the f32 scores *inside* the kernel, not to bf16 q outside, so q is rounded once (7.02 → 6.96 s traced, and error vs exact dropped 2.09e-3 → 1.97e-3).

**Step 5: don't do the same attention twice.** In training with remat, the attention forward runs again in the backward. A checkpoint policy that saves attention outputs over ≥4,096 keys (the long, expensive ones) cut VGGT's step 8.8%; saving all attention residuals once memory allowed: another 5.5%.

### 4b. Matmuls and MLP

Linear layers are the easy part to get to 80-90% of peak, if the compiler is allowed to.

- [ ] **Check achieved FLOP/s per matmul fusion.** H3's MLP was at 71% of peak until the scoped-VMEM limit was raised (90%). VGGT-Omega's linear layers were at about 19% of peak: an open item, and the next place to look.
- [ ] **Watch for fusions that recompute.** XLA fused H3's `SiLU(x)·gate` into the following matmul and recomputed it *per tile*. An `jax.lax.optimization_barrier` to materialize it once: 7.66 → 7.63 s. Same story with selecting AdaLN rows by `select`, which got fused and recomputed (v43, 6.31 s, rejected). Barriers can change bits; VGGT's best barrier (−1.6%) was not adopted because of that.
- [ ] **Separate what shouldn't be fused.** Keeping the FFN output projection apart from the gate/residual/norm chain: 7.00 → 6.92 s.
- [ ] **Fold adapters into the main matmul, padded to the MXU.** H3's Turbo LoRA as a separate matmul cost +7.1%. Folded as `[x | s·A·x] @ [W; B]`: +4.0%. With the LoRA rank padded 64 → 128: +2.7%.
- [ ] **Know when a matmul is memory-bound.** VGGT's decoder memory-side projections ran at 83% of HBM bandwidth with about 384 FLOP/byte, below v6e's ridge of ~560. No kernel will make that compute-bound; changing what is computed (fewer copies of the memory per target view) would.

### 4c. Layout glue: the (8, 128) tax

Copies and relayouts are pure HBM traffic. They were 8% of VGGT's step (about 280 ms) and appear in every port, because GPU code freely slices, concatenates and transposes narrow dimensions.

- [ ] **No narrow lane slices.** RoPE written as split-half-rotate-concat slices 48- or 32-lane pieces. Rewrite it as `x·cos + (x @ R)·sin` with a signed-permutation matrix *R*: H3 6.70 → 6.61 s; VGGT −7.1% (the matmul at HIGHEST precision).
- [ ] **Pad before the head split, not after.** Do norms on the lane-dense `[B, L, 768]` tensor, then reshape to heads. VGGT's "tokens" layout plus the first kernel tuning: −19.8%.
- [ ] **Use selects with padded tables instead of slice + concat** around special tokens (VGGT's 5-token prefix).
- [ ] **Replace gathers with one-hot matmuls** when the index set is small. `jnp.take` on H3's AdaLN tables forced layout copies; a one-hot matmul is bit-identical and 7.62 → 7.50 s.
- [ ] **Pick the layout the kernel wants once**, not per call. Splash's `seq_minor`/`kv_seq_minor` layouts at four VGGT sites were part of the −6.7% tile pin.

### 4d. Elementwise and activations

- [ ] **Check the activation's lowering.** VGGT-Omega's GELU used `erfc`; XLA expands it into polynomials on the VPU. `erf`-based GELU: −6.6%.
- [ ] **No accidental float32 copies.** Moving H3's q/k norm and RoPE after the all-to-all, hoping to fuse them into a copy, made XLA normalize a float32 copy instead (v47: 6.28 s vs 6.17, rejected).
- [ ] **Keep the old XLA attention path from un-fusing neighbors.** Moving VGGT's frame attention to splash alone gave 0%, because the old path had fused the norms and RoPE, which came back as narrow ops. On the transposed kernel, the same change gained 68 ms.

### 4e. Conditioning and per-step constants

Diffusion models repeat the same network many times with only the timestep changing. Anything that depends only on the timestep or the prompt doesn't belong in the step.

- [ ] **Precompute timestep tables.** H3's AdaLN projections are 13B of the 33B parameters. One compiled projection per block for *all* steps, then drop the weights: 7.63 → 7.62 s, and 13B parameters out of HBM.
- [ ] **Run prompt-only modules once per generation.** H3's token refiner had run every step; it cost 6.7% of a step.

### 4f. Output heads and loss

- [ ] **Run heads on local rows** (see §3: 780 MB all-reduce removed).
- [ ] **Account for float32 heads.** VGGT-Omega's DPT head runs in float32 at HIGHEST precision, which TPU executes as 6 bf16 passes. Make sure the profile separates it before deciding it is cheap.
- [ ] **Check the loss network.** VGGT's VGG19 perceptual loss was 2% of the step: fine to leave.

### 4g. Optimizer

- [ ] **Compare the update with its bandwidth floor.** AdamW reads and writes about 28 bytes per parameter. For 1.14B parameters at 1.64 TB/s that is 19.5 ms; VGGT-Omega's took 24 ms. Nothing to win there except sharding it (ZeRO-1) to free memory.

---

## 5. Memory: trade it for compute, carefully

Training is usually limited by what fits. VGGT without remat needed 65.6 GB on a 33.5 GB chip, so every saved activation is a choice.

- [ ] **Save the expensive activations, recompute the cheap ones.** Keep attention residuals (long attention first), recompute norms and MLPs.
- [ ] **Leave headroom below XLA's remat target.** One policy predicted −37 ms but put the program 5 MB under XLA's 31.25 GiB rematerialization target. XLA then rematerialized 361 instructions instead of 69, for a net −9.5 ms. "Just fits" is not the same as fits.
- [ ] **Be skeptical of host offload.** Offloading attention residuals to host memory made VGGT **11.1% slower**: XLA kept each residual until just before its use, then copied it to host and straight back, 5.9 GB each way, not overlapped.
- [ ] **Bigger per-chip batch isn't free throughput.** VGGT-Omega: 14.2 → 13.8 → 11.4 images/s as batch per chip grew. Once the chip is busy, more batch just adds memory pressure.

---

## 6. The whole job, not just the step

Users wait for a video, not a step. After the step was fast, H3's remaining time was everywhere else.

- [ ] **Batch the decoder.** H3's VAE decoded 28 tiles per clip one by one; as one batch over the chips, 57 → 18 s. On 64 chips, one decoder program spread over 16 hosts took 151 s, against 31 s on a single host's four chips, so each host now decodes on its own chips.
- [ ] **Prepare the next stage during the current one** (decode programs built during denoising).
- [ ] **Keep a resident server.** Parked models in pinned host memory go back on chip in 0.53 s, versus reloading. H3-Base: 81.5 s median per request on v6e-64.
- [ ] **For LLM serving, keep the chips fed.** Qwen3.8 captioning went 11.4 → 0.88 s per clip mostly through feeding: frames prepared in worker processes, 160 requests in flight (chips ~97% busy), inputs resized to a fixed set of aspect ratios so vision calls compile to about 140 fixed shapes, the vision tower's attention run as a batch of equal-length temporal segments instead of one masked product over the whole 40,320-patch clip (that one attention had cost 5.9 s per clip; its 72-wide heads are zero-padded to 128), and an elementwise token gather replaced by a jitted row gather (4.4 s per step). Rewrites on one v6e host: 2,283-2,381 tokens/s against 590 on two A6000 GPUs.

End result for H3 on a v6e-4: a 10-second video went from about 18 minutes (v0) to about 9 (v37) from raw request to MP4, and after round three a warm job takes 6.8 minutes. On v6e-64, warm launch to MP4 is about 2 minutes.

---

## 7. What didn't work

Negative results are half of the log. A selection, with the reason:

| Tried | Result | Why |
|---|---|---|
| bf16 exponentials in attention | 10.34-12.14 s vs 9.92 | converting scores to bf16 and back cost more than it saved |
| Overflow check once per step instead of per call | 14.45 s vs 6.86 | every step overflowed, so every step was redone |
| Ring of async permutes for weight gathers | −0.5% on 4 chips, 1.083 vs 0.941 s on 64 | XLA started the permutes only after each attention kernel; also raised audio error, so reverted |
| K/V ring gather across hosts | 1.485 vs 0.941 s | slower than XLA's gathers |
| Stretching an async collective across a Pallas kernel (barriers, scheduling groups) | 6.17 s, no overlap | XLA's latency-hiding scheduler would not, and no libtpu flag changed that |
| MaxDiffusion's attention kernel | 105.4 vs 92.7 ms per call | slower at H3's shapes; its full flag set ran out of HBM |
| MaxText / h3 flag presets on VGGT | +4.0% | starved other kernels of VMEM |
| VPU-side attention tricks on 64-wide heads | more bundles | the bound was MXU pops |
| Token-major custom decoder kernel | +2.5% | removed 9.85 ms of copies, added 24 ms of kernel |
| Host offload of residuals | +11.1% | copy out and back right before use, not overlapped |
| Multi-token prediction on vLLM-TPU (Qwen3.8) | broken outputs | a serving-stack bug; left off |

---

## 8. The checklist on one page

**Measure**
1. Fixed recorded workload at the production shape; analytic FLOPs; MFU.
2. Back-to-back A/B, skip warm-up, traced vs traced, noise floor known.
3. Accept rule and correctness class (bitwise / rounding / approximate) decided before the run.
4. Breakdown by module and op class; roofline per op; deviceless compiles for prediction.

**Launch**
5. Persistent, per-process, compressed compile cache; background and parallel compiles.
6. Shape buckets; no recompiles (`JAX_EXPLAIN_CACHE_MISSES`).
7. Each host places its own shards; read once, gather over ICI; `pread` on fuse mounts.

**Host**
8. No eager ops in the loop; NumPy step arguments; async checkpoints; donated buffers.

**Across chips**
9. Physical mesh; gather only what each chip needs; pad per device.
10. Cost estimates on Pallas calls; head-group pipelining; flags measured, never copied.

**Inside the step**
11. Attention: real kernel → tile sweep at real shape → find the bound (VPU vs MXU pops) → matching tricks → padding out of the loop.
12. Matmuls at 80-90% of peak; no recomputing fusions; adapters folded and padded to 128.
13. No narrow lane slices: RoPE as matmul, pad before head split, one-hot instead of gather.
14. Activation lowering checked; per-step constants precomputed; heads on local rows.

**Memory and job**
15. Save expensive activations, keep headroom under the remat target, distrust host offload.
16. Batch the decoder, overlap stages, keep the server resident, keep LLM chips fed.

The order matters less than the habit: one change, one measurement, one correctness check, one commit. Both projects got their biggest single wins (a tile sweep for H3, a transposed kernel for VGGT) only after the measurements made clear what the chip was actually waiting on.
