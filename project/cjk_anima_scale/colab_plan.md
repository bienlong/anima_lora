# colab_plan.md — what to do here before a Colab run (2026-09-27)

After the T4 smoke (`colab.md`), the next Colab attempt is a paid L4 smoke,
then the run. This is the local work that comes before it.

## 1. A Colab branch (`colab-cu128`)

The branch changes only the install surface and never merges into main.
The goal is a plain `uv sync` on the VM, replacing `colab.md`'s
override-and-uninstall procedure.

- torch / torchvision → the `pytorch-cu128` index, pinned to
  `torch==2.11.0` / `torchvision==0.26.0` (the builds preinstalled on the
  VM).
- flash-attn (linux x86_64) →
  `https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.9.4/flash_attn-2.8.3+cu128torch2.11-cp313-cp313-linux_x86_64.whl`.
  It works on L4 (sm89) but not on T4.
- Make `amd-rocm-100` explicit or drop it. Otherwise `unsafe-best-match`
  pulls `torch 2.11.0+rocm10.0.0`, as it did on the T4 smoke.
- Consider dropping deps the line never imports, to cut install time: sam3,
  pyside6, comfy-*, controlnet-aux, onnxruntime-gpu.
- `uv lock` on the branch; `requires-python` stays 3.13 (the VM has 3.13.15).
- The branch sits below main's `torch>=2.12` floor, so check what the line
  runs for 2.12-only API, starting with the compile / dynamic-shape path
  (`compile_blocks_for_training`).
- Rebase the branch onto main before each Colab launch; the VM clones it
  from origin.

## 2. Asset push script (on the branch)

`colab.md` § What goes over is eight steps done by hand. Put them in one
script on the branch that takes the session name, so a fresh VM is one
command:

- scp / tar the assets;
- create the `wake_probe/` symlinks;
- write `/content/env.sh`;
- fetch the three Anima weights from the Hub on the VM.

It stays on the branch, not on main or in the line (plan_2900 § 3a: "no
Colab-side code in the line").

## 3. Fix plan_2900 § 3a

It is wrong in four places, and should point to `colab.md` instead of
repeating it:

- the scene pools: all four (`s1,s1w,sl1w,ja_comic`) for every run, not
  "s1 / s1w for C-k";
- the corpus files `post_image_dataset/render/ja/{resized,heldout}/boxes.jsonl`
  are missing from its list;
- the `output/wake_probe/scenes_*` symlinks are missing;
- `LD_LIBRARY_PATH` in ssh shells, and `colab upload`'s speed (use scp).

## 4. The L4 smoke checklist

Once paid, create the session with `colab new --gpu L4 -s <name>`, then:

1. On the branch: `uv sync`, then
   `torch.cuda.is_available()` and `import flash_attn`.
2. Run `scale.py t4smoke_a data` (VM-only toml:
   `vocabs = ["chars:精俺感"]`, `read = []`) and check the pack sha
   `7b9fce0bb57b…` in the log.
3. Run `scale.py t4smoke_a train` (270 steps, with compile). Record:
   - it/s at steady state against the local ≈ 2.2 (A0 `train_log.json`);
   - compile time;
   - peak VRAM (L4 24 GB);
   - host RAM (the T4 standard shape had 12 GB).
4. Pull `trained.pt` and `data/{vocabs,build,eval}.json` back, then run
   `eval` here through the daemon to confirm the returned run evaluates
   against the local floor.
5. From the measured it/s, set each run's wall time against the session
   limit (§ 5).

## 5. Before the real run

- **Resume.** `train` writes `trained_partial.pt` every 5 000 steps but
  cannot resume from it. A (27.5 k steps) is ≈ 3.3 h here, and C-k
  (58.2 k at 150) ≈ 7 h. If L4 runs slower than local, or the session limit
  is shorter than the run, either split the run or add resume (rows +
  optimizer + step + rng). Resume is a line code change on main, with a
  test.
- **Data build.** At 2 vCPU, `data` for A (≈ 20 k items) is roughly
  30–40 min by the smoke's rate. The alternative is to build here and
  upload (≈ 4 GB at 5.7 MB/s ≈ 12 min). Both work, since the checkout sits
  at the same path.
- **Which run goes.** plan_2900 puts C on Colab and A local; this session
  started from A on Colab. Settle which runs go over before the push.
- **C-k budget** (150 vs 270) is still open in plan_2900, and it has to be
  settled before C-k's launch.
- Piece runs (B, C-p) also need `MANGA109S` on the VM (the `.env` path +
  `dialogue_2_10.tsv`), which the smoke did not cover.
