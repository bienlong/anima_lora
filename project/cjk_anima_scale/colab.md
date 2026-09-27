# colab.md — running a line run on a Colab VM

How to drive `scale.py <run> data | train` on Colab through the `colab` CLI
(uv tool, `google-colab-cli`), and bring the run back for eval here. Measured
on T4, L4, A100 and G4 smokes (2026-09-27; runs `t4smoke_a` (the L4 one),
`a100smoke_a` and `g4smoke_a`, all `chars:精俺感`, VM-only configs).

## Setup: one command

1. Rebase `colab-cu128` onto main and push it
   (`git push --force-with-lease origin colab-cu128`); the VM clones it
   from origin. The rebase needs a checkout of the branch:
   `git worktree add .claude/worktrees/colab-cu128 colab-cu128`, and
   `git worktree remove` it afterwards.
2. `colab new --gpu G4 -s <name>` (from the CLI, not the browser). G4 is
   the fastest at the same cost per step (§ Running).
3. From main's checkout, with no branch checkout needed:
   `bash <(git show colab-cu128:colab_push.sh) <name> [--pieces]`.
   Do not pipe it into `bash -s`, because the script's ssh calls would read
   the rest of the script from stdin.

`colab_push.sh` refuses unless origin's branch matches the local one, then
clones (or resets to origin) at the local absolute path, sends the assets
(≈ 1 GB, tar over ssh), creates the `wake_probe/` symlinks, writes
`/content/env.sh`, runs `uv sync`, fetches the DiT / TE / VAE with
`tasks.py download-model anima`, and checks torch.cuda + flash_attn.
`--pieces` also sends `$MANGA109S/derived/dialogue_2_10.tsv` to
`/content/manga109s/` and writes the VM's `.env`. From a fresh VM it took
3 min 17 s (L4, A100) and 4 min 47 s (G4). Every step is safe to re-run.

### The `colab-cu128` branch

One commit on top of main that changes only the install surface; it never
merges into main. The VM has torch `2.11.0+cu128` on driver 580, and main
locks cu132 / torch 2.12. The branch:

- takes torch 2.11.0 / torchvision 0.26.0 from `pytorch-cu128`, pinned also
  in `override-dependencies` (anime-tools 0.7.5 declares `torch>=2.12`), and
  `triton==3.6.0` (main's bare `triton` override lets 3.7.0 in);
- uses the `cu128torch2.11` flash-attn wheel (it also runs on sm120 here);
- makes `amd-rocm-100` explicit (otherwise `unsafe-best-match` installs
  `torch 2.11.0+rocm10.0.0`);
- restricts `environments` to linux x86_64, drops the `../anime_tools` dev
  path source (both groups take the tag), and drops sam3, pyside6, comfy-*,
  controlnet-aux and onnxruntime-gpu.

The line runs on torch 2.11 with no 2.12-only API: `data` + `train` with
compile passed on the branch venv here and on the L4. On a rebase that
conflicts in `pyproject.toml` / `uv.lock`, keep the branch's hunks (each is
marked `colab-cu128 branch`) and run `uv lock` again.

## Session and billing

- A session opened in the browser shows as `[?]` in `colab sessions` (no
  local token); `exec` / `ssh` cannot attach to it.
- `colab usage` shows the balance and rate. L4 costs 1.54 CU/h, A100
  5.30 CU/h, G4 8.90 CU/h.
- `colab stop -s <name>` when done. An idle VM keeps billing. Stopping loses
  the VM's disk, so pull what comes back first.
- `colab status` shows the kernel only. It reads `IDLE` while an ssh job
  trains.
- **The backend reclaims a runtime whose kernel is idle, even while a GPU
  job runs on it** (Run B, 2026-09-27: three G4s lost, 31.7 CU).
  `colab skill` says it: "Sessions stay alive as long as the kernel is
  active." A job started with ssh `nohup`, or with a `Popen` that returns
  from a `colab exec`, leaves the kernel `IDLE`. Such a VM lived as long as
  something contacted it at least every 10–14 min (ssh checks count). It
  died in the first gap of 22–25 min: `longb` (ssh only, last contact 19:13
  KST, gone by 19:38) and `longb2` / `longb45` (one `Popen` exec, then ssh
  checks; fine at 20:50, both gone by 21:12). Run A lived through its 66 min
  because its session ran a `colab exec` every ≈ 60 s the whole time
  (`~/.config/colab-cli/history/g4runA.jsonl`). The disk goes with the VM,
  and so does every partial not yet pulled. `session_terminated: pruned`
  in the history is when a local `colab` call noticed, not when the VM died.
  **Launch a run so the kernel stays busy** (§ Running: a blocking
  `colab exec` cell). Keeping it alive by contacting it is the other way,
  but that needs a client that never leaves a gap.

## Shell and transfer

- Use ssh / scp through the proxy (≈ 5.7 MB/s). `colab upload` runs at
  ≈ 0.25 MB/s. No `~/.ssh/config` edit is needed:

  ```bash
  SSHO=(-o "ProxyCommand=colab ssh --proxy-mode -s <name>" \
        -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR)
  ssh "${SSHO[@]}" root@colab '<cmd>'
  ```

- **One ssh connection per runtime.** A second one fails (HTTP 429, exit
  255) while the first is open, so an ssh that waits on a job blocks every
  other ssh. `colab exec` does not count against it. It takes the code on
  stdin, not as an argument:

  ```bash
  echo "import subprocess;print(subprocess.run('tail -c 700 /content/train.log',shell=True,capture_output=True,text=True).stdout)" \
    | colab exec -s <name>
  ```

- An ssh shell lacks the kernel's `LD_LIBRARY_PATH=/usr/lib64-nvidia`, and
  without it torch finds no driver. Start every ssh command with
  `. /content/env.sh` (it also sets `ANIMA_VOCAB_PACK` and cds into the
  checkout).
- `nohup bash -c ". /content/env.sh; .venv/bin/python …" > /content/<log> 2>&1 &`
  suits short jobs only (`data`, smokes). A job that outlives the kernel's
  idle window loses its VM (§ Session and billing).
- `pkill -f "<pattern>"` in an ssh command also matches that ssh's own shell
  (its command line contains the pattern) and kills it. Bracket one letter:
  `pkill -f "[n]vidia-smi"`.
- In `ssh … 'tar cf - …' > x.tar`, nothing else in that command may write to
  stdout, or the text lands in front of the archive.

## The VM

- **Standard shapes**, all with a 236 GB disk, driver 580.82, CUDA 12.8,
  system Python 3.13.15, user `root`:

  | GPU | VRAM | vCPU | RAM |
  |---|---|---|---|
  | L4 | 23 GB | 12 | 52 GB |
  | A100 (SXM4, sm80) | 40 GB | 12 | 83 GB |
  | G4 (RTX PRO 6000 Blackwell Server, sm120) | 96 GB | 48 | 176 GB |

- The branch's flash-attn wheel runs on all three.
- **T4 does not work:** flash-attn 2 has no sm75 build, and the line fixes
  `attn_mode="flash"` (`src/common/models.py`). T4 also has no native bf16.

## What goes over

`colab_push.sh` sends all of this. The checkout sits at the same absolute
path as here (`/home/sorryhyun/anima/anima_lora`), because `train.jsonl` and
the scene records store absolute paths.

| what | size | notes |
|---|---|---|
| DiT, TE, VAE (`anima` catalog pack) | 5.6 GB | fetched on the VM from the Hub |
| raw pack `models/vocab_packs/anima_cjk_vocab_pack.{safetensors,json}` | 274 MB | not on the Hub. Load-log sha `7b9fce0bb57b…` (a pack digest, not the file's sha256) |
| seed rows `output/cjk_anima_scale/rows_step1_0921_merged/trained.pt` | 8.9 MB | all `train` reads of that dir |
| `project/cjk_anima_scale/assets/fonts/` | 80 MB | gitignored |
| `output/cjk_anima_scale/scenes_{s1,s1w,sl1w,ja_comic}/` | 630 MB | all four for every run: `config.py`'s pool list loads them regardless of kind |
| `output/wake_probe/scenes_<p>` → `../cjk_anima_scale/scenes_<p>` | symlinks | the scene records' `file` paths go through `wake_probe/` |
| `post_image_dataset/render/ja/{resized,heldout}/boxes.jsonl` | 1.9 MB | the only corpus files `data` reads |
| `$MANGA109S/derived/dialogue_2_10.tsv` + `.env` | 1.9 MB | `--pieces` only |

## Running

- No daemon on the VM. Run the verbs directly:
  `.venv/bin/python project/cjk_anima_scale/scale.py <run> data`, then
  `… <run> train`. The default `--workers` (cpu − 2) fits every shape.
- Smokes: 270 steps, batch 4. "Steady" is the rate between logged steps
  after compile (`train_log.json`'s `it_s` is the running mean from step 0).
  The `train` time covers load, caches, compile, steps and save.

  | GPU | steady it/s | CU per 1 k steps | `data` / `train` | peak VRAM | peak host RAM |
  |---|---|---|---|---|---|
  | L4 | 1.29 | 0.33 | 0.4 min (`--workers 2`) / 308 s | 17.2 GB | 5.6 GB |
  | A100 | 4.12 | 0.36 | 23 s / 156 s | 18.6 GB | 6.5 GB |
  | G4 | 7.18 | 0.34 | 15 s / 76 s | 17.8 GB | 7.1 GB |

  The local GPU runs ≈ 2.2 it/s. Cost per step is the same on all three, so
  the GPU choice sets wall time only. At batch 4, VRAM stays under 19 GB, and
  the G4's 96 GB goes unused.
- Piece smoke (`--pieces`, L4, 2026-09-27; VM-only run `psmoke_p3`: すごい
  わかる なんだ, 3 glyphs, 90 steps/row): `data` 35 s (180 items, both piece
  tiers, the phrase file read through the VM's `.env`), `train` 270 steps in
  337 s, steady ≈ 1.15 it/s, 14 GB VRAM. The returned rows loaded and
  rendered here.
- **Two runs on one G4 are slower than one** (Run B, 2026-09-27:
  `run0927_long_p3` + `run0927_long_p45` launched together, 32 GB VRAM):
  3.20 + 3.21 = 6.41 it/s steady, against 7.18 for one run. Two processes
  time-slice the GPU (no MPS), and one run at batch 4 already keeps it busy
  (caches built, compiled). Run one at a time. Two runs together only save
  the idle gap if nobody is there to launch the second one.
- Data build at scale (G4, 46 workers): `run0927_long_p3` 23 534 items in
  2.1 min, `run0927_long_p45` 33 600 in 2.9 min. `data/` holds ≈ 1 MB of
  TE cache per item (30 GB / 39 GB for the two runs); the 236 GB disk has
  room.
- Wall time and cost:

  | run | steps | L4 | A100 | G4 |
  |---|---|---|---|---|
  | A | 27.5 k | 5.9 h / 9 CU | 1.9 h / 10 CU | 1.1 h / 9 CU |
  | C-k (150 steps/row) | 58.2 k | 12.5 h / 19 CU | 3.9 h / 21 CU | 2.3 h / 20 CU |
  | C-k (270 steps/row) | ≈ 105 k | 22.6 h / 35 CU | 7.1 h / 38 CU | 4.1 h / 36 CU |
  | C-p | 73.1 k | 15.7 h / 24 CU | 4.9 h / 26 CU | 2.8 h / 25 CU |
- The VM has no `/usr/bin/time`.
- **Launch `train` as a blocking kernel cell**, so the kernel reads `BUSY`
  for the whole run (§ Session and billing). Send this through
  `colab exec`, and let the local client exit (or `timeout` it). Jupyter
  keeps running the cell after the websocket closes. The kernel's env
  already has the driver path.

  ```python
  import subprocess
  subprocess.run(['bash', '-c', cmd], stdout=open(log, 'w'), stderr=subprocess.STDOUT)
  ```

  Untested as of 2026-09-27. Check it on the first long run: early on, leave
  the session untouched for more than 30 min and confirm it survives. A
  `Popen(..., start_new_session=True)` returns at once, and the kernel goes
  `IDLE`, so it is not a launch for a long job.

## What comes back

Pull `<run>/{trained.pt,train_log.json,train_record.json}` and
`<run>/data/{vocabs.json,build.json,eval.json}` into the same paths here:

```bash
ssh "${SSHO[@]}" root@colab 'cd /home/sorryhyun/anima/anima_lora && tar cf - \
  output/cjk_anima_scale/<run>/{trained.pt,train_log.json,train_record.json} \
  output/cjk_anima_scale/<run>/data/{vocabs.json,build.json,eval.json}' | tar xf -
```

Then check `md5sum trained.pt` on both sides. Eval runs here through the
daemon (`scale.py <run> eval --submit`), against the floor cache in
`rows_step1_0921_merged/`; a VM-side eval would render a second floor. On
the smoke, the returned rows loaded as a merged run and read single 6/6, EN
24/24. A full eval renders every ruler (the smoke's was stopped after
7.5 min); to check that a run trained, `report.md` and `sheet_single.png`
are enough.

## Open before a real run

- **No resume; partials instead** (user, 2026-09-27). `train` writes
  `trained_partial.pt` every 5 000 steps (`SAVE_EVERY`), in the same merged
  format as `trained.pt`, so eval reads it as is. It lives on the VM's disk,
  which a dropped session loses, so pull it during a long run. The drops so
  far are idle-kernel reclaims (§ Session and billing), not a session cap.
  The C runs take 2.3–4.1 h on a G4 (12–23 h on an L4).
- **Which runs go.** A ran on a G4. B lost its VMs (2026-09-27). What came
  back is `run0927_long_p3/trained_partial.pt` at 20 k of 31 770 steps and
  `run0927_long_p45/trained_partial.pt` at 15 k of 45 360: cosine schedule
  cut mid-run, no resume, so neither is the recipe's result. C goes on
  Colab as well.
- **C-k budget** (150 or 270 steps/row) is open in plan_2900.
