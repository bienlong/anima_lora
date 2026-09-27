# colab.md — running a line run on a Colab VM

How to drive a `scale.py <run> data | train` on Colab through the `colab` CLI
(uv tool, `google-colab-cli`). Everything below was measured on a T4 smoke
(2026-09-27, run `t4smoke_a` = `chars:精俺感`, VM-only config). What is still
open before a real run is in `colab_plan.md`.

## Session

- Create the session **from the CLI**: `colab new --gpu L4 -s <name>`. A
  session opened in the browser shows as `[?]` in `colab sessions` (an orphan
  with no local token); `exec` / `ssh` cannot attach to it.
- `colab sessions` lists what the server holds; `colab usage` shows the
  compute-unit balance and rate.
- `colab stop -s <name>` when done. An idle VM keeps billing.

## Shell and transfer

- `colab upload` runs at ≈ 0.25 MB/s (8.9 MB in 38 s). Use ssh / scp through
  the proxy instead, which ran at ≈ 5.7 MB/s (285 MB in 50 s). No
  `~/.ssh/config` edit is needed:

  ```bash
  SSHO=(-o "ProxyCommand=colab ssh --proxy-mode -s <name>" \
        -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR)
  ssh "${SSHO[@]}" root@colab '<cmd>'
  scp "${SSHO[@]}" <local> root@colab:<remote>
  tar cf - <dirs> | ssh "${SSHO[@]}" root@colab 'cd <root> && tar xf -'
  ```

- **One ssh connection per runtime.** A second one gets HTTP 429 while the
  first is open, so never leave an ssh that sleeps. `colab exec -s <name>`
  (Python in the kernel) does not count against it, so use it to read logs
  while an ssh is busy.
- The kernel sets `LD_LIBRARY_PATH=/usr/lib64-nvidia`, but an ssh shell does
  not, and without it torch reports "Found no NVIDIA driver". Source an env
  file in every ssh command:

  ```bash
  # /content/env.sh
  export LD_LIBRARY_PATH=/usr/lib64-nvidia
  export LIBRARY_PATH=/usr/local/cuda/lib64/stubs
  export PATH=/usr/local/cuda/bin:$PATH
  export ANIMA_VOCAB_PACK=models/vocab_packs/anima_cjk_vocab_pack
  cd /home/sorryhyun/anima/anima_lora
  ```

- Long jobs run detached: `nohup bash -c ". /content/env.sh; .venv/bin/python …" > /content/<log> 2>&1 &`.
  Read the log with `colab exec` or a short ssh.

## The VM (T4, standard shape)

Tesla T4 (sm75, 15 GB), driver 580.82, CUDA 12.8. System Python 3.13.15
already has `torch 2.11.0+cu128`, `torchvision 0.26.0+cu128`, `triton 3.6.0`
and `uv 0.12.9`. 12 GB RAM, 2 vCPU, 113 GB disk (≈ 66 GB free), user `root`.
T4 has no native bf16 (`is_bf16_supported(including_emulation=False)` is
False), and flash-attn 2 does not support sm75.

## Checkout

Put the repo at **the same absolute path** as here
(`/home/sorryhyun/anima/anima_lora`). `train.jsonl` and the scene records
store absolute paths, so a different root breaks both. Clone origin, then
bring unpushed commits over as a bundle:

```bash
git bundle create tail.bundle origin/main..main            # here
scp "${SSHO[@]}" tail.bundle root@colab:/content/
ssh "${SSHO[@]}" root@colab 'mkdir -p /home/sorryhyun/anima && cd /home/sorryhyun/anima \
  && git clone -q https://github.com/sorryhyun/anima_lora.git \
  && cd anima_lora && git pull -q /content/tail.bundle main'
```

## Install (main, no branch — what the smoke used)

Reuse the VM's torch through a venv that sees system site-packages, and
install the rest with `--no-sources` plus overrides:

```bash
uv venv --system-site-packages --python /usr/bin/python3 .venv
# /content/overrides.txt
#   torch==2.11.0
#   torchvision==0.26.0
#   flash-attn ; sys_platform == 'never'       (likewise sam3, pyside6, comfy-cli,
#   ...                                          comfy-aimdo, comfy-kitchen,
#                                                controlnet-aux, onnxruntime-gpu)
#   diffusers @ git+https://github.com/huggingface/diffusers@80c7ed262aeffbeb43ef13ae04baeb9b84515a69
uv pip install --python .venv/bin/python --no-sources --prerelease allow \
  --override /content/overrides.txt -e . \
  "anime-tools @ git+https://github.com/sorryhyun/anime_tools@v0.7.5"
```

**Trap:** this still installs `torch 2.11.0+rocm10.0.0`. The pyproject's
`amd-rocm-100` index is not explicit, and `index-strategy = "unsafe-best-match"`
lets it win. Fix after install:
`uv pip uninstall --python .venv/bin/python torch torchvision triton rocm rocm-bootstrap rocm-sdk-core rocm-sdk-libraries`.
The venv then falls back to the system cu128 build. A Colab branch removes the
need for this (`colab_plan.md` § 1).

## What goes over

| what | size | how |
|---|---|---|
| DiT `anima-base-v1.0`, TE `qwen_3_06b_base`, VAE `qwen_image_vae` | 4.2 GB + 1.2 GB + 254 MB | on the VM: `hf_hub_download("circlestone-labs/Anima", "split_files/<kind>/<file>")` → `models/<kind>/` |
| raw pack `models/vocab_packs/anima_cjk_vocab_pack.{safetensors,json}` | 274 MB | scp (not on the Hub; the `-cjk` repo holds only preview / preview2). Its load-log sha is `7b9fce0bb57b…` (a pack digest, not the file's sha256) |
| seed rows `output/cjk_anima_scale/rows_step1_0921_merged/trained.pt` | 8.9 MB | scp |
| `project/cjk_anima_scale/assets/fonts/` | 80 MB | tar over ssh |
| `output/cjk_anima_scale/scenes_{s1,s1w,sl1w,ja_comic}/` | 630 MB | tar over ssh. **All four for every run**: `config.py`'s pool list loads them regardless of kind |
| `output/wake_probe/scenes_<p>` → `../cjk_anima_scale/scenes_<p>` | symlinks | create on the VM; the scene records' `file` paths go through `wake_probe/` |
| `post_image_dataset/render/ja/{resized,heldout}/boxes.jsonl` | 1.6 MB + 0.3 MB | scp (the only corpus files `data` reads; no images) |
| `$MANGA109S/derived/dialogue_2_10.tsv` + `.env` | — | piece runs only; untested on the VM |

## Running

- No daemon on the VM. Run the verbs directly:
  `.venv/bin/python project/cjk_anima_scale/scale.py <run> data --workers 2`,
  then `… <run> train`.
- `data` on the T4 VM (2 vCPU): 200 items (single, `b0709`) in 0.4 min, and
  the pack loads with the right sha.
- `train` loads the DiT, builds the latent / TE caches and reaches compile,
  then fails at the first attention call (`TypeError: 'NoneType' object is
  not callable` in `attention_dispatch.flash_attn_func`).
  `src/common/models.py` fixes `attn_mode="flash"`, and the VM had no
  flash-attn. On L4 the cu128 flash-attn wheel should cover it; on T4 it
  would need SDPA. Throughput is still unmeasured on any Colab GPU (local
  A0: ≈ 2.2 it/s on the 5070 Ti).
- `train` cannot resume. `trained_partial.pt` is written every 5 000 steps,
  but a dropped VM loses the run.

## What comes back

`<run>/trained.pt`, `train_record.json`, `train_log.json`, and
`<run>/data/{vocabs.json,build.json,eval.json}`. Eval runs here through the
daemon, against the floor cache in `rows_step1_0921_merged/`.
