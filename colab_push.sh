#!/usr/bin/env bash
# colab-cu128 branch only — sets up a fresh Colab VM for a cjk_anima_scale run
# (project/cjk_anima_scale/colab.md § Setup). Run from the local
# checkout, after `git push origin colab-cu128`:
#
#   ./colab_push.sh <session> [<context run> ...]
#
# The session must exist (`colab new --gpu L4 -s <session>`). Steps: clone or
# reset the branch at the local absolute path, tar the assets over ssh, send
# $MANGA109S/derived/dialogue_2_10.tsv and write the VM's .env (the piece tiers
# and every single run's windows read it), create the wake_probe symlinks,
# write /content/env.sh, `uv sync`, fetch the Anima weights on the VM, and
# check torch.cuda + flash_attn. Each <context run> also sends that run's
# trained.pt + data/vocabs.json (a run whose file sets `context`, e.g.
# retrain_kana for retrain_kanji_b1). Every step is safe to re-run.
set -euo pipefail

SESSION=${1:?usage: colab_push.sh <session> [<context run> ...]}
shift
CONTEXT=("$@")
ROOT=/home/sorryhyun/anima/anima_lora # train.jsonl + scene records store absolute paths
BRANCH=colab-cu128
REPO=https://github.com/sorryhyun/anima_lora.git
cd "$ROOT"

SSHO=(-o "ProxyCommand=colab ssh --proxy-mode -s $SESSION"
    -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR)
# One ssh connection per runtime (a second gets HTTP 429): every step below is
# sequential.
vm() { ssh "${SSHO[@]}" root@colab "$@"; }

local_sha=$(git rev-parse "$BRANCH")
remote_sha=$(git ls-remote origin "refs/heads/$BRANCH" | cut -f1)
if [[ "$local_sha" != "$remote_sha" ]]; then
    echo "origin/$BRANCH (${remote_sha:-missing}) != local $BRANCH ($local_sha): push the branch first" >&2
    exit 1
fi
echo "== checkout $BRANCH @ ${local_sha:0:10}"
vm "set -e
if [ -d $ROOT/.git ]; then
  cd $ROOT && git fetch -q origin $BRANCH && git checkout -q -B $BRANCH origin/$BRANCH
else
  mkdir -p $(dirname $ROOT) && git clone -q -b $BRANCH $REPO $ROOT
fi
cd $ROOT && git log --oneline -1"

echo "== assets"
ASSETS=(
    models/vocab_packs/anima_cjk_vocab_pack.safetensors
    models/vocab_packs/anima_cjk_vocab_pack.json
    output/cjk_anima_scale/rows_step1_0921_merged/trained.pt
    project/cjk_anima_scale/assets/fonts
    output/cjk_anima_scale/scenes_s1
    output/cjk_anima_scale/scenes_s1w
    output/cjk_anima_scale/scenes_sl1w
    output/cjk_anima_scale/scenes_ja_comic
    post_image_dataset/render/ja/resized/boxes.jsonl
    post_image_dataset/render/ja/heldout/boxes.jsonl
)
for r in ${CONTEXT[@]+"${CONTEXT[@]}"}; do
    [[ "$r" == --* ]] && { echo "unknown flag $r (--pieces is gone: the phrase file always goes)" >&2; exit 1; }
    for f in trained.pt data/vocabs.json; do
        [[ -f "output/cjk_anima_scale/$r/$f" ]] || { echo "context run $r: no output/cjk_anima_scale/$r/$f" >&2; exit 1; }
        ASSETS+=("output/cjk_anima_scale/$r/$f")
    done
done
du -chL "${ASSETS[@]}" | tail -1
tar chf - "${ASSETS[@]}" | vm "cd $ROOT && tar xf -"

echo "== MANGA109S phrase file"
M=$(grep -E '^MANGA109S=' .env | cut -d= -f2-)
vm "mkdir -p /content/manga109s/derived && echo MANGA109S=/content/manga109s > $ROOT/.env"
scp "${SSHO[@]}" "$M/derived/dialogue_2_10.tsv" root@colab:/content/manga109s/derived/

echo "== symlinks, env.sh, uv sync, weights"
vm "set -e
cd $ROOT && mkdir -p output/wake_probe
for p in s1 s1w sl1w ja_comic; do ln -sfn ../cjk_anima_scale/scenes_\$p output/wake_probe/scenes_\$p; done
cat > /content/env.sh <<'EOF'
# ssh shells lack the kernel's LD_LIBRARY_PATH; without it torch finds no driver
export LD_LIBRARY_PATH=/usr/lib64-nvidia
export LIBRARY_PATH=/usr/local/cuda/lib64/stubs
export PATH=/usr/local/cuda/bin:\$PATH
export ANIMA_VOCAB_PACK=models/vocab_packs/anima_cjk_vocab_pack
cd $ROOT
EOF
. /content/env.sh
uv sync -q
.venv/bin/python tasks.py download-model anima
.venv/bin/python -c 'import torch, flash_attn; print(\"torch\", torch.__version__, \"cuda\", torch.cuda.is_available(), torch.cuda.get_device_name(0), \"flash_attn\", flash_attn.__version__)'"

echo "== ready: ssh in, '. /content/env.sh', then scale.py <run> data | train (colab.md § Running)"
