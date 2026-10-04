#!/usr/bin/env bash
# First start: fetch the first-run model set into the /app/models volume.
# Already-installed rows are skipped, so this is a no-op once the volume is filled.
# ANIMA_SKIP_DOWNLOAD=1 skips it (e.g. a host models dir mounted read-only).
set -e

if [ -z "${ANIMA_SKIP_DOWNLOAD:-}" ] \
   && [ ! -f /app/models/diffusion_models/anima-base-v1.0.safetensors ]; then
    echo "==> models not found in /app/models — running download-models"
    python tasks.py download-models
fi

exec "$@"
