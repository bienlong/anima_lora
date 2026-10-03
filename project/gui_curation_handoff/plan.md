# GUI curation handoff → anime_tools panel

The trainer GUI stops curating. Curation (captions, tagging, masks, grouping,
exclusion) moves to the `anime_tools` web panel (`anime-tools-gui`). The trainer
GUI keeps cache building and training.

## Phases

1. **Cache-only Preprocess tab** — done, `11fdd794`. Autotag / position / SAM
   sections are gone, and the GUI env pins `CAPTION_AUTOTAG` /
   `CAPTION_POSITION_CLAUSES` to `0`.
2. **"Open anime_tools" button** — done. Pin at `v0.7.6`; the Preprocess
   tab's status row launches the panel (`gui.anime_tools_panel`), seeds Export
   to `sidecars_only`, and flags a workspace edit newer than the last applied
   Export.
3. **Retire the Dataset tab** — done, together with phase 2 (so there was
   never a window with two caption writers). The four modules, their ~110
   strings per language and the five Dataset-only Settings rows are gone. The
   resize preview is a dialog off the Preprocess tab
   (`gui.tabs.preprocess.resize_preview`).
4. **(Optional) Embed** — a `QWebEngineView` tab behind `LazyTabHolder`, only if
   switching windows turns out to be a real annoyance. It reuses phase 2's
   launcher.

## Data flow

Images never leave `image_dataset/`. The panel curates in its own workspace and
exports only the decisions; the trainer resizes from the originals itself.

```
image_dataset/                         originals, read-only for the panel
  ├─ panel: workspace/                 resized proxy, captions, masks, ledger
  │    └─ Export --sidecars_only  ──→  post_image_dataset/resized/{rel}.txt (+ .variants.txt)
  │                               ──→  post_image_dataset/masks/…  (original size)
  │                               ──→  image_dataset/{rel}.txt     (revised master)
  │    workspace/_excluded/excluded.json  ── read by the trainer as a skip list
  └─ trainer resize (image_dataset → post_image_dataset/resized/{rel}.png, skips excluded)
       └─ VAE / TE / PE caches
```

- `--sidecars_only` (anime_tools **v0.7.6**) publishes no `image` rows and no
  `_excluded/` mirror. Each mask is fitted to its original, uncapped. The
  request refuses `--resize_cap` / `--webp` with it.
- The caption row falls back revised → workspace master → `image_dataset/`
  master, so every image in the workspace gets a `resized/{rel}.txt`. That
  closes the position-clause trap: TE reads what the panel wrote.
- Exclusion already works. `curation_actions.py` unions `workspace/_excluded`
  (`PACKAGE_WORKSPACE_EXCLUDED_DIR`) into `ResizeRequest.skip`.
- `workspace/resized` stays as the panel's working copy. Masking, grouping and
  OCR need one geometry, so it is not removable without reworking the stages.

## Phase 2 sketch

- **Bump the pin** to `v0.7.6` (`pyproject.toml`, `tag = "v0.7.5"`) and
  `uv sync`.
- Launch `python -m anime_tools.gui --home <ANIMA_HOME> --open` as a detached
  process. It is not a daemon job: there is no GPU work, and the panel's
  `--exit-with-window` reaps the server.
- **Roots: the package defaults**, with one exception. `dst` / `masks` /
  `master` stay under `<home>/workspace/`, and `out` stays `post_image_dataset`
  (Export appends `resized/`, `masks/`, `captions/`). Seed only `src` in
  `<home>/.anime_tools_gui.json` → `"dataset"`, and only when the Preprocess
  tab's `source_image_dir` is not `image_dataset`. Do **not** point `dst` /
  `masks` at the trainer trees, and do not set `ANIME_TOOLS_WORKSPACE`.
- **Export must run with `sidecars_only`.** Default Export publishes originals
  into `post_image_dataset/resized/`, where they collide with the trainer's
  `{stem}.png` (the trainer refuses two images with one stem). Options: seed it
  as the saved form default, or document it. Find out whether the settings file
  can carry a stage default.
- **Order**: Export → `make preprocess-resize` → `make preprocess-te` (and the
  other caches). The Preprocess tab could warn when the newest file under
  `workspace/` is newer than `workspace/captions/export/report.json`.
- Reuse a running server: the port is picked from 8790 upward, so a second
  click should focus the open server, not start a new one.

## Open questions

- **Mask crop**: masks publish at the original's size. `_load_mask`
  (`library/datasets/cache.py`) interpolates to latent size but ignores the
  resize crop (anchor / margins), so a cropped image's mask is off by the crop.
  Either the trainer resize crops masks alongside images, or `_load_mask` reads
  the `anima_resize_*` keys off the resized PNG and crops first.
- **Excluded after resize**: an image excluded after the trainer already
  resized it keeps its `resized/*.png` and caches. Check whether
  `make preprocess-reconcile` drops skipped images, or add that.
- **Coverage**: Export walks `workspace/resized`, so an image added to
  `image_dataset/` after the panel last resized gets no caption from Export.
  The panel runs resize as a preflight. Check that the Export button triggers
  it too, or warn on a count mismatch.
- ~~**Two writers during phase 2**~~ — moot: phases 2 and 3 shipped together.
- **Migration**: existing curation lives in `post_image_dataset/resized`.
  `anime_tools.workspace.migrate --apply` *moves* it into `workspace/`, which
  empties the trainer tree until the next resize + Export. Document the
  sequence: migrate → Export (sidecars only) → resize.
- **Pin lag**: the panel runs at the pinned tag, so any further panel change
  this needs means cutting a tag first.
- **Task-side leftovers**: the `scripts/tasks/` readers are pruned
  (`preprocess.py` autotag form, `masking.py` GUI rule cards). Still left:
  `stage_form.py`'s `autotag` / `masks_sam` entries (`TRAINER_FIELDS`,
  `SHOWN_BOUND`, `FIELD_ORDER`, `_PP_SEEDS`, the card branches of
  `load_stage_values` / `merge_stages_into_meta`) and the `test_gui_stage_form`
  cases that render those schemas. They keep an old variant's
  `[variant.stages.masks_sam]` rows round-tripping on Save; drop them with
  those rows.
- **`curation_decisions.json`**: nothing writes it any more (the Dataset
  tab's skip / move marks), but `_curation_skips` still honours an existing
  file. Drop the reader once no checkout has one.
- **Pre-existing test failures** (not from this work):
  `test_curation_boundary` flags `scripts/tasks/masking.py` importing
  `library.config.sam_masks`, and `test_doc_refs` flags stale paths in
  unrelated docs.
