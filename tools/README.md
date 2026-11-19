# tools/

| File | What |
|---|---|
| `export_flipped.py` | Fusion 360 script: exports a body as 3MF, then writes a copy mirrored top-to-bottom (vertices `(x, y, z) → (x, −y, zmax − z)`) so a plate prints face-down without supports, and opens it in Bambu Studio |
| `clean_photos.py` | copies photos into `docs/images/` with every EXIF / GPS / maker-note block stripped, resized, and re-checks them (`--check`) |
| `hooks/pre-commit` | leak guard, enabled with `git config core.hooksPath tools/hooks`. Refuses data files (audio, video, embeddings, models, logs, datasets), secrets (key patterns + `gitleaks`), real household names (read from a git-ignored file, never written in the hook) and files > 5 MB |
