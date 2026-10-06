# 🎬 SparseChron — 4D Viewer Demo Runbook

## Before the slot (30–45 min early)

1. Open **notebooks/kaggle_demo.ipynb** on Kaggle (File → Import Notebook, once)
2. Session settings: **Internet ON**, **Accelerator = GPU T4**
3. **Nothing to attach** — checkpoints download automatically from the GitHub release
4. Run cells in order:

| Cell | What | Time |
|---|---|---|
| 1 | Clone repo + pip install | 10–15 min, **silent is normal** (gsplat compile) |
| 2 | Download D-NeRF mutant data | 2–5 min |
| 3 | Download checkpoint (GitHub release) | ~1 min — prints iteration + Gaussians |
| 4 | Eval | ~5 min — prints PSNR/SSIM/LPIPS |
| 3b (optional) | Switch to August model (dramatic motion) | ~2 min (237 MB) |
| last | **Viewer + tunnel** | 30 s |

After cell 3b, re-run the LAST cell — the viewer automatically uses the August model.

## Cell order summary

- **Default demo:** 1 → 2 → 3 → 4 → last (retrained model, metrics story)
- **Dramatic motion:** after the above, run 3b → last again (August model, wow story)

## Last cell output — how to read it

- **Top line** (`curl ifconfig.me`): the tunnel **password** (an IP address)
- **Bottom**: the `https://xxxx.loca.lt` link → open it, paste the password
- Wait ~20–30 s → drag to orbit, **Resolution slider → 1024** for sharpness, scrub **TIME slider**

## If something misbehaves

| Symptom | Fix |
|---|---|
| Blurry | Resolution slider → 1024 (4D Controls panel) |
| Tunnel URL dead / white page | Re-run the last cell — fresh URL every launch; refresh + wait 30 s |
| "port in use" | `!pkill -f interactive_viewer; !pkill -f localtunnel` then re-run viewer cell |
| gsplat "No CUDA toolkit" | Session options → Accelerator = **GPU T4**, then re-run from cell 1 |
| Checkpoint download fails | Re-run cell 3 — it's just a GitHub download |
| Everything fails | **Fallback: play the videos** — zero risk |

## Fallback files (on this laptop, in `outputs/`)

- `mutant_4d_playback.mp4` — 30 s loop, sharp August model, real 4D motion
- `mutant_video_retrained.mp4` — turntable of the retrained model
- `mutant_aug_light.ply` — SuperSplat interactive (superspl.at/editor), proven on this laptop
- `metrics.json` — **PSNR 29.17 / SSIM 0.945 / LPIPS 0.083**

## Numbers to quote

- Two trained 4D models, both on D-NeRF *mutant*, 30,000 iterations, free Kaggle T4
- Retrained model: **PSNR 29.17 dB, SSIM 0.945, LPIPS 0.083** (held-out views)
- August model: 347,437 Gaussians, dramatic motion; retrained: 1,013 Gaussians (densification = future work)
- Checkpoints + scene_transform.json hosted on GitHub Releases — fully reproducible pipeline

## Where things live

- GitHub repo: `github.com/rajhodedara/SparseChron` (notebook + code, always current)
- GitHub release: `releases/tag/trained-checkpoints` (both checkpoints + scene_transform.json)
- Kaggle datasets (legacy, no longer needed): `checkpoint`, `checkpoint-august`
