# SparseChron — Rules

Coding standards, constraints, and project rules. Every contributor (including
future-you resuming after a break) must follow these.

---

## 1. Language & Style

| Rule | Detail |
|---|---|
| **Python version** | 3.10 — match Kaggle's runtime exactly |
| **Type hints** | Required on all function signatures (params + return) |
| **Docstrings** | Required on all public classes and functions; Google style |
| **Linter** | `ruff` — zero warnings before any commit |
| **Formatter** | `ruff format` — run before every commit |
| **Line length** | 88 characters (ruff default) |
| **Imports** | Absolute imports only (`from sparsechron.models.gaussians import ...`), no relative imports |
| **Naming** | `snake_case` for functions/variables, `PascalCase` for classes, `UPPER_SNAKE` for constants |

---

## 2. Project Constraints

### Hard Constraints (never violate)

1. **No silent failures.** Every function that can fail must raise a clear
   exception or return an explicit error. Never `pass` in an `except` block
   without logging.

2. **Checkpoint correctness.** If you change any trainable parameter, the
   checkpoint save/load round-trip must still work. Test this after every
   model change.

3. **Reproducibility.** All random operations must use seeded RNG. The
   checkpoint must capture and restore all RNG states (Python, NumPy, PyTorch,
   CUDA). Given the same checkpoint + config, training must produce identical
   results.

4. **VRAM safety.** Never exceed the `max_gaussians` cap. Every densification
   step must check the Gaussian count before adding new ones. If at cap, skip
   densification.

5. **No hardcoded paths.** All file paths come from config or CLI arguments.
   Never write `/home/user/...` or `C:\Users\...` in source code. Use
   `pathlib.Path` throughout.

6. **Kaggle compatibility.** Code must run on Kaggle's default Python 3.10 +
   PyTorch 2.x + CUDA 12.x environment. Test any new dependency on Kaggle
   before merging.

### Soft Constraints (follow unless there's a good reason not to)

7. **Single responsibility.** Each module does one thing. Don't put training
   logic in the model class or rendering logic in the loss function.

8. **No global state.** No module-level mutable variables. Pass state
   explicitly via function arguments or class attributes.

9. **Fail fast.** Validate inputs at function entry. Check tensor shapes,
   dtypes, and value ranges early — don't let bad data propagate silently.

10. **Prefer composition over inheritance.** `GaussianModel` uses a
    `DeformationMLP`, it does not subclass it.

---

## 3. Git Workflow

| Rule | Detail |
|---|---|
| **Branching** | `main` is always stable. Develop on feature branches (`feat/deformation-mlp`, `fix/checkpoint-resume`) |
| **Commits** | Conventional commits: `feat:`, `fix:`, `refactor:`, `test:`, `docs:`, `chore:` |
| **Commit size** | Small, atomic commits. One logical change per commit |
| **`.gitignore`** | Never commit: `data/`, `outputs/`, `*.ckpt`, `*.npy`, `__pycache__/`, `.venv/` |
| **Notebooks** | Clear all cell outputs before committing `.ipynb` files |
| **Secrets** | No API keys, tokens, or credentials in source code — ever |

---

## 4. Tensor Conventions

These conventions are non-negotiable — inconsistency here causes silent bugs.

| Convention | Standard |
|---|---|
| **Image format** | `(C, H, W)`, float32, range `[0, 1]` — convert on load |
| **Batch images** | `(B, C, H, W)` |
| **Depth maps** | `(1, H, W)`, float32, meters — `0.0` or `NaN` for invalid |
| **Quaternions** | `(w, x, y, z)` order — scalar first |
| **Coordinate system** | OpenCV: +X right, +Y down, +Z forward |
| **Extrinsics** | World-to-camera `(4, 4)` matrix |
| **Device** | Never assume device. Always use `tensor.to(device)` or respect the model's device |
| **Dtype** | Default float32. Use float16 only inside `torch.cuda.amp.autocast` blocks |

---

## 5. Checkpoint Rules

1. **Save atomically.** Write to a temp file, then `os.replace()` to the
   final path. Never leave a half-written checkpoint.

2. **Save metadata.** Every checkpoint includes: iteration number, config
   dict, wall-clock time, current metrics, and a version string.

3. **Version the format.** The checkpoint dict must include a `"version"` key.
   If you change the checkpoint schema, bump the version and write migration
   logic in `load_checkpoint()`.

4. **Validate on load.** After loading a checkpoint, assert that tensor shapes
   match the current config. If they don't, raise a clear error — don't
   silently continue with mismatched state.

5. **30-minute timer.** The checkpoint timer measures wall-clock time, not
   iteration count. Use `time.monotonic()`, not `time.time()` (immune to
   system clock changes).

---

## 6. Testing Rules

| Rule | Detail |
|---|---|
| **Framework** | `pytest` |
| **CPU-only** | All tests must run on CPU (no GPU required). Use small synthetic tensors |
| **Naming** | `test_<module>.py` → `test_<function_name>()` |
| **Fixtures** | Use `@pytest.fixture` for reusable test data (e.g. a small GaussianModel with 10 Gaussians) |
| **No network** | Tests must not download anything or make HTTP requests |
| **Speed** | Individual tests should complete in < 5 seconds |
| **Coverage target** | Core modules (`models/`, `losses/`, `training/checkpoint.py`) must have tests. Preprocessing wrappers (DUSt3R, Depth Anything) are tested manually on Kaggle |

---

## 7. Dependency Rules

1. **Pin everything.** `requirements.txt` must specify exact versions
   (`gsplat==1.0.0`, not `gsplat>=1.0`).

2. **Minimal dependencies.** Don't add a library for something achievable in
   10 lines of code. Every new dependency is a potential Kaggle compatibility
   risk.

3. **Test on Kaggle first.** Before adding a new dependency, verify it
   installs and imports correctly in a fresh Kaggle notebook.

4. **Vendor if necessary.** If a dependency has Kaggle issues, copy the
   specific function you need into `utils/` with attribution, rather than
   fighting the install.

---

## 8. Documentation Rules

1. **README.md** is the entry point. It must contain: what the project does,
   how to install, how to run (preprocess → train → view), and results.

2. **Docstrings** explain *why*, not *what*. The code shows what happens; the
   docstring explains the reasoning.

3. **Comments** are for non-obvious logic only. Don't comment
   `# increment counter` above `counter += 1`.

4. **Planning docs** (PRD, TechSpec, etc.) are living documents. Update them
   when decisions change — they are your memory across sessions.

---

## 9. Kaggle-Specific Rules

1. **Never rely on `/kaggle/working/` persistence.** It is wiped between
   sessions. Always commit outputs to a Kaggle Dataset.

2. **Install deps at the top of every notebook.** Don't assume previous
   installs persist.

3. **Use `%%time` on expensive cells.** Know how long each stage takes so you
   can plan your 9-hour session budget.

4. **Print a session summary at the end.** Iteration reached, time elapsed,
   checkpoint path, metrics — so you know where to resume.

5. **Avoid `!git clone` for the main repo in notebooks.** Upload the code as
   a Kaggle Dataset instead — it's faster and doesn't depend on GitHub
   availability.

---

## 10. Performance Rules

1. **Profile before optimizing.** Don't guess at bottlenecks. Use
   `torch.profiler` or `time.monotonic()` measurements.

2. **Mixed precision is mandatory.** Always train inside
   `torch.cuda.amp.autocast(dtype=torch.float16)`. The GradScaler handles
   loss scaling.

3. **No Python loops over Gaussians.** All per-Gaussian operations must be
   vectorized (PyTorch tensor ops) or handled by gsplat's CUDA kernels.

4. **Batch depth estimation.** Run Depth Anything V2 on multiple images in a
   single batch where VRAM allows, rather than one-by-one.

---

*These rules exist to prevent the bugs that will cost you hours of debugging
on Kaggle, where iteration cycles are slow and sessions are limited.*
