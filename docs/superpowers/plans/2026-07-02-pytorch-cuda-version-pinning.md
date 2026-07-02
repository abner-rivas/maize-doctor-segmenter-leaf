# Pinning PyTorch/CUDA Versions to Eliminate pip Backtracking Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stop `pip install` from backtracking across dozens of `torch`/`nvidia-cusparse-cu12` builds by pinning an exact, verified-compatible `torch`+`torchvision`+CUDA-index combination everywhere the project installs PyTorch (Docker image, vast.ai `onstart.sh`, `pyproject.toml` bounds).

**Architecture:** No code changes — this is dependency/config pinning across three files that currently disagree with each other:
1. `pyproject.toml` declares `torch>=2.2` / `torchvision>=0.17` with **no upper bound**.
2. `Dockerfile` pre-installs `torch torchvision` (unpinned versions) from the **CUDA 12.1** wheel index, which is stale — vast.ai's default template (`vastai/pytorch:2.6.0-cuda-12.6.3-py312`, see `scripts/vastai/launch.py:19`) ships CUDA 12.6.3, and the cu121 index tops out at torch 2.5.1 (confirmed by querying `download.pytorch.org/whl/cu121/torch/`).
3. `scripts/vastai/onstart.sh` — the actual root cause — creates a fresh venv and runs `pip install -e ".[cloud]"` **directly**, with no `--index-url` and no pinned torch version. On Linux this resolves `torch` from plain PyPI, which bundles `nvidia-cusparse-cu12` and friends as platform-gated deps. Combined with the unbounded floors in `pyproject.toml`, pip's resolver has to backtrack across dozens of torch releases (2.2 through 2.12.1 today) × their respective nvidia-cu12 pins — this is exactly the "pip is looking at multiple versions of nvidia-cusparse-cu12" warning reported.

The fix: pick one verified-compatible pin, apply it consistently in `Dockerfile` and `onstart.sh` (pre-install torch from the CUDA index *before* the extras install, so the later `pip install -e ".[cloud]"` sees `torch` already satisfied and never touches PyPI's torch candidates), and bound the floor/ceiling in `pyproject.toml` so any direct `pip install -e .` also has a small search space.

**Tech Stack:** pip, PyTorch wheel index (`download.pytorch.org/whl/<cuda-tag>`), Docker, bash.

## Global Constraints

- Target CUDA wheel index: **cu126** (matches vast.ai's default template CUDA 12.6.3 exactly — confirmed via `scripts/vastai/launch.py:19` `DEFAULT_IMAGE = "vastai/pytorch:2.6.0-cuda-12.6.3-py312"`).
- Pinned versions: **torch==2.12.1**, **torchvision==0.27.1** — confirmed present on `https://download.pytorch.org/whl/cu126/` for `cp311` (`manylinux_2_28_x86_64` and `win_amd64`), and confirmed as an official compatible pair per pytorch/vision's compatibility table (`torch 2.12 <-> torchvision 0.27`).
- Python floor stays `>=3.11` (`pyproject.toml` unchanged) — cu126 wheels exist for cp311 through at least cp313; no need to touch `requires-python`.
- Never edit `raw/` or `clean/` — not applicable to this plan (dependency files only), noted per project convention.
- Do not remove the existing Docker-based smoke test workflow (`docker build -t corn-leaf-baselines .`) — extend it, don't replace it.

---

### Task 1: Bound `torch`/`torchvision` versions in `pyproject.toml`

**Files:**
- Modify: `pyproject.toml:5-6`

**Interfaces:**
- Consumes: nothing (leaf config file).
- Produces: the dependency range that `pip install -e .` (local dev, `make install`) and `pip install -e ".[cloud]"` (onstart.sh, Task 3) must resolve within.

- [ ] **Step 1: Reproduce the unbounded resolution risk**

Run (from repo root, PowerShell or Bash):
```bash
pip index versions torch
```
Expected: a long list of versions from `0.1.1` (or similar ancient release) through `2.12.1` with no upper bound in `pyproject.toml` to narrow it — this is the search space pip must consider on a fresh Linux install.

- [ ] **Step 2: Add version ceilings**

In `pyproject.toml`, change:
```toml
dependencies = [
    "torch>=2.2",
    "torchvision>=0.17",
    "timm>=0.9",
```
to:
```toml
dependencies = [
    "torch>=2.2,<2.13",
    "torchvision>=0.17,<0.28",
    "timm>=0.9",
```

- [ ] **Step 3: Verify the file still parses and the project installs metadata cleanly**

Run:
```bash
pip install --dry-run -e . 2>&1 | tail -20
```
Expected: no syntax/metadata errors from `pyproject.toml` (the dry-run may still fail later on unrelated network/version issues on this Windows CPU machine — that's fine; the goal here is confirming the TOML is valid and hatchling reads the new constraint strings without error). If it errors immediately with a TOML parse error, fix the syntax.

- [ ] **Step 4: Commit**

```bash
git add pyproject.toml
git commit -m "chore: bound torch/torchvision version ranges to curb pip backtracking"
```

---

### Task 2: Re-point the Dockerfile at the cu126 index with pinned versions

**Files:**
- Modify: `Dockerfile:1-4` (header comment), `Dockerfile` (the `RUN python -m venv venv ...` block)

**Interfaces:**
- Consumes: the version pin decided in Global Constraints (torch==2.12.1, torchvision==0.27.1, cu126).
- Produces: a Docker image whose venv has torch/torchvision pre-satisfied before `pip install -e ".[cloud]"` runs, matching what Task 3 does for onstart.sh.

- [ ] **Step 1: Update the header comment**

Current (`Dockerfile:1-4`):
```dockerfile
# Imagen para entrenar en una GPU remota (p.ej. vast.ai). Python 3.11 (requisito del
# proyecto, ver pyproject.toml) + wheels de PyTorch con CUDA 12.1 desde el índice oficial
# de PyTorch. La GPU/drivers los aporta el host (nvidia-container-toolkit en local, o
# directamente el runtime de vast.ai); la imagen no necesita el toolkit de CUDA completo.
```
Replace with:
```dockerfile
# Imagen para entrenar en una GPU remota (p.ej. vast.ai). Python 3.11 (requisito del
# proyecto, ver pyproject.toml) + wheels de PyTorch con CUDA 12.6 desde el índice oficial
# de PyTorch — misma línea de CUDA que la plantilla vastai/pytorch:2.6.0-cuda-12.6.3-py312
# usada por scripts/vastai/launch.py. La GPU/drivers los aporta el host
# (nvidia-container-toolkit en local, o directamente el runtime de vast.ai); la imagen no
# necesita el toolkit de CUDA completo. torch/torchvision van con versión exacta (no solo
# el índice) porque sin tope pip hace backtracking entre builds de torch y sus
# dependencias nvidia-cu12, dejando el install colgado minutos buscando combinaciones.
```

- [ ] **Step 2: Pin the install command**

Current:
```dockerfile
RUN python -m venv venv \
    && venv/bin/pip install --no-cache-dir --upgrade pip \
    && venv/bin/pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cu121 \
    && venv/bin/pip install --no-cache-dir -e ".[cloud]"
```
Replace with:
```dockerfile
RUN python -m venv venv \
    && venv/bin/pip install --no-cache-dir --upgrade pip \
    && venv/bin/pip install --no-cache-dir torch==2.12.1 torchvision==0.27.1 --index-url https://download.pytorch.org/whl/cu126 \
    && venv/bin/pip install --no-cache-dir -e ".[cloud]"
```

- [ ] **Step 3: Build the image and confirm no backtracking**

Run (from repo root):
```bash
docker build -t corn-leaf-baselines . 2>&1 | tee docker-build.log
```
Expected: the log contains no `INFO: pip is looking at multiple versions` lines, and the `torch`/`torchvision` install step completes in well under a minute of resolver time (the bulk of the time should be download, not backtracking). Grep to confirm:
```bash
grep -c "looking at multiple versions" docker-build.log
```
Expected: `0`.

- [ ] **Step 4: Sanity-check the installed versions inside the image**

```bash
docker run --rm corn-leaf-baselines venv/bin/python -c "import torch, torchvision; print(torch.__version__, torchvision.__version__)"
```
Expected output: `2.12.1+cu126 0.27.1+cu126`.

- [ ] **Step 5: Clean up the build log and commit**

```bash
rm -f docker-build.log
git add Dockerfile
git commit -m "fix: pin torch/torchvision to 2.12.1/0.27.1 on cu126 index in Docker image"
```

---

### Task 3: Fix `onstart.sh` to pre-install pinned torch/torchvision (the actual root cause)

**Files:**
- Modify: `scripts/vastai/onstart.sh`

**Interfaces:**
- Consumes: the same pin as Task 2 (torch==2.12.1, torchvision==0.27.1, cu126) — must stay identical to the Dockerfile so the two provisioning paths (build-your-own image vs. onstart.sh on the stock template) produce the same environment.
- Produces: a venv where `pip install -e ".[cloud]"` never has to resolve `torch` from PyPI, eliminating the backtracking reported by the user.

- [ ] **Step 1: Reproduce the bug in isolation (documentation of root cause, no code yet)**

This step is evidence-gathering, not a fix. Read `scripts/vastai/onstart.sh` and confirm the current install line has no `--index-url` and no version pin:
```bash
grep -n "pip install" scripts/vastai/onstart.sh
```
Expected (before fix):
```
venv/bin/pip install --no-cache-dir -e ".[cloud]"
```
This is the line that, on a fresh Linux venv, forces pip to resolve `torch>=2.2` (now `<2.13` after Task 1, but still a 10+ release search space) from plain PyPI — triggering the nvidia-cusparse-cu12 backtracking.

- [ ] **Step 2: Add the pinned pre-install step**

Current:
```bash
echo "[onstart] Instalando el proyecto (pip install -e .[cloud])"
venv/bin/pip install --no-cache-dir -e ".[cloud]"
```
Replace with:
```bash
echo "[onstart] Instalando PyTorch (CUDA 12.6, versión fija para evitar backtracking de pip)"
venv/bin/pip install --no-cache-dir torch==2.12.1 torchvision==0.27.1 --index-url https://download.pytorch.org/whl/cu126

echo "[onstart] Instalando el proyecto (pip install -e .[cloud])"
venv/bin/pip install --no-cache-dir -e ".[cloud]"
```

- [ ] **Step 3: Verify the script is still valid bash**

```bash
bash -n scripts/vastai/onstart.sh
```
Expected: no output (exit code 0 means syntax is valid).

- [ ] **Step 4: Verify the fix end-to-end via the Docker image (closest available stand-in for a vast.ai Linux instance)**

There's no vast.ai instance in this workflow, so validate the exact same two-step install pattern that `onstart.sh` now uses, on Linux, via the already-built image from Task 2:
```bash
docker run --rm corn-leaf-baselines bash -c "
  python -m venv /tmp/onstart-venv &&
  /tmp/onstart-venv/bin/pip install --no-cache-dir --upgrade pip &&
  /tmp/onstart-venv/bin/pip install --no-cache-dir torch==2.12.1 torchvision==0.27.1 --index-url https://download.pytorch.org/whl/cu126 &&
  /tmp/onstart-venv/bin/pip install --no-cache-dir -e '.[cloud]' 2>&1 | tail -20
"
```
Expected: the final `pip install -e '.[cloud]'` line reports `torch` and `torchvision` as "already satisfied" (or installs quickly without re-resolving), with no backtracking warnings.

- [ ] **Step 5: Commit**

```bash
git add scripts/vastai/onstart.sh
git commit -m "fix: pin torch/torchvision before installing project in onstart.sh, matching Dockerfile"
```

---

### Task 4: Update `docs/es/deployment/vast-ai.md` to match the new pins

**Files:**
- Modify: `docs/es/deployment/vast-ai.md:15` (flow table, "Instalación" row), `docs/es/deployment/vast-ai.md:40` (Docker image description)

**Interfaces:**
- Consumes: the final wording/behavior from Tasks 2 and 3 (must describe the actual two-step install, not the old one-liner).
- Produces: docs that don't contradict the scripts (this file currently has unrelated uncommitted formatting changes per `git status` — preserve those, only touch the content described below).

- [ ] **Step 1: Update the flow table's "Instalación" row**

Current (`docs/es/deployment/vast-ai.md`, in the `## Flujo, local vs. vast.ai` table — note the table may be pipe-aligned or not depending on prior uncommitted formatting passes; match on content, not whitespace):
```
| Instalación | `make install` (venv creado a mano) | `venv/bin/pip install -e ".[cloud]"` (venv creado por `onstart.sh`) |
```
Replace the "Instancia vast.ai (GPU)" cell content with:
```
`onstart.sh` instala torch/torchvision fijos (CUDA 12.6) y luego `-e ".[cloud]"`
```
so the row reads (adjust padding/spaces to match whatever alignment style the rest of the table already uses in the file you're editing):
```
| Instalación | `make install` (venv creado a mano) | `onstart.sh` instala torch/torchvision fijos (CUDA 12.6) y luego `-e ".[cloud]"` |
```

- [ ] **Step 2: Update the Docker image description to say CUDA 12.6 and mention the pin**

In the `## 2. Imagen reproducible (Docker)` section, find the paragraph describing the Dockerfile (it mentions `python:3.11-slim`, "CUDA 12.1", and "índice oficial de PyTorch" — exact line-wrapping may vary). Replace the phrase:
```
+ wheels de PyTorch con CUDA 12.1 desde el índice oficial de PyTorch,
```
with:
```
+ wheels de PyTorch 2.12.1 / torchvision 0.27.1 con CUDA 12.6 desde el índice oficial de
PyTorch (misma línea CUDA que la plantilla vast.ai de abajo),
```
and insert a new sentence immediately after the existing sentence that ends "...que el entorno instala limpio antes de tocar una instancia real." (still within the same paragraph or as an appended sentence), reading:
```
`scripts/vastai/onstart.sh` instala el mismo par torch/torchvision antes de
`pip install -e ".[cloud]"` — mantener ambos en sync evita que pip haga backtracking
resolviendo `torch` desde PyPI sin índice CUDA.
```
Reflow the paragraph naturally (this repo's markdown is not line-length-enforced by a formatter in this area — match the wrapping style already present in the file you're editing).

- [ ] **Step 3: Proofread the diff**

```bash
git diff docs/es/deployment/vast-ai.md
```
Expected: only the two content changes above are new on top of whatever formatting changes were already pending in the working tree; no accidental reversion of the existing uncommitted formatting.

- [ ] **Step 4: Commit**

```bash
git add docs/es/deployment/vast-ai.md
git commit -m "docs: describe pinned torch/torchvision CUDA 12.6 install in vast.ai guide"
```

---

### Task 5: Final validation — full local dev install still works

**Files:**
- None modified — verification only.

**Interfaces:**
- Consumes: the final state of `pyproject.toml` (Task 1), `Dockerfile` (Task 2), `onstart.sh` (Task 3).
- Produces: confidence that local Windows/CPU dev (`make install`) still resolves fine with the new ceiling, since that path never hits the CUDA index.

- [ ] **Step 1: Run the local install target**

```bash
make install
```
Expected: completes without error; `pip` resolves CPU-appropriate `torch<2.13` for the local Python. If it fails because the local venv's Python version has no matching torch wheel, note the exact error — that's a separate, pre-existing local-environment issue (not introduced by this plan) and should be reported to the user rather than silently worked around.

- [ ] **Step 2: Confirm the installed torch version respects the new ceiling**

```bash
python -c "import torch; print(torch.__version__)"
```
Expected: a version `>=2.2` and `<2.13`.

- [ ] **Step 3: Re-run the Docker build one final time end-to-end to confirm nothing regressed across all three files together**

```bash
docker build --no-cache -t corn-leaf-baselines . 2>&1 | grep -c "looking at multiple versions"
```
Expected: `0`.

- [ ] **Step 4: Commit nothing further** (this task is verification-only) — if any step fails, fix the relevant task above and re-run this task's steps before considering the plan complete.
