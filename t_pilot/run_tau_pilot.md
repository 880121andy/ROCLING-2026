# RQ1 NLA Pipeline — Environment Rebuild & Troubleshooting Runbook

TWCC, on a compute node (`hgpn*`). Do **not** run GPU work on a login node (`*-lgn*`). The full pipeline needs **two conda environments**; see §0 for why.

---

## 0. Why two conda environments (read this first)

| Env | transformers | Handles | Reason |
|---|---|---|---|
| `rocling` | **5.3.0** | SGLang server only (AV inference service) | SGLang 0.5.10 hard-pins `transformers==5.3.0` |
| `nla_client` | **4.57.6** | extract / verbalize / score / calibrate (everything that touches the tokenizer directly) | The checkpoint's injection char `㈎` gets merged away under the transformers 5.x chat template; only 4.x tokenizes it correctly |

The two talk over HTTP (localhost:port), so the version conflict never surfaces: the server runs in `rocling`, the client in `nla_client`. `tau_pilot.py` dispatches its subprocesses using two interpreter paths that point straight at each env:

```python
# in tau_pilot.py (around line 66)
PY_SERVER = "/home/tyleryeh47/.conda/envs/rocling/bin/python"     # SGLang
PY_CLIENT = "/home/tyleryeh47/.conda/envs/nla_client/bin/python"  # everything else
```

---

## 1. Load modules

```bash
module load miniconda3/24.11.1
module load cuda/12.4
# If SGLang's JIT compile complains about a missing <version> header (see §6-A),
# you also need a gcc >= 11:
module avail gcc                 # list what's available
module load gcc/<version >= 11>  # gives nvcc's host compiler the C++20 <version> header
```

> `module load` only affects the current shell. Reload after every new terminal, reconnect, or re-login, otherwise SGLang's JIT compile fails again at startup.

---

## 2. Build `rocling` (SGLang server, transformers 5.3.0)

```bash
conda create -n rocling python=3.11 -y
conda activate rocling

pip install torch transformers safetensors httpx orjson pyyaml numpy accelerate
pip install "sglang[all]>=0.5.6"      # pulls in transformers==5.3.0
conda install -c conda-forge ninja    # needed for JIT compile (see §6-B)
```

Verify (torch's CUDA must match what the driver supports, which is 12.8 on this node):

```bash
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
# expect: 2.9.1+cu128 12.8 True
```

---

## 3. Build `nla_client` (extraction / client, transformers 4.57.6)

```bash
conda create -n nla_client python=3.11 -y
conda activate nla_client

# Don't pin tokenizers (it conflicts with 4.57's deps); let transformers pick it.
pip install "transformers>=4.50,<5" safetensors httpx orjson pyyaml numpy pandas pyarrow accelerate

# torch must match the driver; don't take the default latest build (see §6-C).
pip install "torch==2.9.1" --index-url https://download.pytorch.org/whl/cu128
```

Verify two things:

```bash
# (a) torch reaches the GPU
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
# expect: 2.9.1+cu128 12.8 True
```

```bash
# (b) the injection char tokenizes to a single token in context (only 4.x passes)
python - <<'PY'
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("checkpoint/nla-qwen2.5-7b-L20-av")
c = "Here is the vector:\n\n<concept>㈎</concept>\n\nPlease provide an explanation."
ids = list(tok.apply_chat_template([{"role":"user","content":c}], tokenize=True, add_generation_prompt=True))
print("OK" if ids.count(149705) == 1 else "FAIL", "count =", ids.count(149705))
PY
# expect: OK count = 1
```

---

## 4. Download checkpoints (AV and AR are two different sets of weights!)

```bash
hf download kitft/nla-qwen2.5-7b-L20-av --local-dir checkpoint/nla-qwen2.5-7b-L20-av
huggingface-cli download kitft/nla-qwen2.5-7b-L20-ar --local-dir checkpoint/nla-qwen2.5-7b-L20-ar

hf download kitft/nla-gemma3-12b-L32-av --local-dir checkpoint/nla-gemma3-12b-L32-av
hf download kitft/nla-gemma3-12b-L32-ar --local-dir checkpoint/nla-gemma3-12b-L32-ar
```

**After downloading, always confirm AV ≠ AR** (hit this today: the AR folder was once a full copy of AV):

```bash
grep -H "^role:" checkpoint/nla-qwen2.5-7b-L20-av/nla_meta.yaml \
                 checkpoint/nla-qwen2.5-7b-L20-ar/nla_meta.yaml
# expect: the av file says  role: av ; the ar file says  role: ar  (or critic).
# If both say av, the AR download is wrong; re-fetch it.

ls -la checkpoint/nla-qwen2.5-7b-L20-av/*.safetensors
ls -la checkpoint/nla-qwen2.5-7b-L20-ar/*.safetensors
# shard count/sizes should differ (AR has 3 shards, AV has 4). Identical means duplicate files.
```

> Run the same role/size check on the Gemma checkpoints (`nla-gemma3-12b-L32-av/-ar`) too.

---

## 5. Run the pilot

```bash
# put rocling's ninja on PATH (the server subprocess needs it)
export PATH="/home/tyleryeh47/.conda/envs/rocling/bin:$PATH"

# launch from the nla_client env (only the PY_SERVER line in tau_pilot.py switches to rocling)
conda activate nla_client

PYTHONPATH=/home/tyleryeh47/ROCLING-2026/verify_script \
CUDA_VISIBLE_DEVICES=0 python tau_pilot.py --model qwen \
    --pairs-csv ../rq1_review_all.csv --ckpt-root checkpoint \
    --nla-repo natural_language_autoencoders --force
```

What each flag is for:

- `PYTHONPATH=.../verify_script`: `extract_activations.py` does `import verify_tokenization` (shared span-location logic), and that file lives in `verify_script/`.
- `--force`: wipe old intermediate artifacts and rerun the whole batch. Use it only when you really mean to start over; to resume (reuse existing activations/verbalizations), drop `--force`.
- Output lands in `results/pilot_qwen/`: look at `preview.md` first, then `gate.summary.md` and the τ summary printed to the terminal.

---

## 6. Troubleshooting quick reference (in the order we hit them)

### A. SGLang JIT compile: `fatal error: version: No such file or directory`
nvcc's host C++ compiler is too old and lacks the C++20 `<version>` header.

**Proper fix:** `module load gcc/<version >= 11>` in the shell that launches the server. Verify:
```bash
echo | g++ -std=c++20 -E -x c++ - -include version >/dev/null 2>&1 && echo OK || echo BROKEN
```
**Temporary workaround** (fine for the small pilot, the perf hit doesn't matter): add `--disable-cuda-graph --disable-piecewise-cuda-graph` to the SGLang launch flags. CUDA graph and piecewise CUDA graph are two independent mechanisms, so turn off both; inference can still trigger other JIT kernels (e.g. `resolve_future_token_ids`), which is why the gcc fix stays the first choice.

### B. `FileNotFoundError: 'ninja'`
JIT compile needs ninja, but it's installed in the wrong env. SGLang runs in `rocling`, so:
```bash
conda install -n rocling -c conda-forge ninja
# and make sure rocling/bin is on PATH (see the export PATH in §5)
conda run -n rocling which ninja   # should print a path
```

### C. `RuntimeError: The NVIDIA driver on your system is too old (found version 12040)`
`pip install torch` grabbed a torch built against too new a CUDA (e.g. 2.13). The driver only supports up to CUDA 12.8. Install a matching build (same as rocling):
```bash
/home/tyleryeh47/.conda/envs/nla_client/bin/pip install \
    "torch==2.9.1" --index-url https://download.pytorch.org/whl/cu128
```

### D. `injection token appears 0× in canonical prompt`
Wrong tokenizer version. `nla_client` must be transformers 4.x; 5.x merges `㈎` away in context. Confirm count==1 with the check in §3(b).

### E. `sidecar role='av', expected 'critic' or 'ar'`
The folder you pointed at as AR actually contains AV files (wrong download). Re-fetch AR and confirm AV ≠ AR with the role/size check in §4.

### F. `pip's dependency resolver ... transformers==5.3.0 required by sglang`
You've mixed the two envs. SGLang wants 5.3.0, the NLA client wants 4.x. Don't force both into one env; use two conda envs (§0).

### G. `HFValidationError: Repo id must be in the form ...`
The `--ckpt-root` path doesn't exist, so transformers treats it as an HF repo id and tries to download it. Check that the path you passed in (relative vs absolute; `checkpoint` vs `/checkpoint`) actually exists.

### H. `ModuleNotFoundError: No module named 'verify_tokenization'`
`PYTHONPATH` isn't set. Point it at the `verify_script/` folder where `verify_tokenization.py` lives (see §5).

### I. Process `Killed` with no traceback
Usually you're on a **login node** and the memory cgroup limit killed it. Move to a compute node:
```bash
srun --gres=gpu:1 --partition=<gpu partition> --account=<your project> --pty bash
```

---

## 7. Snapshot (save this, so the full 360-sentence run and Gemma can reuse it)

Once both sides work end to end, freeze the deps so you don't have to rederive them next time:
```bash
conda run -n rocling    pip freeze > requirements-rocling.txt
conda run -n nla_client pip freeze > requirements-nla_client.txt
# or export the full environments:
conda env export -n rocling    > env-rocling.yml
conda env export -n nla_client > env-nla_client.yml
```

---

## 8. One-page flow (mental model)

```
pick sentences (nla_client) ─► extract_activations (nla_client, transformers 4.x)
                                  │  needs PYTHONPATH=verify_script
                                  ▼
                    SGLang AV server (rocling, transformers 5.x)  ◄── needs gcc>=11 + ninja + torch cu128
                                  │  HTTP :30000, injects the vector via input_embeds
                                  ▼
                    verbalize (nla_client) ─► score_roundtrip AR (nla_client, in-process, needs torch.cuda OK)
                                  │
                                  ▼
                    calibrate_gate → τ → preview.md
```