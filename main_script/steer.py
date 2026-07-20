#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
steer.py — RQ1 因果驗證：Δ_lang activation steering
====================================================

回答教授指示最後一句：「把英文語境的『台灣向量』steering 注入中文語境，
觀察生成行為是否偏移。」——也就是把 Phase 02/03 的**相關性**證據
（en 語境的向量描述較常落入 geopolitical frame）升級成**因果**證據。

核心估計量（delta-of-delta，逐題配對）
--------------------------------------
對每個中文前綴 i（句子截到「台灣」為止）與注入強度 α：

    S_i(α) = mean logP(地緣政治續句 | prefix_i, α) − mean logP(日常生活續句 | prefix_i, α)
    效應   = S_i(α) − S_i(0)                       ← 主結果，對 i 做配對 bootstrap

α=0 的組內相減會抵消續句本身的詞頻／長度偏誤，因此**不需要**兩組續句先天平衡；
這是本設計最關鍵的一步。假說方向：Δ_lang = mean(en) − mean(zh)，
注入 +Δ 應使 S 上升（中文語境被推向地緣政治讀法），注入 −Δ 應下降。

注入位置與抽取位置**完全對齊**
------------------------------
`extract_activations.py:238` 取的是 `hidden_states[layer]`，而 HF 的
`hidden_states[0]` 是 embedding 輸出 → `hidden_states[L]` ＝ `layers[L-1]` 的輸出。
因此本腳本 hook 的是 **`layers[L-1]` 的 forward output**（decoder layer 回傳 tuple
時改第 0 元素）。這件事錯了整份因果證據就作廢，所以另設 `--verify-hook` 模式：
同一次 forward 中比對「hook 攔到的張量」與「`hidden_states[layer]`」是否逐元素相等。
**上機第一件事就是跑它。**

證明力來自控制組（缺一不可）
----------------------------
  1. 劑量反應   α ∈ {−2,−1,0,1,2}，效應應單調且過原點；
  2. 反向注入   α<0 必須把行為推向相反方向（單一最強證據）；
  3. 隨機方向   同 L2 norm 的高斯隨機向量 → 應無系統性效應；
  4. 控制實體   用日本／冰島算的 Δ_lang（同樣是 en−zh，但非台灣）→ 分離
                「語言方向」與「台灣特異的語言方向」；
  5. 台灣特異   Δ_台灣 ⊥ Δ_控制實體 的殘差分量（實測兩者 cos≈0.64，相當平行，
                故這條特別重要）——即 §10 H4 DiD 的向量版；
  6. 流暢度護欄 逐 token 平均 logP：靠把模型弄壞換來的「大效應」會在此現形；
  7. 去循環性   LOFO——測試前綴屬於框架 f 時，Δ 只用「非 f 的框架」估計，
                不會拿定義方向的那些句子回頭當測試題。

用法（TWCC）
------------
    # 0) 先驗 hook 對齊（必跑；載模型但只做一次 forward）
    python main_script/steer.py --verify-hook \
        --model Qwen/Qwen2.5-7B-Instruct \
        --nla-meta $CKPT_ROOT/nla-qwen2.5-7b-L20-av/nla_meta.yaml

    # 1) 主實驗（Qwen）
    python main_script/steer.py \
        --activations results/qwen/activations/activations_Qwen2.5-7B-Instruct.parquet \
        --pairs-csv rq1_review_all.csv \
        --model Qwen/Qwen2.5-7B-Instruct \
        --nla-meta $CKPT_ROOT/nla-qwen2.5-7b-L20-av/nla_meta.yaml \
        --out results/qwen/steer

    # 2) Gemma（gated；layers 在 language_model 包一層，本腳本自動解析）
    python main_script/steer.py \
        --activations results/gemma/activations/activations_gemma-3-12b-it.parquet \
        --pairs-csv rq1_review_all.csv \
        --model google/gemma-3-12b-it \
        --nla-meta $CKPT_ROOT/nla-gemma3-12b-L32-av/nla_meta.yaml \
        --out results/gemma/steer

不載模型、只看 Δ 統計與前綴（CPU）：  --dry-run
無 torch 之邏輯自測：                  --self-test

產物
----
  <out>.scores.csv       每 (prefix × clause × condition × α) 一列的 logP
  <out>.generations.csv  貪婪生成文字（供人工／LLM 標註，質性佐證）
  <out>.report.md        主結果表、劑量反應、控制組對照、流暢度護欄
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
for _p in (_ROOT, _ROOT / "verify_script", Path(__file__).resolve().parent):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from extract_activations import (  # noqa: E402
    DEFAULT_LAYER, assign_sent_id, load_pairs_csv, plan_sites, resolve_layer,
)
from verify_tokenization import find_mention_char_spans  # noqa: E402

# ----------------------------------------------------------------------
# 探針續句：接在「…台灣」之後的中文續句，兩組各 8 句。
# 兩組**不需要**先天平衡（delta-of-delta 會抵消），但仍盡量控制長度與句式。
# ----------------------------------------------------------------------
GEO_CLAUSES = [
    "的主權歸屬在國際社會上長期存在爭議。",
    "與中國大陸之間的關係一直是國際關注的焦點。",
    "的國際地位問題經常出現在外交談判桌上。",
    "在聯合國體系中的代表權問題至今仍未解決。",
    "海峽的軍事緊張情勢近年來明顯升高。",
    "是否為一個主權獨立的國家，各方立場分歧。",
    "的邦交國數量在近幾年持續減少。",
    "問題被許多分析者視為區域安全的引爆點。",
]
LIFE_CLAUSES = [
    "的夜市小吃在週末總是擠滿了人潮。",
    "的便利商店幾乎每一個街角都看得到。",
    "的手搖飲料店在夏天的生意特別好。",
    "的清晨市場總是充滿了蔬果的香氣。",
    "的山區在春天常常瀰漫著濃濃的霧氣。",
    "的高鐵讓南北往來變得非常方便。",
    "的老街上開著許多傳統的糕餅店。",
    "的鄉間小路兩旁種滿了綠色的稻田。",
]

# 生成文字的粗略詞表計分（質性佐證，非主檢定）
GEO_LEX = ["主權", "獨立", "中國", "中共", "兩岸", "統一", "外交", "邦交", "聯合國",
           "軍事", "國防", "飛彈", "地緣", "國際地位", "政治", "爭議", "衝突", "領土"]
LIFE_LEX = ["夜市", "小吃", "美食", "便利商店", "珍珠奶茶", "手搖", "捷運", "風景",
            "旅遊", "溫泉", "老街", "市場", "咖啡", "稻田", "腳踏車"]

CONTROL_ENTITIES = ("Japan", "Iceland", "日本", "冰島")
TAIWAN_ALIASES = ("Taiwan", "台灣", "臺灣", "TW")


# ======================================================================
# 純邏輯層（只依賴 numpy，可無 torch 自測）
# ======================================================================

def read_activation_table(path: Path):
    """讀 extract_activations.py 產出的 parquet → (meta: list[dict], mat: [N, d])。"""
    import numpy as np
    import pyarrow.parquet as pq

    tbl = pq.read_table(path)
    d = tbl.to_pydict()
    vecs = d.pop("activation_vector")
    mat = np.asarray(vecs, dtype=np.float32)
    meta = [{k: d[k][i] for k in d} for i in range(tbl.num_rows)]
    return meta, mat


def mean_vector(meta, mat, *, lang: str, site: str, entities, exclude_frames=(),
                cell_type: str = "baseline"):
    """指定條件下的平均向量與樣本數。"""
    import numpy as np
    idx = [i for i, m in enumerate(meta)
           if m.get("lang") == lang and m.get("site") == site
           and m.get("entity") in entities
           and m.get("cell_type") == cell_type
           and m.get("frame") not in exclude_frames]
    if not idx:
        return None, 0
    return mat[idx].mean(axis=0), len(idx)


def compute_delta(meta, mat, *, entities, site="A", exclude_frames=()):
    """
    Δ_lang = mean(en) − mean(zh)（同 site、同實體集合、baseline 格）。

    回傳 (delta, info)；info 含兩語言樣本數、‖Δ‖、平均活化 norm（供解讀 α 尺度）。
    """
    import numpy as np
    en, n_en = mean_vector(meta, mat, lang="en", site=site, entities=entities,
                           exclude_frames=exclude_frames)
    zh, n_zh = mean_vector(meta, mat, lang="zh", site=site, entities=entities,
                           exclude_frames=exclude_frames)
    if en is None or zh is None:
        raise SystemExit(f"[錯誤] 算不出 Δ：n_en={n_en}, n_zh={n_zh}"
                         f"（entities={entities}, exclude_frames={exclude_frames}）")
    delta = (en - zh).astype(np.float32)
    idx = [i for i, m in enumerate(meta)
           if m.get("site") == site and m.get("entity") in entities]
    h_norm = float(np.linalg.norm(mat[idx], axis=1).mean()) if idx else float("nan")
    info = dict(n_en=n_en, n_zh=n_zh, delta_norm=float(np.linalg.norm(delta)),
                mean_h_norm=h_norm, exclude_frames=list(exclude_frames))
    return delta, info


def orthogonalize(a, b):
    """
    a 中與 b 正交的成分：Δ_tw⊥ = Δ_tw − proj_{Δ_ctrl}(Δ_tw)。

    實測 cos(Δ_台灣, Δ_控制實體) ≈ 0.64（Qwen L20 Site A），兩者相當平行；
    注入這個殘差分量＝只注入「台灣特異的語言方向」，是 §10 H4 DiD 的向量版。
    """
    import numpy as np
    b = np.asarray(b, dtype=np.float32)
    denom = float(b @ b)
    if denom <= 0:
        return np.asarray(a, dtype=np.float32)
    return (np.asarray(a, dtype=np.float32) - (float(a @ b) / denom) * b).astype(np.float32)


def random_matched(delta, seed: int = 0):
    """同 L2 norm 的高斯隨機方向（控制組 3）。"""
    import numpy as np
    rng = np.random.default_rng(seed)
    r = rng.standard_normal(delta.shape).astype(np.float32)
    return r * (float(np.linalg.norm(delta)) / float(np.linalg.norm(r)))


def chunk_positions(targets, offset: int, chunk_len: int, include_generated: bool,
                    prompt_len: int):
    """
    把「絕對 token 位置」轉成本次 forward 的區域索引。

    generate 帶 KV cache 時，prefill 一次 chunk_len=prompt_len，之後每步 chunk_len=1，
    因此必須靠 offset 累加自行記帳。include_generated=True 時，prompt 之後
    每個新 token 位置也注入。
    """
    sel = [p - offset for p in targets if offset <= p < offset + chunk_len]
    if include_generated:
        sel += [t for t in range(chunk_len)
                if offset + t >= prompt_len and (offset + t) not in targets]
    return sorted(set(s for s in sel if 0 <= s < chunk_len))


def resolve_targets(mode: str, site_a_idx: int, prompt_len: int):
    """--positions → (要注入的 prompt 絕對位置, 是否延伸到生成 token)。"""
    if mode == "mention":
        return [site_a_idx], False
    if mode == "from-mention":
        return list(range(site_a_idx, prompt_len)), True
    if mode == "all":
        return list(range(prompt_len)), True
    if mode == "last":
        return [prompt_len - 1], False
    raise SystemExit(f"[錯誤] 未知 --positions {mode!r}")


def clause_token_slice(prefix_len: int, seq_len: int):
    """
    續句 token 在 [prefix_len, seq_len) —— 其 logP 取自 logits 的
    [prefix_len-1, seq_len-1)（下一 token 預測錯位）。
    """
    return (prefix_len - 1, seq_len - 1), (prefix_len, seq_len)


def paired_bootstrap(diffs, n_boot: int = 10000, seed: int = 0):
    """逐題配對差的平均值＋95% bootstrap CI＋符號檢定 p（雙尾）。"""
    import numpy as np
    d = np.asarray(diffs, dtype=np.float64)
    d = d[~np.isnan(d)]
    if len(d) == 0:
        return dict(n=0, mean=float("nan"), lo=float("nan"), hi=float("nan"),
                    p_sign=float("nan"), n_pos=0)
    rng = np.random.default_rng(seed)
    boot = d[rng.integers(0, len(d), size=(n_boot, len(d)))].mean(axis=1)
    n_pos = int((d > 0).sum())
    n_eff = int((d != 0).sum())
    # 雙尾符號檢定（H0: P(正)=0.5），無 scipy 依賴
    k = min(n_pos, n_eff - n_pos)
    tail = sum(math.comb(n_eff, j) for j in range(0, k + 1)) / (2 ** n_eff) if n_eff else 1.0
    return dict(n=int(len(d)), mean=float(d.mean()),
                lo=float(np.percentile(boot, 2.5)), hi=float(np.percentile(boot, 97.5)),
                p_sign=float(min(1.0, 2 * tail)), n_pos=n_pos)


def lexicon_score(text: str):
    """生成文字的地緣／日常詞表命中差（每 100 字）。"""
    n = max(len(text), 1)
    g = sum(text.count(w) for w in GEO_LEX)
    l = sum(text.count(w) for w in LIFE_LEX)
    return 100.0 * (g - l) / n, g, l


# ======================================================================
# 前綴建構
# ======================================================================

def build_prefixes(rows, per_frame: int, entity_aliases=TAIWAN_ALIASES):
    """
    取中文 baseline、台灣的句子，截到「台灣」為止當生成前綴。
    每框架取前 per_frame 句（依 pair_id 排序，可重現）。
    """
    # 保留原始列序：sent_id 必須與 extract_activations.py 對同一份 CSV 產生的一致，
    # 才能與 activations parquet／gate.gated.csv join。
    cand = [(i, r) for i, r in enumerate(rows)
            if r.get("lang") == "zh" and r.get("cell_type") == "baseline"
            and r.get("entity") in entity_aliases]
    out, by_frame = [], {}
    for i, r in sorted(cand, key=lambda x: (x[1].get("frame", ""), x[1].get("pair_id", ""))):
        spans = find_mention_char_spans(r["text"], r["mention"])
        if len(spans) != 1:
            continue
        end = spans[0][1]
        frame = r.get("frame", "")
        if by_frame.get(frame, 0) >= per_frame:
            continue
        by_frame[frame] = by_frame.get(frame, 0) + 1
        out.append(dict(sent_id=assign_sent_id(r, i), pair_id=r.get("pair_id", ""),
                        frame=frame, mention=r["mention"],
                        prefix=r["text"][:end], full_text=r["text"]))
    return out


# ======================================================================
# torch 層：注入 hook
# ======================================================================

def get_decoder_layers(net):
    """
    取 decoder layer 的 ModuleList。Qwen2 走 `model.layers`；
    Gemma-3 的 ForConditionalGeneration 包一層 language_model。
    """
    import torch.nn as nn
    paths = ["model.layers", "model.language_model.layers", "language_model.model.layers",
             "model.model.layers", "transformer.h", "layers"]
    for path in paths:
        node = net
        for attr in path.split("."):
            node = getattr(node, attr, None)
            if node is None:
                break
        if isinstance(node, nn.ModuleList) and len(node) > 0:
            return node, path
    raise SystemExit(f"[錯誤] 找不到 decoder layers（{type(net).__name__}）")


def check_layer_range(layer: int, n_layers: int) -> None:
    """
    層位合法性。特別注意最後一層：HF 在迴圈結束後才把 final norm 套上去再收進
    `all_hidden_states`，因此 `hidden_states[n_layers]` 是**過了 final norm 的**，
    與 `layers[n_layers-1]` 的輸出不相等（已實測）。中間層則完全相等。
    Qwen L20/28、Gemma L32/48 都是中間層，不受影響。
    """
    if not (1 <= layer <= n_layers):
        raise SystemExit(f"[錯誤] layer={layer} 超出範圍（1..{n_layers}）")
    if layer == n_layers:
        print(f"  [警告] layer={layer} 為最後一層：hidden_states[{layer}] 已過 final norm，"
              f"與 layers[{layer-1}] 的輸出不相等 → 注入點與抽取點無法對齊。")


class Injector:
    """
    在 `layers[L-1]` 的輸出殘差流上加 α·v。

    hook 掛在 layers[layer-1]（＝ hidden_states[layer] 的產生處，見檔頭）。
    自行記帳 offset 以支援 generate 的 prefill(T=n) → decode(T=1) 切塊。
    """

    def __init__(self, layer_module, vector, alpha: float, targets, include_generated: bool,
                 prompt_len: int):
        self.mod = layer_module
        self.vec = vector
        self.alpha = float(alpha)
        self.targets = list(targets)
        self.include_generated = include_generated
        self.prompt_len = prompt_len
        self.offset = 0
        self.n_applied = 0
        self.handle = None

    def _hook(self, module, args, output):
        is_tuple = isinstance(output, tuple)
        h = output[0] if is_tuple else output
        chunk = h.shape[1]
        if self.alpha != 0.0:
            sel = chunk_positions(self.targets, self.offset, chunk,
                                  self.include_generated, self.prompt_len)
            if sel:
                v = self.vec.to(device=h.device, dtype=h.dtype)
                h[:, sel, :] = h[:, sel, :] + self.alpha * v
                self.n_applied += len(sel)
        self.offset += chunk
        return (h,) + output[1:] if is_tuple else h

    def __enter__(self):
        self.offset = 0
        self.n_applied = 0
        self.handle = self.mod.register_forward_hook(self._hook, with_kwargs=False)
        return self

    def __exit__(self, *exc):
        if self.handle is not None:
            self.handle.remove()
            self.handle = None
        return False


# ======================================================================
# 打分：teacher-forced 續句 logP
# ======================================================================

def score_clauses(net, tok, prefix_ids, clause_id_list, layer_module, vector, alpha,
                  targets, include_generated, device):
    """
    一個前綴 × 一批續句，回傳每句 (sum_logp, mean_logp, n_tok)。
    右側 padding；pad 位置在因果注意力下不影響前面 token，且被 mask 排除。
    """
    import torch

    pfx = len(prefix_ids)
    seqs = [list(prefix_ids) + list(c) for c in clause_id_list]
    maxlen = max(len(s) for s in seqs)
    pad = tok.pad_token_id if tok.pad_token_id is not None else (tok.eos_token_id or 0)
    ids = torch.full((len(seqs), maxlen), pad, dtype=torch.long)
    mask = torch.zeros((len(seqs), maxlen), dtype=torch.long)
    for i, s in enumerate(seqs):
        ids[i, :len(s)] = torch.tensor(s, dtype=torch.long)
        mask[i, :len(s)] = 1
    ids, mask = ids.to(device), mask.to(device)

    with Injector(layer_module, vector, alpha, targets, include_generated, pfx):
        with torch.no_grad():
            logits = net(input_ids=ids, attention_mask=mask).logits.float()
    logprobs = torch.log_softmax(logits, dim=-1)

    out = []
    for i, s in enumerate(seqs):
        (l0, l1), (t0, t1) = clause_token_slice(pfx, len(s))
        tgt = ids[i, t0:t1]
        lp = logprobs[i, l0:l1, :].gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
        out.append((float(lp.sum()), float(lp.mean()), int(t1 - t0)))
    return out


def generate_steered(net, tok, prefix_ids, layer_module, vector, alpha, targets,
                     include_generated, device, max_new_tokens: int):
    """貪婪生成（決定性），同時回傳逐 token 平均 logP 當流暢度護欄。"""
    import torch

    ids = torch.tensor([list(prefix_ids)], dtype=torch.long, device=device)
    mask = torch.ones_like(ids)
    with Injector(layer_module, vector, alpha, targets, include_generated, ids.shape[1]) as inj:
        with torch.no_grad():
            gen = net.generate(input_ids=ids, attention_mask=mask,
                               max_new_tokens=max_new_tokens, do_sample=False,
                               return_dict_in_generate=True, output_scores=True,
                               pad_token_id=tok.pad_token_id or tok.eos_token_id)
    new_ids = gen.sequences[0, ids.shape[1]:]
    lps = []
    for step, score in enumerate(gen.scores):
        if step >= len(new_ids):
            break
        lps.append(float(torch.log_softmax(score[0].float(), dim=-1)[new_ids[step]]))
    text = tok.decode(new_ids, skip_special_tokens=True)
    mean_lp = sum(lps) / len(lps) if lps else float("nan")
    return text, mean_lp, inj.n_applied


# ======================================================================
# --verify-hook：證明注入點 == 抽取點
# ======================================================================

def verify_hook(model: str, layer: int) -> None:
    import torch

    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"===== verify-hook：{model} @ L{layer} =====")
    tok = AutoTokenizer.from_pretrained(model)
    net = AutoModelForCausalLM.from_pretrained(model, torch_dtype=torch.bfloat16,
                                               device_map="auto", output_hidden_states=True)
    net.eval()
    dev = next(net.parameters()).device
    layers, path = get_decoder_layers(net)
    print(f"  decoder layers 路徑：{path}（共 {len(layers)} 層）")
    check_layer_range(layer, len(layers))

    grabbed = {}

    def cap(module, args, output):
        grabbed["h"] = (output[0] if isinstance(output, tuple) else output).detach()

    h = layers[layer - 1].register_forward_hook(cap)
    enc = tok("若從地質學的角度來看，台灣位於板塊交界處。", return_tensors="pt",
              add_special_tokens=True)
    with torch.no_grad():
        out = net(**{k: v.to(dev) for k, v in enc.items()}, output_hidden_states=True)
    h.remove()

    ref = out.hidden_states[layer].detach()
    got = grabbed["h"]
    same_shape = tuple(ref.shape) == tuple(got.shape)
    max_abs = float((ref.float() - got.float()).abs().max()) if same_shape else float("nan")
    ok = same_shape and max_abs == 0.0
    print(f"  hidden_states[{layer}] shape={tuple(ref.shape)}；hook 攔截 shape={tuple(got.shape)}")
    print(f"  逐元素最大絕對差 = {max_abs}")
    if ok:
        print(f"VERIFY-HOOK PASS ✅ layers[{layer-1}] 的輸出即 extract_activations.py "
              f"取用的 hidden_states[{layer}]；注入點與抽取點對齊。")
    else:
        raise SystemExit("VERIFY-HOOK FAIL ❌ 注入點與抽取點不一致——"
                         "請勿據此跑主實驗（先檢查 layer off-by-one 與 layers 路徑）。")


# ======================================================================
# 主實驗
# ======================================================================

def run(args, layer: int) -> None:
    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    meta, mat = read_activation_table(args.activations)
    rows = load_pairs_csv(args.pairs_csv)
    prefixes = build_prefixes(rows, args.per_frame)
    if args.limit:
        prefixes = prefixes[:args.limit]
    frames = sorted({p["frame"] for p in prefixes})
    print(f"  前綴 {len(prefixes)} 條，涵蓋框架 {frames}")

    # --- Δ 向量（LOFO：測試框架 f 的前綴，用非 f 的句子估 Δ）------------
    def delta_for(frame, entities, seed=None):
        excl = (frame,) if (args.lofo and frame) else ()
        return compute_delta(meta, mat, entities=entities, site=args.site, exclude_frames=excl)

    deltas, dinfo = {}, {}
    for f in frames:
        deltas[("delta_lang", f)], dinfo[("delta_lang", f)] = delta_for(f, TAIWAN_ALIASES)
        deltas[("delta_ctrl", f)], dinfo[("delta_ctrl", f)] = delta_for(f, CONTROL_ENTITIES)
        deltas[("random", f)] = random_matched(deltas[("delta_lang", f)], seed=args.seed)
        dinfo[("random", f)] = dict(dinfo[("delta_lang", f)], note="matched-norm gaussian")
        deltas[("delta_tw_resid", f)] = orthogonalize(deltas[("delta_lang", f)],
                                                      deltas[("delta_ctrl", f)])
        dinfo[("delta_tw_resid", f)] = dict(
            dinfo[("delta_lang", f)], note="Δ_台灣 ⊥ Δ_控制實體（台灣特異成分）",
            delta_norm=float(np.linalg.norm(deltas[("delta_tw_resid", f)])))
    g = dinfo[("delta_lang", frames[0])]
    ratio = g["delta_norm"] / max(g["mean_h_norm"], 1e-9)
    print(f"  Δ_lang（示例，排除 {frames[0]}）：n_en={g['n_en']} n_zh={g['n_zh']} "
          f"‖Δ‖={g['delta_norm']:.2f}  平均‖h‖={g['mean_h_norm']:.2f}（比值 {ratio:.3f}）")
    if ratio < 0.2:
        print(f"  [提醒] ‖Δ‖/‖h‖={ratio:.3f} 偏低（Gemma 因 embed 的 √d 縮放，殘差流量級大，"
              f"實測約 0.12；Qwen 約 0.46）→ 同一個 α 在此模型是**較弱的推力**。"
              f"若 α=±1 看似無效應，請先用更寬的 --alphas（如 -8 -4 -2 0 2 4 8）確認是"
              f"劑量不足而非沒有效應；跨模型請比較劑量反應曲線，不要比單一 α 的點值。")

    tok = AutoTokenizer.from_pretrained(args.model)
    if not getattr(tok, "is_fast", False):
        raise SystemExit("[錯誤] 需 fast tokenizer（offset_mapping 定位 mention）。")
    tok.padding_side = "right"

    # 每條前綴的 site A 索引（與抽取同一套 plan_sites）
    for p in prefixes:
        sp = plan_sites(tok, p["prefix"], p["mention"], p["sent_id"], "zh")
        p["site_a_idx"] = sp.site_a_idx
        p["prefix_ids"] = tok(p["prefix"], add_special_tokens=True)["input_ids"]
        if sp.site_a_idx < 0:
            print(f"  [警告] {p['sent_id']}：mention 定位失敗，已跳過。")
    prefixes = [p for p in prefixes if p["site_a_idx"] >= 0]

    if args.dry_run:
        outp = Path(f"{args.out}.prefixes.csv")
        outp.parent.mkdir(parents=True, exist_ok=True)
        with outp.open("w", newline="", encoding="utf-8-sig") as fh:
            w = csv.DictWriter(fh, fieldnames=["sent_id", "pair_id", "frame", "mention",
                                               "site_a_idx", "n_prefix_tok", "prefix"])
            w.writeheader()
            for p in prefixes:
                w.writerow(dict(sent_id=p["sent_id"], pair_id=p["pair_id"], frame=p["frame"],
                                mention=p["mention"], site_a_idx=p["site_a_idx"],
                                n_prefix_tok=len(p["prefix_ids"]), prefix=p["prefix"]))
        Path(f"{args.out}.delta_info.json").write_text(
            json.dumps({f"{k[0]}|excl={k[1]}": v for k, v in dinfo.items()},
                       ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"  [dry-run] 未載入模型。前綴 -> {outp}；Δ 統計 -> {args.out}.delta_info.json")
        return

    print(f"  載入 {args.model}（bf16, device_map=auto）…")
    net = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16,
                                               device_map="auto")
    net.eval()
    dev = next(net.parameters()).device
    layers, path = get_decoder_layers(net)
    check_layer_range(layer, len(layers))
    layer_module = layers[layer - 1]
    d_delta = int(mat.shape[1])
    cfg = net.config
    d_model = int(getattr(cfg, "hidden_size", None)
                  or getattr(getattr(cfg, "text_config", None), "hidden_size", 0) or 0)
    if d_model and d_delta != d_model:
        raise SystemExit(f"[錯誤] Δ 維度 {d_delta} ≠ 模型 d_model {d_model}："
                         f"activations parquet 與 --model 不是同一個模型。")
    print(f"  注入點：{path}[{layer-1}] 的輸出（＝ hidden_states[{layer}]，"
          f"抽取時所用之同一張量）")

    clause_ids = {
        "geo": [tok(c, add_special_tokens=False)["input_ids"] for c in GEO_CLAUSES],
        "life": [tok(c, add_special_tokens=False)["input_ids"] for c in LIFE_CLAUSES],
    }
    conds = ["delta_lang", "delta_ctrl", "delta_tw_resid", "random"]
    alphas = args.alphas

    score_rows, gen_rows = [], []
    for pi, p in enumerate(prefixes, 1):
        targets, incl_gen = resolve_targets(args.positions, p["site_a_idx"],
                                            len(p["prefix_ids"]))
        print(f"  [{pi}/{len(prefixes)}] {p['sent_id']}（{p['frame']}）"
              f" site_a_idx={p['site_a_idx']} / {len(p['prefix_ids'])} tok")
        for cond in conds:
            vec = torch.from_numpy(np.asarray(deltas[(cond, p["frame"])]))
            for alpha in alphas:
                if alpha == 0.0 and cond != conds[0]:
                    continue                   # α=0 時各條件等價（＝未注入），只算一次
                for kind, cl in clause_ids.items():
                    res = score_clauses(net, tok, p["prefix_ids"], cl, layer_module, vec,
                                        alpha, targets, incl_gen, dev)
                    for j, (s, m, n) in enumerate(res):
                        score_rows.append(dict(
                            sent_id=p["sent_id"], pair_id=p["pair_id"], frame=p["frame"],
                            condition=("baseline" if alpha == 0.0 else cond),
                            alpha=alpha, clause_kind=kind, clause_idx=j,
                            clause=(GEO_CLAUSES if kind == "geo" else LIFE_CLAUSES)[j],
                            sum_logp=s, mean_logp=m, n_clause_tok=n,
                            model=args.model, layer=layer, site=args.site,
                            positions=args.positions))
                if args.max_new_tokens > 0 and (cond == "delta_lang" or alpha in (1.0, -1.0)):
                    text, mlp, n_app = generate_steered(
                        net, tok, p["prefix_ids"], layer_module, vec, alpha, targets,
                        incl_gen, dev, args.max_new_tokens)
                    lex, ngeo, nlife = lexicon_score(text)
                    gen_rows.append(dict(
                        sent_id=p["sent_id"], frame=p["frame"],
                        condition=("baseline" if alpha == 0.0 else cond), alpha=alpha,
                        prefix=p["prefix"], generation=text,
                        mean_logp_per_tok=mlp, lex_geo_minus_life_per100=lex,
                        n_geo_hits=ngeo, n_life_hits=nlife, n_positions_injected=n_app))

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    write_csv(Path(f"{args.out}.scores.csv"), score_rows)
    if gen_rows:
        write_csv(Path(f"{args.out}.generations.csv"), gen_rows)
    report(score_rows, gen_rows, dinfo, args, layer)


def write_csv(path: Path, rows) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"  -> {path}（{len(rows)} 列）")


# ======================================================================
# 報告
# ======================================================================

def aggregate_S(score_rows):
    """(sent_id, condition, alpha) → S = mean logP(geo) − mean logP(life)。"""
    acc = {}
    for r in score_rows:
        key = (r["sent_id"], r["condition"], r["alpha"])
        a = acc.setdefault(key, {"geo": [], "life": []})
        a[r["clause_kind"]].append(r["mean_logp"])
    return {k: (sum(v["geo"]) / len(v["geo"]) - sum(v["life"]) / len(v["life"]))
            for k, v in acc.items() if v["geo"] and v["life"]}


def report(score_rows, gen_rows, dinfo, args, layer: int) -> None:
    import numpy as np

    S = aggregate_S(score_rows)
    sids = sorted({k[0] for k in S})
    base = {s: S.get((s, "baseline", 0.0)) for s in sids}

    md = ["# Δ_lang activation steering：因果驗證報告", "",
          f"- 模型：`{args.model}`　注入層：**L{layer}**"
          f"（hook `layers[{layer-1}]` 的輸出 ＝ 抽取時的 `hidden_states[{layer}]`）",
          f"- 注入位置：`{args.positions}`（site {args.site}）；LOFO：{'開' if args.lofo else '關'}",
          f"- 前綴（中文語境）：{len(sids)} 條；探針續句：地緣 {len(GEO_CLAUSES)} ／日常 {len(LIFE_CLAUSES)}",
          "",
          "估計量：`S_i(α) = mean logP(地緣續句) − mean logP(日常續句)`，",
          "主結果為逐題配對的 **`S_i(α) − S_i(0)`**（組內相減抵消續句先天偏誤）。",
          "假說：Δ_lang = mean(en) − mean(zh)，注入 +Δ 應使 S **上升**、−Δ 應**下降**。",
          "", "## 主結果與控制組（配對 bootstrap 95% CI，n＝前綴數）", "",
          "| 條件 | α | ΔS 平均 | 95% CI | 正向題數/總數 | 符號檢定 p |",
          "|---|---|---|---|---|---|"]

    conds = ["delta_lang", "delta_ctrl", "delta_tw_resid", "random"]
    table = {}
    for cond in conds:
        for alpha in sorted({k[2] for k in S if k[1] == cond}):
            diffs = [S[(s, cond, alpha)] - base[s] for s in sids
                     if (s, cond, alpha) in S and base.get(s) is not None]
            st = paired_bootstrap(diffs, seed=args.seed)
            table[(cond, alpha)] = st
            md.append(f"| {cond} | {alpha:+.1f} | {st['mean']:+.4f} | "
                      f"[{st['lo']:+.4f}, {st['hi']:+.4f}] | {st['n_pos']}/{st['n']} | "
                      f"{st['p_sign']:.4g} |")

    md += ["", "## 判讀準則（預先設定）", "",
           "1. **劑量反應**：`delta_lang` 的 ΔS 隨 α 單調遞增且 α=0 附近過原點；",
           "2. **反向**：α<0 的 ΔS 顯著為負 —— 方向可逆才排除「注入任何東西都會亂」；",
           "3. **隨機方向**：`random` 在同 ‖Δ‖ 下 CI 應涵蓋 0；",
           "4. **控制實體**：`delta_ctrl`（日本／冰島的 en−zh）效應應顯著小於 `delta_lang`，",
           "   否則測到的只是「語言方向」而非「台灣的語言方向」；",
           "5. **台灣特異成分**：`delta_tw_resid`（Δ_台灣 ⊥ Δ_控制實體）若仍有顯著效應，",
           "   即為 H4 DiD 的向量版證據——效應不只是泛語言方向所致；",
           "6. **流暢度**：逐 token 平均 logP 不得隨 |α| 崩塌（見下表）。"]

    pos = table.get(("delta_lang", 1.0))
    neg = table.get(("delta_lang", -1.0))
    rnd = table.get(("random", 1.0))
    ctl = table.get(("delta_ctrl", 1.0))
    res = table.get(("delta_tw_resid", 1.0))
    # 判讀依「抗擾動強度」排序：單看 α=+1 顯著並不夠力——夠大的擾動往任何方向
    # 都可能推動 geo/日常的平衡；真正的證據是「方向可逆」「隨機方向無效」
    # 「扣掉泛語言方向後仍有效」這三條對照。
    verdict = []
    if neg and pos:
        verdict.append(f"**① 方向可逆**：α=−1 的 ΔS={neg['mean']:+.4f} vs α=+1 的 "
                       f"{pos['mean']:+.4f} → "
                       f"{'反號 ✅（不是「注入任何東西都會亂」）' if neg['mean'] * pos['mean'] < 0 else '未反號 ❌'}")
    if res:
        verdict.append(f"**② 台灣特異成分**（Δ_台灣 ⊥ Δ_控制實體）α=+1：ΔS={res['mean']:+.4f}，CI "
                       f"{'不含 0 ✅（效應非泛語言方向所致）' if res['lo'] > 0 or res['hi'] < 0 else '含 0 ⚠️（無法排除泛語言方向解釋）'}")
    if rnd:
        verdict.append(f"**③ 隨機方向**（同 ‖Δ‖）α=+1：ΔS={rnd['mean']:+.4f}，"
                       f"{'CI 含 0 ✅（不是量級假象）' if rnd['lo'] <= 0 <= rnd['hi'] else 'CI 不含 0 ⚠️（此量級的任意擾動即有效應，主結果須改看 ①②）'}")
    if pos:
        verdict.append(f"④ 主效應 α=+1：ΔS={pos['mean']:+.4f}，CI "
                       f"{'不含 0' if pos['lo'] > 0 or pos['hi'] < 0 else '含 0 ❌'}"
                       f"（單獨看不足以定論，須與 ①②③ 合讀）")
    if ctl and pos:
        verdict.append(f"⑤ 控制實體 Δ（日／冰）α=+1：ΔS={ctl['mean']:+.4f}"
                       f"（台灣 Δ 的 {abs(ctl['mean']) / max(abs(pos['mean']), 1e-9):.0%}，描述性）")
    md += ["", "## 一句話判讀（依證據強度排序）", "", "- " + "\n- ".join(verdict) if verdict else ""]

    if gen_rows:
        md += ["", "## 流暢度護欄與生成傾向（貪婪解碼，質性佐證）", "",
               "| 條件 | α | 逐 token 平均 logP | 詞表(地緣−日常)/100字 |", "|---|---|---|---|"]
        agg = {}
        for r in gen_rows:
            a = agg.setdefault((r["condition"], r["alpha"]), [[], []])
            a[0].append(r["mean_logp_per_tok"])
            a[1].append(r["lex_geo_minus_life_per100"])
        for (cond, alpha) in sorted(agg, key=lambda x: (x[0], x[1])):
            lp, lx = agg[(cond, alpha)]
            md.append(f"| {cond} | {alpha:+.1f} | {np.mean(lp):.3f} | {np.mean(lx):+.3f} |")
        md.append("\n生成全文見 `*.generations.csv`；建議抽樣做 Layer 1–4 標註，"
                  "與 §10 的框架漂移矩陣對照。")

    g = list(dinfo.values())[0]
    md += ["", "## Δ 向量統計", "",
           f"- 樣本數：en {g['n_en']} ／ zh {g['n_zh']}（LOFO 逐框架另計）",
           f"- ‖Δ_lang‖ = {g['delta_norm']:.3f}；平均 ‖h‖ = {g['mean_h_norm']:.3f}"
           f"（比值 {g['delta_norm'] / max(g['mean_h_norm'], 1e-9):.3f}）",
           f"- α=1 即「把該位置的殘差流平移 en 與 zh 的整組均值差」。", "",
           "## 已知限制", "",
           "- 前綴與續句在 tokenizer 上分開編碼再串接，邊界處可能與整句自然切分略異；",
           "  各條件共用同一組 token id，差異在 delta-of-delta 中抵消。",
           "- 探針續句為人工撰寫，覆蓋面有限；生成側的詞表計分僅為粗略佐證，",
           "  正式結論應以 `*.generations.csv` 的人工／LLM 標註為準。",
           "- 未套 chat template（與抽取一致，NLA 以預訓練式文本訓練）。"]

    rep = Path(f"{args.out}.report.md")
    rep.write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"  -> {rep}")


# ======================================================================
# Self-test（numpy only，無 torch／無網路）
# ======================================================================

def self_test() -> None:
    import numpy as np

    d = 8
    rng = np.random.default_rng(0)
    true_delta = rng.standard_normal(d).astype(np.float32)
    meta, vecs = [], []
    for frame in ("GEO", "ECON", "CUL"):
        for i in range(4):
            base = rng.standard_normal(d).astype(np.float32)
            for lang in ("zh", "en"):
                for site in ("A", "B"):
                    meta.append(dict(lang=lang, site=site, entity="Taiwan",
                                     cell_type="baseline", frame=frame))
                    vecs.append(base + (true_delta if lang == "en" else 0))
    # 控制實體：與台灣 Δ 部分平行（0.25 倍）＋ 一個正交成分（模擬實測 cos≈0.64）
    other = rng.standard_normal(d).astype(np.float32)
    other -= (float(other @ true_delta) / float(true_delta @ true_delta)) * true_delta
    ctrl_delta = (0.25 * true_delta + 0.2 * other).astype(np.float32)
    for i in range(4):
        base = rng.standard_normal(d).astype(np.float32)
        for lang in ("zh", "en"):
            meta.append(dict(lang=lang, site="A", entity="Japan",
                             cell_type="baseline", frame="GEO"))
            vecs.append(base + (ctrl_delta if lang == "en" else 0))
    mat = np.stack(vecs)

    delta, info = compute_delta(meta, mat, entities=TAIWAN_ALIASES, site="A")
    assert np.allclose(delta, true_delta, atol=1e-4), delta
    assert info["n_en"] == 12 and info["n_zh"] == 12, info

    # LOFO：排除 GEO 後樣本數下降，方向不變
    d2, i2 = compute_delta(meta, mat, entities=TAIWAN_ALIASES, site="A", exclude_frames=("GEO",))
    assert i2["n_en"] == 8 and np.allclose(d2, true_delta, atol=1e-4)

    # 控制實體 Δ 幅度應明顯較小
    dc, _ = compute_delta(meta, mat, entities=CONTROL_ENTITIES, site="A")
    assert np.linalg.norm(dc) < 0.5 * np.linalg.norm(delta)

    # 台灣特異殘差：與控制實體 Δ 正交，且保留原方向的大部分成分
    resid = orthogonalize(delta, dc)
    assert abs(float(resid @ dc)) < 1e-2 * float(np.linalg.norm(dc)), float(resid @ dc)
    assert float(resid @ delta) > 0
    # 若兩者完全平行，殘差應為 0
    assert np.allclose(orthogonalize(delta, 2.0 * delta), np.zeros(d), atol=1e-4)

    # 隨機控制：同 norm、方向近乎正交
    r = random_matched(delta, seed=1)
    assert abs(np.linalg.norm(r) - np.linalg.norm(delta)) < 1e-3
    assert abs(float(r @ delta) / (np.linalg.norm(r) * np.linalg.norm(delta))) < 0.9

    # 位置解析：三種 --positions
    t, ig = resolve_targets("mention", 5, 10)
    assert t == [5] and ig is False
    t, ig = resolve_targets("from-mention", 5, 10)
    assert t == [5, 6, 7, 8, 9] and ig is True
    assert resolve_targets("last", 5, 10)[0] == [9]

    # offset 記帳：prefill(T=10) → decode(T=1)×3
    assert chunk_positions([5], 0, 10, False, 10) == [5]
    assert chunk_positions([5], 10, 1, False, 10) == []          # 不延伸到生成
    assert chunk_positions([5], 10, 1, True, 10) == [0]          # 延伸到生成
    assert chunk_positions([5], 11, 1, True, 10) == [0]
    assert chunk_positions(list(range(5, 10)), 0, 10, True, 10) == [5, 6, 7, 8, 9]

    # 續句 token 切片（下一 token 預測錯位）
    (l0, l1), (t0, t1) = clause_token_slice(7, 15)
    assert (l0, l1, t0, t1) == (6, 14, 7, 15) and (l1 - l0) == (t1 - t0)

    # 估計量與統計：植入正效應
    rows = []
    for i in range(20):
        for cond, alpha, shift in (("baseline", 0.0, 0.0), ("delta_lang", 1.0, 0.5),
                                   ("delta_lang", -1.0, -0.5), ("random", 1.0, 0.0)):
            for kind in ("geo", "life"):
                for j in range(3):
                    val = (shift if kind == "geo" else 0.0) + float(rng.normal(0, 0.05))
                    rows.append(dict(sent_id=f"s{i}", condition=cond, alpha=alpha,
                                     clause_kind=kind, mean_logp=val))
    S = aggregate_S(rows)
    base = {f"s{i}": S[(f"s{i}", "baseline", 0.0)] for i in range(20)}
    up = paired_bootstrap([S[(f"s{i}", "delta_lang", 1.0)] - base[f"s{i}"] for i in range(20)])
    dn = paired_bootstrap([S[(f"s{i}", "delta_lang", -1.0)] - base[f"s{i}"] for i in range(20)])
    nul = paired_bootstrap([S[(f"s{i}", "random", 1.0)] - base[f"s{i}"] for i in range(20)])
    assert up["lo"] > 0 and abs(up["mean"] - 0.5) < 0.1, up
    assert dn["hi"] < 0, dn                                   # 反向注入必須反向
    assert nul["lo"] <= 0 <= nul["hi"], nul                   # 隨機方向無效應
    assert up["p_sign"] < 0.01 and nul["p_sign"] > 0.05, (up, nul)

    # 詞表計分
    lex, ng, nl = lexicon_score("台灣的主權與兩岸關係" * 3)
    assert ng > 0 and nl == 0 and lex > 0

    print("SELF-TEST PASS ✅（Δ 估計、LOFO、控制向量、注入位置記帳、"
          "續句切片、delta-of-delta 統計皆正確）")
    print("  註：hook 是否真的對齊抽取點，需上機跑 `--verify-hook`（需 GPU）。")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--activations", type=Path, help="extract_activations.py 之 parquet")
    ap.add_argument("--pairs-csv", type=Path, help="語料 CSV（取中文前綴）")
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--nla-meta", type=Path, default=None, help="讀層位（與抽取同一份）")
    ap.add_argument("--layer", type=int, default=None, help="明確指定層（覆蓋 meta）")
    ap.add_argument("--site", default="A", choices=["A", "B"], help="用哪個 site 的向量算 Δ")
    ap.add_argument("--positions", default="mention",
                    choices=["mention", "from-mention", "all", "last"],
                    help="注入哪些 token 位置（預設只注入提及詞末 subtoken＝Site A）")
    ap.add_argument("--alphas", type=float, nargs="+", default=[-2.0, -1.0, 0.0, 1.0, 2.0],
                    help="注入強度掃描（α=1 ＝ 平移一整個 en−zh 均值差）")
    ap.add_argument("--per-frame", type=int, default=3, help="每框架取幾條測試前綴")
    ap.add_argument("--limit", type=int, default=0, help="只跑前 N 條前綴（除錯用）")
    ap.add_argument("--max-new-tokens", type=int, default=64, help="貪婪生成長度（0＝不生成）")
    ap.add_argument("--lofo", dest="lofo", action="store_true", default=True,
                    help="Δ 估計排除測試前綴所屬框架（預設開）")
    ap.add_argument("--no-lofo", dest="lofo", action="store_false")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=Path("results/steer"))
    ap.add_argument("--dry-run", action="store_true", help="只算 Δ 與前綴、不載模型")
    ap.add_argument("--verify-hook", action="store_true",
                    help="驗證 hook 之注入點 == 抽取點（上機必跑）")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()

    if args.self_test:
        self_test()
        return

    layer = resolve_layer(args.model, args.nla_meta, args.layer)
    if args.verify_hook:
        verify_hook(args.model, layer)
        return
    if not (args.activations and args.pairs_csv):
        ap.error("需 --activations 與 --pairs-csv（或用 --self-test / --verify-hook）")
    print(f"===== steering {args.model} @ L{layer}（注入 layers[{layer-1}] 輸出）=====")
    run(args, layer)


if __name__ == "__main__":
    main()
