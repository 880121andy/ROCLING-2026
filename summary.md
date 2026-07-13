# Tokenizer 驗證報告（RQ1）

## Qwen/Qwen2.5-7B-Instruct

### 孤立形式切分

| 形式 | n_subtokens | pieces |
|---|---|---|
| '台灣' | 1 | 'åı°çģ£' |
| '臺灣' | 1 | 'èĩºçģ£' |
| '台湾' | 1 | 'åı°æ¹¾' |
| 'Taiwan' | 2 | 'Tai' | 'wan' |
| ' Taiwan' | 1 | 'ĠTaiwan' |
| 'Japan' | 1 | 'Japan' |
| ' Japan' | 1 | 'ĠJapan' |
| '日本' | 1 | 'æĹ¥æľ¬' |
| '冰島' | 2 | 'åĨ°' | 'å³¶' |
| 'Iceland' | 2 | 'I' | 'celand' |
| ' Iceland' | 1 | 'ĠIceland' |

### 語境內診斷摘要

- 句數：32；帶警告：0
- mention subtoken 數（lang, n）：[('en', 1), ('zh', 1)]
- 樸素 sublist 搜尋與 offset 法不一致：16 句（不一致即為樸素法之失敗案例，抽取管線務必採 offset 法）
- 前導 < 10 tokens：19 句 → GEO-01/zh(9), POL-INT-01/zh(9), POL-INT-01/en(9), POL-INT-02/zh(9), POL-INT-02/en(7), POL-DOM-01/zh(8), POL-DOM-01/en(9), POL-DOM-02/zh(9), POL-DOM-02/en(6), ECON-01/zh(9), ECON-01/en(7), ECON-02/en(9), CUL-01/en(9), HIST-02/zh(8), LIFE-01/zh(7), LIFE-01/en(6), LIFE-02/zh(9), LIFE-02/en(9), TRAV-02/en(8)
- 句對長度匹配（±20%）：12/16 通過 → 超標：ECON-01(gap=0.261), ECON-02(gap=0.222), LIFE-01(gap=0.219), POL-DOM-01(gap=0.226)

## unsloth/gemma-3-12b-it

### 孤立形式切分

| 形式 | n_subtokens | pieces |
|---|---|---|
| '台灣' | 1 | '台灣' |
| '臺灣' | 1 | '臺灣' |
| '台湾' | 1 | '台湾' |
| 'Taiwan' | 1 | 'Taiwan' |
| ' Taiwan' | 1 | '▁Taiwan' |
| 'Japan' | 1 | 'Japan' |
| ' Japan' | 1 | '▁Japan' |
| '日本' | 1 | '日本' |
| '冰島' | 2 | '冰' | '島' |
| 'Iceland' | 1 | 'Iceland' |
| ' Iceland' | 1 | '▁Iceland' |

### 語境內診斷摘要

- 句數：32；帶警告：0
- mention subtoken 數（lang, n）：[('en', 1), ('zh', 1)]
- 樸素 sublist 搜尋與 offset 法不一致：16 句（不一致即為樸素法之失敗案例，抽取管線務必採 offset 法）
- 前導 < 10 tokens：19 句 → GEO-01/zh(7), GEO-01/en(8), POL-INT-01/zh(9), POL-INT-01/en(9), POL-INT-02/zh(8), POL-INT-02/en(7), POL-DOM-01/zh(8), POL-DOM-01/en(9), POL-DOM-02/en(6), ECON-01/zh(8), ECON-01/en(7), ECON-02/en(9), CUL-01/zh(9), HIST-02/zh(8), LIFE-01/zh(7), LIFE-01/en(6), LIFE-02/zh(9), LIFE-02/en(9), TRAV-02/en(7)
- 句對長度匹配（±20%）：16/16 通過
