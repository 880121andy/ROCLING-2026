# RQ1 語料設計書：「台灣」跨語境內部表徵之最小對比句對語料
## Minimal-Pair Corpus Design for Probing the Internal Representation of "Taiwan" across English and Taiwan Mandarin Contexts (NLA-based)

版本：v0.1（設計稿，待審後生成完整語料）
方法框架：Natural Language Autoencoders（Fraser-Taliente et al. 2026）
目標模型：Qwen2.5-7B-Instruct（L20）／Gemma-3-12B-IT（L32），使用官方釋出之 AV/AR checkpoints

---

## 1. 研究問題與可檢驗假說

**RQ1**：同一命題內容下，「台灣」出現於英語語境 vs 台灣中文語境時，模型內部表徵（residual stream activation）的語意構造有何系統性差異？

以 AV 的自然語言描述（verbalization）作為表徵的可讀投影，預先登錄（pre-register）以下假說：

- **H1（框架漂移 frame drift）**：英語語境的 verbalization 更常出現刺激句未觸發之地緣政治框架（geopolitical intrusion）；中文語境更常維持刺激句原框架或漂向在地生活框架。
- **H2（中國錨定 China-anchoring）**：刺激句未提及中國時，英語語境的 verbalization 提及 China／cross-strait 的比率顯著較高。
- **H3（識解類型 construal type）**：英語語境下「台灣」更常被識解為 POLITY（政體），中文語境下更常為 PLACE／PEOPLE／日常生活場域。
- **H4（台灣特異性，DiD）**：上述效應在控制實體（日本、冰島）上顯著較弱，即為「台灣」特有而非跨語言的一般效應。
- **H5（語境 vs 詞形之分離，Design B）**：框架漂移主要由「語境語言」驅動，而非提及詞的「文字形式」（台灣 vs Taiwan）——或相反；兩種結果皆有理論意義。

---

## 2. 設計總覽（factorial structure）

| 因子 | 水準 | 角色 |
|---|---|---|
| 語境語言 Language | English／台灣中文（zh-TW） | 操弄變項 |
| 刺激框架 Stimulus frame | 8 種（§3） | 分層變項 |
| 實體 Entity | 台灣（目標）／日本、冰島（控制） | DiD 控制 |
| 提及文字形式 Mention script | 漢字／拉丁字母（僅 Design B） | 分離詞形與語境 |
| 抽取位置 Extraction site | 目標詞末 subtoken（Site A）／句末 token（Site B） | 測量位置 |

規模規劃（§7 詳述）：核心 192 句＋控制 120 句＋Design B 48 句 ≈ **360 句**，每句 2 個抽取位置、每向量取樣 5 則 AV 描述。

---

## 3. 刺激框架分層（stimulus frame inventory）

框架定義取徑 FrameNet 精神但為本研究自訂（domain-level scenes）。每框架之刺激句必須：只喚起該框架、不提及中國、不含明顯評價詞（核心語料一律中性陳述）。

| 代碼 | 框架 | 定義（觸發條件） | 排除 |
|---|---|---|---|
| GEO | Natural_geography | 板塊、氣候、地形、天然災害 | 涉主權之領土表述 |
| POL-INT | International_status | 國際組織、邦交、外交場合 | 直接點名中國、軍事 |
| POL-DOM | Domestic_politics | 選舉、政黨、投票、民主轉型 | 統獨議題 |
| ECON | Commerce_and_technology | 半導體、製造、供應鏈、貿易 | 出口管制等地緣化表述 |
| CUL | Cultural_practice | 飲食、廟會、節慶、宗教 | — |
| HIST | History | 殖民時期、大航海時代、建設史 | 1949 後兩岸史 |
| LIFE | Everyday_life | 超商、外送、健保、通勤 | — |
| TRAV | Tourism | 登山、公路、景點、旅遊路線 | — |

設計要點：POL-INT 刻意「去中國化」——句子談國際地位但**不**提及中國；如此 H2 的 China-anchoring 才是表徵層的「補入」而非表層複誦。

---

## 4. 句型構造規範（template grammar）

### 4.1 三段式結構（強制）

```
[前導子句 Lead-in] ＋ [目標子句：實體為主語] ＋ [延續子句 Continuation]
```

- **前導子句**：≥ 10 tokens（以目標模型 tokenizer 計），確立框架、**不得**提及實體。理由：自迴歸模型中，語境只能由左向右影響；若「台灣」居句首，語境語言操弄對 Site A 幾乎無因果作用域（causal scope）。
- **目標子句**：實體為主語，位居句中段；每句**僅一次**提及。
- **延續子句**：使 Site B（句末 token）承載完整命題之 gist。

### 4.2 最小對比之對等性規範

1. **命題對等**：中英句對為謹慎的翻譯等值句，經回譯檢核。
2. **自然度**：各自須是道地的台灣中文／英文，禁止翻譯腔；每句由母語者以 1–5 分評自然度，< 4 者改寫。
3. **長度匹配**：句對在目標模型 tokenizer 下之 token 數差異 ≤ ±20%。
4. **提及形式**：核心語料固定用「台灣」（robustness 子集另備「臺灣」）；英文固定 "Taiwan"。
5. **禁用型式**：實體不得作為專名內部修飾語（如「台灣海峽」"Taiwan Strait"、「台灣中部」"central Taiwan"）——該位置抽取的不是實體本身的表徵。

### 4.3 範例句對（每框架 2 例；完整版每框架 12 例）

**GEO-01**
- zh：從板塊構造的角度來看，台灣正好位於歐亞板塊與菲律賓海板塊的交界，因此地震十分頻繁。
- en：From the standpoint of plate tectonics, Taiwan sits right on the boundary between the Eurasian and Philippine Sea plates, which is why earthquakes are so frequent.

**GEO-02**
- zh：受到季風與地形交互作用的影響，台灣的東北部在冬季經常陰雨綿綿。
- en：Owing to the interaction of monsoon winds and local terrain, Taiwan tends to see long drizzly winters in its northeast.

**POL-INT-01**
- zh：在許多國際組織的正式場合裡，台灣的會員資格始終是各方交涉的焦點。
- en：In the formal settings of many international organizations, Taiwan's membership status has long been a focal point of negotiation.

**POL-INT-02**
- zh：在近年來的外交往來之中，台灣與若干邦交國之間的關係變化備受關注。
- en：In diplomatic exchanges of recent years, Taiwan's shifting ties with several of its formal allies have drawn close attention.

**POL-DOM-01**
- zh：每逢選舉年的冬天一到，台灣的街頭就掛滿競選旗幟，造勢晚會一場接著一場。
- en：When the winter of an election year arrives, Taiwan's streets fill with campaign flags, and rallies follow one after another.

**POL-DOM-02**
- zh：歷經數十年的政治轉型之後，台灣如今以高投票率與激烈的政黨競爭聞名。
- en：After decades of political transformation, Taiwan is now known for high voter turnout and fierce competition between parties.

**ECON-01**
- zh：在全球半導體供應鏈之中，台灣生產了絕大多數的先進製程晶片。
- en：Within the global semiconductor supply chain, Taiwan produces the vast majority of advanced-node chips.

**ECON-02**
- zh：在許多跨國企業的採購清單上，台灣的精密機械與自行車零件享有極高的評價。
- en：On the procurement lists of many multinational firms, Taiwan's precision machinery and bicycle components enjoy an excellent reputation.

**CUL-01**
- zh：對許多喜歡深夜覓食的人來說，台灣的夜市小吃是難以抗拒的誘惑，蚵仔煎更是必點。
- en：For anyone fond of late-night food hunts, Taiwan's night-market snacks are hard to resist, and the oyster omelet is a must-order.

**CUL-02**
- zh：每年春天媽祖遶境的季節一到，台灣就會湧現徒步進香九天八夜的人潮。
- en：Each spring when the Mazu pilgrimage season arrives, Taiwan sees crowds of devotees walking the route for nine days and eight nights.

**HIST-01**
- zh：在二十世紀初的殖民統治時期，台灣興建了縱貫南北的鐵路系統。
- en：During the colonial period of the early twentieth century, Taiwan built a railway system running the length of the island.

**HIST-02**
- zh：早在十七世紀的大航海時代，台灣就已經是東亞貿易網絡的重要節點。
- en：As early as the seventeenth-century age of sail, Taiwan was already a key node in East Asian trade networks.

**LIFE-01**
- zh：就日常生活的便利程度而言，台灣的超商密度名列世界前茅，半夜也能繳費、領包裹。
- en：In terms of everyday convenience, Taiwan ranks near the top worldwide in convenience-store density; you can pay bills and pick up parcels in the middle of the night.

**LIFE-02**
- zh：對外送平台的重度使用者來說，台灣的都會區幾乎能在三十分鐘內送達任何餐點。
- en：For heavy users of food-delivery apps, Taiwan's urban areas can get almost any meal to the door within thirty minutes.

**TRAV-01**
- zh：對喜愛山海景色的旅人來說，台灣東岸的蘇花公路是一段令人屏息的路線。
- en：For travelers who love mountain-and-sea scenery, Taiwan's Suhua Highway along the east coast is a breathtaking route.

**TRAV-02**
- zh：在許多登山愛好者的口袋名單上，台灣的玉山主峰是必須完成的目標之一。
- en：On many hikers' bucket lists, Taiwan's main peak of Yushan is a goal that must be completed.

---

## 5. 控制實體（DiD 設計）

觀察到「英中語境差異」時，須排除其為**跨語言的一般現象**（如英文語料整體更政治化）。故加入：

- **日本／Japan**：高國際能見度、無主權爭議 → 分離「亞洲實體」的一般效應。
- **冰島／Iceland**：低地緣政治顯著性 → 低鹽基線（geopolitically bland baseline）。

僅使用可跨實體遷移之框架（GEO、ECON、CUL、TRAV、LIFE），採**槽位模板**（slot template）確保跨實體最小對比。內容須對實體為真且合宜（felicitous），例如 GEO 模板三實體皆恰好位於板塊交界／火山帶：

```
zh：從板塊構造的角度來看，｛實體｝正好位於｛板塊甲｝與｛板塊乙｝的交界，因此｛地質現象｝十分頻繁。
en：From the standpoint of plate tectonics, {ENTITY} sits right on the boundary between {PLATE-A} and {PLATE-B}, which is why {PHENOMENON} is so frequent.
```

檢定量：語言 × 實體之交互作用（difference-in-differences）。

## 6. Design B：語境語言 × 提及文字形式（2×2 code-switching 子實驗）

分離兩個天然混淆的變項——**語境語言**與**目標詞形式**：

| | 提及＝台灣（漢字） | 提及＝Taiwan（拉丁） |
|---|---|---|
| 中文語境 | 基線 | 「…，Taiwan 正好位於…」 |
| 英文語境 | "… , 台灣 sits right on …" | 基線 |

- 範圍：GEO、ECON 兩框架 × 各 6 模板 × 4 格 ＝ 48 句。
- 機轉問題：Site A 的表徵差異來自 token embedding（詞形）還是先前語境（脈絡）？若 verbalization 的框架分布追隨語境語言而非文字形式，即為「語境主導」的直接證據。
- 注意：code-switched 句對 fineweb 訓練的 NLA 屬輕度 out-of-distribution，**所有結論須經 AR round-trip MSE 閘門**（§9 Layer 0），並於報告中呈現該格的 MSE 分布。

---

## 7. 規模與取樣

| 區塊 | 計算 | 句數 |
|---|---|---|
| 核心（台灣） | 8 框架 × 12 模板 × 2 語言 | 192 |
| 控制（日、冰） | 2 實體 × 5 框架 × 6 模板 × 2 語言 | 120 |
| Design B | 2 框架 × 6 模板 × 4 格 | 48 |
| **合計** | | **360** |

向量數＝360 × 2 抽取位置＝720；AV 每向量以 temperature 取樣 **k=5** 則描述（估計描述分布之穩定性），共 3,600 則 → 單卡 H100 上為小時級工作量。

---

## 8. 抽取規範（extraction protocol）

1. **純文字、不套 chat template**：釋出之 NLA 以 fineweb（預訓練式文本）之 activations 訓練，raw text 較 in-distribution；可另做 chat-template 複驗以檢查穩健性。
2. **層位**：依 checkpoint 之 `nla_meta.yaml`（Qwen L20／Gemma-12B L32），不硬編碼。
3. **Site A**：目標詞**最後一個 subtoken**（自迴歸資訊匯聚處）；輔以 subtoken 平均池化作敏感度分析。先以 tokenizer 驗證「台灣」與 "Taiwan"／" Taiwan"（含前導空白變體）的切分。
4. **Site B**：句末 token（句層 gist）。
5. 每句一次提及、首次提及；長度匹配依 §4.2-3。

---

## 9. AV 描述之框架標註 scheme（annotation scheme for verbalizations）

標註對象為 AV 產出的自由文本描述。採「閘門 ＋ 四層」設計：

### Layer 0：忠實度閘門（faithfulness gate）
- 以 AR round-trip MSE＝2(1−cos) 為每則描述的可信度分數。
- 先導批（pilot）上校準閾值 τ：主分析取 MSE 最佳三分位（tercile），穩健性分析納入全體並以 MSE 為共變項。**報告全體 MSE 分布**，跨語言、跨格比較（若某語言之描述系統性較不忠實，本身即是發現）。

### Layer 1：識解類型 construal（單選，取主導識解）
| 代碼 | 定義 | 判準字眼（例） |
|---|---|---|
| PLACE | 地理空間、地貌 | island, location, 位於, 島嶼 |
| POLITY | 政體、政府、政治行動者 | government, self-governing, state, 當局 |
| PEOPLE | 人群、社會、認同 | people, society, identity, 民眾 |
| ECONOMY | 產業、市場行動者 | manufacturer, exporter, 產業 |
| CULTURE | 文化圈、文化實踐 | cuisine, tradition, 習俗 |
| UNDERSPEC | 無明確識解 | — |

決策規則：多重識解並現時，取描述**首句**之識解；仍不可判則 UNDERSPEC。

### Layer 2：喚起框架（多標籤，含刺激框架 8 類＋擴充 3 類）
擴充類：**Sovereignty_dispute**（主權爭議）、**Military_conflict**（軍事衝突）、**Media_discourse**（新聞論述場景）。

派生測量：
- **frame match**：verbalized frames 是否涵蓋刺激框架。
- **frame intrusion**：描述含刺激句未觸發之框架；其中「geopolitical intrusion」＝ Sovereignty_dispute ∪ Military_conflict ∪ International_status 之侵入 → **H1 主檢定量**。

### Layer 3：視角標記 perspectival markers（特徵式編碼）
| 特徵 | 值域 | 定義 |
|---|---|---|
| CHINA_ANCHOR | 0/1 | 刺激未提及中國，描述卻以 China／PRC／cross-strait／兩岸 為參照 → **H2** |
| VIEWPOINT | internal／external／neutral | 內部視角（在地、we/our、生活者）vs 外部觀察者（"the island of…"） |
| CONTESTED | 0/1 | 出現爭議性 hedge："claimed"、"so-called"、"self-governing"、「所謂」 |
| VALENCE | pos／neu／neg | 描述之整體評價色彩 |

### Layer 4：輸出後設資料（metadata）
描述語言（en／zh／mixed；AV 輸出語言本身會漂移，是資料點不是雜訊）、長度、k=5 樣本間之框架一致率（描述穩定性）。

### 標註示例（fabricated AV outputs）

> **例 1**（刺激：ECON-01, en）AV 輸出："The concept of Taiwan as a leading producer of semiconductors, framed within global supply-chain dependence and strategic vulnerability."
> → L1: ECONOMY；L2: Commerce_and_technology ＋ **intrusion: International_status**；L3: CHINA_ANCHOR=0, VIEWPOINT=external, CONTESTED=0, VALENCE=neu；L4: en。

> **例 2**（刺激：LIFE-01, zh）AV 輸出：「描述台灣日常生活的便利，帶有在地生活經驗的親切感，如深夜的超商。」
> → L1: PLACE（生活場域）；L2: Everyday_life（match，無 intrusion）；L3: CHINA_ANCHOR=0, VIEWPOINT=internal, VALENCE=pos；L4: zh。

### 標註流程
1. **原語編碼**：雙語標註者直接就描述原文編碼（不先翻譯，避免翻譯引入框架）。
2. **先導雙盲**：50 則雙人獨立標註，各層 Cohen's κ ≥ 0.7 方進入量產；歧異逐條裁決並回寫 codebook 決策規則。
3. **LLM 輔助量產**：以 codebook 提示 LLM 初標，人工抽驗 20%（分層抽樣涵蓋各格），κ(LLM, human) 一併報告。

---

## 10. 分析計畫（sketch）

1. **主檢定（H1、H2）**：混合效應邏輯迴歸
   `geopolitical_intrusion ~ language × entity + (1|template) + (1|frame)`；
   CHINA_ANCHOR 同構。DiD 檢定量＝ language × entity 交互作用（H4）。
2. **框架漂移矩陣**：刺激框架 × verbalized 框架之混淆矩陣，分語言呈現；以 Jensen–Shannon divergence 比較兩語言矩陣，permutation test 檢定。
3. **識解分布（H3）**：L1 分布 × 語言，多項邏輯迴歸。
4. **幾何側寫（與 verbalization 三角驗證）**：各框架內兩語言條件均值向量之 cosine；語言方向 Δ_lang 之一致性（跨框架 cosine of Δs）。
5. **Design B（H5）**：`construal ~ context_language × mention_script`，看主效應誰大。
6. **因果延伸（本語料之後續用途）**：以 Δ_lang 作 steering vector 注入異語境，觀察生成偏移；並將 Δ_lang 本身餵入 AV 求其描述（OOD，須過 Layer 0 閘門）。

---

## 11. 效度威脅與因應（threats to validity）

| 威脅 | 因應 |
|---|---|
| 語言間詞頻／surprisal 不可全控 | 多模板 ＋ 隨機效應；DiD 減除語言一般效應 |
| AV 輸出語言與描述框架混淆 | Layer 4 記錄；分語言分層分析；原語編碼 |
| 詮釋學循環（Qwen 之 AV 解釋 Qwen） | 雙模型（Qwen、Gemma）平行複驗；limitations 明述 |
| 差異源於對齊訓練而非「文化」 | 措辭上主張 representational difference，佐以表層行為對照組 |
| code-switch 句 OOD | AR MSE 閘門＋呈現分布 |
| 「台灣」訓練語料中本身多與政治共現 | 這正是研究對象：base rate 由控制框架（LIFE、TRAV）之 intrusion 率估計 |

---

## 12. 待決事項（審後定案）
1. 每框架 12 模板之完整生成與母語者自然度評分流程。
2. τ 閾值之校準批規模（建議 pilot＝核心語料 10%）。
3. 是否加入第三控制實體（如泰國：亞洲＋觀光顯著＋無主權爭議）。
4. Gemma-3-12B 之平行複驗排程（同語料、L32）。
