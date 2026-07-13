ROCLINGGG
實驗的大概流程：

┌──────────────────────────────────────────────┐
 │ 01 資料準備 Data Preparation │
 └──────────────────────────────────────────────┘
 │
 ├─ Pilot驗證（16對，定出前導≥10tok／長度差≤20%規則）:ok:
 ├─ 語料量產（核心192＋控制120＋DesignB48＝360句）:ok: 
 ├─ Tokenizer結構驗證（360/360通過）:ok: 
 ├─ 人工內容驗證（命題對等＋自然度）（7/13）
 └─ τ閾值校準批
 │
 ▼
 ┌──────────────────────────────────────────────┐
 │ 02 模型環境建設 Model & Extraction Env │
 └──────────────────────────────────────────────┘
 │
 ├─ 取得NLA官方AV/AR checkpoint
 ├─ 建置抽取環境（GPU，純文字不套chat template）
 ├─ Activation抽取（360句×2 site＝720個向量）
 ├─ AV自然語言化（720×k=5＝3,600則描述）
 └─ AR round-trip忠實度計算（全部3,600則算MSE）
 │
 ▼
 ┌──────────────────────────────────────────────┐
 │ 03 結果分析 Results Analysis │
 └──────────────────────────────────────────────┘
 │
 ├─ Layer 0 忠實度閘門（τ篩選）
 ├─ Layer 1–4 標註（雙盲pilot κ≥0.7→LLM輔助量產）
 ├─ 主檢定 H1/H2/H4（混合效應迴歸，語言×實體交互作用）
 ├─ 框架漂移矩陣（JS divergence + permutation test）
 ├─ 識解分布 H3（多項邏輯迴歸）
 ├─ 幾何側寫（向量cosine分析，跨框架Δ_lang一致性）
 ├─ Design B 分析 H5（語境語言 vs 提及詞形主效應）
 └─ 撰寫發現，對照§11效度威脅逐項檢視

RQ1 資料建構重點整理

給人工審閱 rq1_review_all.csv（360 列）時參考，以及釐清 Design / Site 這兩組容易混淆的 A/B 命名。

一、審閱表要填什麼

檔案：GoogleSheet（360 列 = 核心192 + 控制120 + Design B 48）
總共有以下Metadata欄位：pair_id、frame、entity、lang、mention_script、cell_type、mention、text

propositional_equivalence（命題對等）

* 適用對象：只有 cell_type = baseline 的列（也就是核心語料 192 句＋控制語料 120 句，共 312 列）。
* 怎麼填：回譯核可——把 zh 版跟 en 版對照看，確認是同一個命題的翻譯等值句。
* 填法建議：同一個 pair_id 的 zh 列跟 en 列填同一個值（因為評的是「這一對」對不對等，不是單句）。可以用簡單的 Y / N / partial（部分對等，需在 reviewer_notes 註明哪裡不對等）。
* Design B（cell_type = codeswitch 或 Design B 的 baseline）：這欄留空——因為 Design B 是從已核可的核心句做「只換提及詞形」的最小改動，命題內容跟原句保證一致，不需要重新評。

naturalness（自然度）

* 適用對象：全部 360 列都要填，每一列獨立評分（同一個 pair 的 zh、en 兩列分開打分，不是填一樣的值）。
* 填寫：1–5 分，<4 分視為需要改寫。
* 評分標準依 cell_type 不同：
    * baseline（核心／控制語料、以及 Design B 對角線那兩格）：這句話讀起來像不像道地母語者會寫的句子（zh 用台灣繁體用語習慣，en 用道地英文）。
    * codeswitch（Design B 跳詞形句）：標準不是「這是不是通順的純中文/純英文」——它本來就故意夾雜兩種語言，評的是「這讀起來像不像雙語者真的會這樣說/寫的自然 code-switch」，不是語法純度。

reviewer_notes（備註）

自由文字。建議至少記錄：

* 任何 naturalness <4 分的句子，具體哪裡讀起來不順、建議怎麼改
* propositional_equivalence 標 N/partial 的句子，具體哪裡不對等
* 順手發現的規則違反（例如不小心把實體用成專名內部修飾語 e.g. 台灣牛奶、不小心觸犯框架排除項），即使兩個機器檢查（tokenizer 驗證、排除詞掃描）都過了，人工還是可能抓到機器漏掉的問題


二、Design A/B 跟 Site A/B 的差別

這是兩組完全獨立、互不相關的東西，只是剛好都用 A/B 命名，容易搞混。


	回答的問題
	A
	B

Site A / Site B
	一句話裡，從哪個位置抽模型向量
	提及詞（「台灣」/"Taiwan"）本身的位置
	句子最後一個字的位置

Design A / Design B
	用哪一種句子做實驗
	正常語料：核心 192 句＋控制 120 句，語境語言跟詞形永遠一致（中文配台灣、英文配Taiwan）
	跳詞形語料：48 句，故意把語境語言跟詞形錯開（Code-switching）


兩者是交叉關係，不是誰包含誰：每一句話（不管 Design A 還是 Design B）都要在 Site A、Site B 各抽一次向量。


	Site A
	Site B

Design A（正常句）
	抽一次
	抽一次

Design B（跳詞形句）
	抽一次
	抽一次


