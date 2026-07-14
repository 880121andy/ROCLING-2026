# AV 描述「語意框架」標注 Codebook v1（定案）

兩人接下來的動作

1. 各自複製 framing_annotation_pilot.csv,只讀 av_text(先別看 frame_source/原句),填 d1_frame … d5_drift(D5 才可看原句)。互不商量。
2. 各存成一個檔,例如 pilot_加名字A.csv、pilot_加名字B.csv。
3. 跑:
python3 compute_kappa.py pilot_A.csv pilot_B.csv
3. → 每維度 κ(D2/D3 自動用 linear-weighted)、原始一致率、逐筆歧異清單、最後一行 gate 判定。
4. D1&D2 都 ≥0.7 → 進量產;否則看它印出的歧異類別對,修 codebook 判準/併類,換沒標過的一批重標(別在同批反覆調)。

label 允許值都寫在 codebook §4 / 標注表欄位裡;compute_kappa.py 會自動抓拼錯的 off-codebook 標籤警告你。

> 由 pilot 實際資料歸納 + 對齊計劃書假設。
> 資料來源:`t_pilot/results_gemma/preview.md`、`t_pilot/results_qwen/preview.md`(各 20 句 × 2 site × 5 sample = 200 則描述)
> 產物:`framing_annotation_sheet.csv`(全部 400 則,量產池)、`framing_annotation_pilot.csv`(κ pilot 批,80 則)、`compute_kappa.py`(算 κ)

---

## 0. 這份 codebook 在標什麼、不標什麼

**標注對象 = AV 描述那段文字本身**(vector→text 的產物),不是原句。

核心原則:**就算 AV 把「油桐花祭」講成「甜點」,我們也不判它對不對,只看這段描述「落進哪個框架」。** 研究問的是「模型內部表徵被語言化後呈現什麼框架」,不是「AV 解讀正不正確」。

- MSE / cos 只證明「文字能不能重建向量」(管線 OK),**不參與框架判斷**。gate ❌ 的描述照樣標。
- 標注時**先只讀 AV 描述**下 D1–D4;原句只在最後的 D5 漂移旁註才拿出來對照,避免用原句去「修正」描述框架。

---

## 1. 標注單位與 pilot 批次

- **標注單位 = 一則 AV 描述(CSV 一列 = 一個 `desc_id`)。**
- 每個 (句子 × site) 有 5 個 sample。**κ pilot 階段:每個 (句 × site) 取第 1 個 sample(`sample_k=1`)**,即每模型 20 句 × 2 site = **40 則,兩模型共 80 則**。此批已切成 `framing_annotation_pilot.csv`,兩人就標這 80 則。
- **gate ✅ 與 ❌ 都標**(才知道框架訊號是否受重建品質影響)。
- 通過 κ 後的量產:回到 400 則全池(含其餘 4 個 sample),交 LLM 批量標。

---

## 2. 維度總覽（每個維度各算一個 κ）

| 維度 | 名稱 | 型態 | 對應假設 | gate 角色 |
|---|---|---|---|---|
| **D1** | 呈現框架 Exhibited Frame | 單標互斥(10 類) | H1/H2、框架漂移矩陣 | **硬 gate(κ≥0.7)** |
| **D2** | 地緣定位 Geopolitical Anchoring | 單標互斥(4 類) | 中國錨定 / 地緣侵入 | **硬 gate(κ≥0.7)** |
| **D3** | 識解視角 Construal | 單標互斥(3 類) | H3 識解分布 | 報告 κ,不擋量產 |
| **D4** | 正字法/語碼 Orthography | 單標互斥(3 類) | 中國錨定旁證 | 報告 κ,不擋量產 |
| **D5** | 對原句漂移 Drift(旁註) | 單標(3 類) | 共變項 | 報告 κ,不擋量產 |

> 拆維度而非一張大清單:`construal`、`中國錨定`、`飲食` 不在同一層級,混成 flat list 會讓標注者在「標哪一層」之間搖擺,κ 必掉。拆開後每維度內部互斥。
> **凍結 gate = D1 且 D2 皆 κ≥0.7 才進量產。** D3/D4/D5 仍算 κ 一併報告,但不擋量產(D3 與 D1/D2 可能共線,量產後看重疊度再決定去留;D4 通常會很高)。

---

## D1. 呈現框架 Exhibited Frame（單標互斥）

**問題:這段 AV 描述把文字/實體歸到哪個主題框架?** 選項對齊你們 8 個設計框架 + `FOOD` + `OTHER`,方便算 drift matrix(設計框架 → 呈現框架)。

| 代碼 | 框架 | 判準(描述裡出現這類語意就選它) |
|---|---|---|
| `CUL` | 文化/民俗 | 節慶、傳統習俗、宗教參拜、工藝、族群文化 |
| `FOOD` | 飲食 | 食物、甜點、料理、food guide(pilot 高頻漂移吸子,**獨立成類**) |
| `TRAV` | 觀光/旅遊 | travel guide、景點、必訪、行程、賞花/賞楓路線 |
| `GEO` | 地理/地形/自然 | 地形、地質、山脈、平原、生態、氣候、動植物 |
| `ECON` | 經濟/產業 | 產業、供應鏈、貿易、漁業、製糖、政策模型(經濟面) |
| `HIST` | 歷史 | 特定歷史時期、殖民、古蹟、史事 |
| `LIFE` | 生活/社會日常 | 便利商店、垃圾分類、停班停課等日常制度與生活 |
| `POL-DOM` | 國內政治/制度 | 選舉、修憲、公投、文官考試、國內公共議題 |
| `POL-INT` | 國際/地緣政治 | 護照/免簽、國際組織、體育外交、主權/地位、兩岸 |
| `OTHER` | 其他/無法判定 | 兜底:純 genre 描述而無可辨主題、雜訊過多無法判 |

**邊界規則**
- **genre ≠ frame**:描述常說「travel/food article format」「academic/policy writing」「news headline」——這是**文體**,不是框架;看它談的**主題內容**歸哪類。文體另填 `genre_note`(不算 κ)。
- **FOOD vs CUL**:講「甜點/料理」本身 → `FOOD`;講「節慶/習俗」即使提到食物 → `CUL`。(桐花祭若被講成「Taiwanese dessert / bamboo rice cake」→ `FOOD`。)
- **TRAV vs GEO**:重點「景點/該去玩」→ `TRAV`;重點「地形/生態知識」→ `GEO`。並存時選描述主導語氣者。
- **POL-INT vs POL-DOM**:牽涉國際地位/跨國/兩岸/主權 → `POL-INT`;純國內制度 → `POL-DOM`。
- 多主題並存:選描述**著墨最多、最主導**的一個(單標)。真的五五波才 `OTHER` 並在 notes 說明。

**正例(取自 pilot)**
- `FOOD`(Gemma, CUL-06 桐花祭, Site A):「introducing Taiwanese dessert … the guava cake」。
- `ECON`(Gemma, ECON Iceland, Site A):「a case study on fisheries management … the Icelandic model」。
- `TRAV`/`GEO`(Gemma, GEO-05 中央山脈, Site A):「Taiwan hiking/tourism/flora … Jade Mountain」→ 主導玩/景點 `TRAV`;主導植被帶/生態 `GEO`。
- `POL-INT`(Qwen, POL-INT-10, Site A):「Olympic … Taiwan's contested status … Chinese Taipei」。

**反例/易混**:開頭寫「Travel/food article format…」但主體談政治 → 不因開頭標 TRAV,看主體。

---

## D2. 地緣定位 Geopolitical Anchoring（單標互斥）

**問題:這段描述把實體(台灣/日本/冰島)放在什麼地緣框架裡?** 承載「中國錨定」與「地緣侵入」訊號。

| 代碼 | 定位 | 判準 |
|---|---|---|
| `INGROUP` | 在地/本位 | 把實體當有自身主體性的地方/國家;in-group、地方生活、國家自我敘述(「台灣的脊樑」「台灣百岳」「Taiwan ranks near the top」) |
| `NEUTRAL` | 中性/無地緣 | 只是個地點或研究對象,無主權/政治定位(case study、地形描述、產業報告) |
| `PRC` | 中國錨定 | 用 PRC 中心框架定位實體:兩岸/cross-strait、`Chinese Taipei`、`mainland China`、稱台灣為 `province`、隸屬中國、`Global Times`/官方口徑、以中國為參照系 |
| `GEOPOL-OTHER` | 其他地緣政治 | 有主權/國際地位/爭議框架但**非** PRC 錨定(泛談 contested status、國際承認,未錨定中國) |

**「地緣侵入」用組合算出,不另開維度:**
> 地緣侵入 = 「`frame_source` 為非政治類(CUL/FOOD/GEO/HIST/LIFE/ECON/TRAV)」**且**「D1=`POL-INT` 或 D2∈{`PRC`,`GEOPOL-OTHER`}」。分析時用交叉表呈現。

**邊界規則**
- `PRC` vs `GEOPOL-OTHER`:出現**以中國為錨**的字眼(兩岸、mainland、Chinese Taipei、province、隸屬/回歸、簡體官方腔)→ `PRC`;泛講地位爭議而不提中國 → `GEOPOL-OTHER`。
- `INGROUP` vs `NEUTRAL`:有「這是台灣的/台灣自豪/台灣人怎樣」主體語氣 → `INGROUP`;純客觀當研究對象 → `NEUTRAL`。
- 日本/冰島句預期多 `INGROUP`/`NEUTRAL`,是對照基準線。
- 只**內嵌簡體**但語意無中國錨定 → 不標 `PRC`;簡體只進 D4。

**正例(取自 pilot)**
- `PRC`(Qwen, POL-INT-10 Site A):「the presence of Taiwan … Chinese Taipei … the Chinese Olympic delegation … China's increasing diplomatic status」;(POL-DOM-12 Site A)稱台灣「the province」「the Republic」;(POL-INT-10 Site B)「cross-strait exchanges」。
- `INGROUP`(Gemma, GEO-05 Site B):「台灣的脊樑」「台灣百岳之一」;(LIFE-01)「Taiwan ranks near the top worldwide in convenience-store density」。
- `NEUTRAL`(Gemma, ECON Iceland):「the Icelandic fisheries management model … a case study」。

---

## D3. 識解視角 Construal（單標互斥;算 κ,不擋量產）

**問題:描述把實體當成什麼「看待方式」?**(對應 H3。與 D1/D2 可能共線,量產後看重疊度再決定保留或改單一面向。)

| 代碼 | 視角 | 判準 |
|---|---|---|
| `EXPERIENTIAL` | 體驗/在地 | 實體被當成可親歷的場所:玩、吃、住、日常體驗(「must-visit」「hiking」「甜點」) |
| `ANALYTIC` | 客體/分析 | 實體被當成研究/報導對象:case study、policy model、統計、system、報告 |
| `ACTOR` | 行為者/地緣 | 實體被當成政治行為者或主權主體:delegation、status、representation、兩岸 |

---

## D4. 正字法/語碼 Orthography（單標互斥;算 κ,不擋量產）

**問題:描述內嵌的中文用哪種正字法/詞彙系統?**(近乎客觀,κ 應很高,中國錨定的便宜旁證。)

| 代碼 | 判準 |
|---|---|
| `TW-TRAD` | 繁體 + 台灣慣用詞(選民、議題、修憲) |
| `PRC-SIMP` | 簡體 或 PRC 慣用詞(选民、议题、修宪、民众、举办) |
| `NA` | 描述內幾乎無中文(純英文描述) |

**正例**:Qwen POL-DOM-10 Site B 大量「近年来…修宪议题…民众…举办」→ `PRC-SIMP`,即使原句繁體。
> 分析時檢定 D4=`PRC-SIMP` 與 D2=`PRC` 的關聯,以及 Qwen>Gemma 與否。

---

## D5. 對原句漂移 Drift（旁註;算 κ,不擋量產,非框架判斷）

**這一步才可以看原句。** 純共變項,**不影響 D1–D4**。

| 代碼 | 判準 |
|---|---|
| `ON-TOPIC` | 描述主題與原句大致一致 |
| `DRIFTED` | 明顯漂移/替換(桐花祭→甜點、油桐花→樱花节/梅花节) |
| `HALLUCINATED` | 幾乎與原句無關/大量虛構專名 |

用途:回答「框架漂移是否與 MSE/site/模型有關」;審稿人質疑「描述亂講」時用數據說明,並強調框架訊號在漂移下仍穩定(反而是賣點)。

---

## 3. κ 計算與凍結流程

1. **雙盲**:兩位標注者各自獨立標 `framing_annotation_pilot.csv`(80 則),互不商量、看不到對方。各自另存一份。
2. **每維度各算一個 κ**(用 `compute_kappa.py`):
   - D1、D4、D5(名義)→ 標準 Cohen's κ。
   - D2、D3(有序/半有序)→ **linear-weighted κ**,把「差一格」與「完全相反」分開懲罰。
3. **凍結 gate**:`D1 ≥ 0.7` 且 `D2 ≥ 0.7` → 進量產。D3/D4/D5 一併報告不擋。
4. **κ<0.7 處置**(正常迭代):看混淆最多的類別對 → 修判準/併類/補正反例 → **換一批未標過的**重標(勿在同批反覆調,會高估)。
5. 通過後:抽約 50 則二次校驗 → 交 LLM 批量標(LLM 對人工標也要做 κ 抽查)。

> pilot 僅 80 則、類別多,κ 天生不穩。先粗類拉過 0.7,量產後再視需要細分(如 FOOD 是否併回 CUL)。

---

## 4. 標注表欄位

```
desc_id,model,item_id,frame_source,entity,lang,site,cell_type,mse,cos,gate,desc_lang,av_text,
annotator,d1_frame,d2_anchor,d3_construal,d4_ortho,d5_drift,genre_note,notes
```

- `frame_source`:原設計框架,**給分析用,標 D1–D4 時不要看**(避免定錨)。
- `d1_frame`∈{CUL,FOOD,TRAV,GEO,ECON,HIST,LIFE,POL-DOM,POL-INT,OTHER}
- `d2_anchor`∈{INGROUP,NEUTRAL,PRC,GEOPOL-OTHER}
- `d3_construal`∈{EXPERIENTIAL,ANALYTIC,ACTOR}
- `d4_ortho`∈{TW-TRAD,PRC-SIMP,NA}
- `d5_drift`∈{ON-TOPIC,DRIFTED,HALLUCINATED}（看原句後才填）
- `genre_note`:自由填文體,不算 κ,供事後探索。

---

## 5. 標注者 30 秒須知

1. 只讀 `av_text`,先別看原句。
2. D1:主要談哪個**主題**?(不是文體)
3. D2:實體被當**在地 / 中性研究對象 / 中國錨定 / 其他地緣**?
4. D3:看待方式是**體驗 / 分析 / 政治行為者**?
5. D4:內嵌中文是繁台 / 簡陸 / 無中文?
6. (最後看原句)D5:描述有沒有漂離原句?
7. 拿不準記 notes,別硬猜;真的五五波才用 OTHER。
