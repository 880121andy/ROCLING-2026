上次開會提到要給 F3 投稿 ROCLING 有關 BETEL 的規劃方向，以下整理一下想法給大家，也歡迎加入討論。

首先我覺得LLM文化價值與推裡議題，在符碼操弄上，已經到一個蠻難有突破的瓶頸了。我們也許可以採取的是mechanistic interpretation 的approach。剛好目前也有國家網路中心提供的算力環境，也可以接著目前大廠風口，使用 Natural Language Autoencoder 的方式來探討這些議題。（這個部分也關聯到 @chongzhe 的 PACLIC 研究）。除了Anthropic 的paper外，我們也可以參考專案 https://github.com/kitft/natural_language_autoencoders

上次開會提到的NLA，可以說是某種“回譯檢驗” back-translation as faithfulness check。它其實就是一對模型：AV（activation verbalizer）把 residual stream 的一個向量「翻譯」成自然語言描述（vector → text），AR（activation reconstructor）只讀那段描述，「譯回」向量（text → vector）。而忠實度的判準MSE(v^,v)就是回譯品質。MSE 低，表示 AR 光靠 AV 的文字就能重建原向量的方向，某個意義上描述確實承載了向量裡的資訊。這種方法對語言學家來說是非常友善的，此外，它也給我們之前用過的 SAE 沒有提供的東西，因為每一筆解釋都帶有可信度分數（round-trip MSE），像做文化語意這種很容易被質疑「過度詮釋」的研究，這是在經驗方法論上的優勢。
另外，時間來不及的話，NLA 官方釋出的 checkpoints 剛好就是 Qwen2.5-7B-Instruct（L20）與 Gemma-3-12B/27B-IT（L32／L41），我們甚至不必重跑 SFT＋GRPO 的訓練流程，只需要做推論，國網中心算力應該綽綽有餘。

投稿 ROCLING (7/20)

模型在英語語境與台灣中文語境之下，提到「台灣」（或其他更有文化意涵的關鍵詞彙們）時的內部觸發機制？（「台灣」的跨語境表徵，within-model）

我們可以建構最小對比句對 (當成是 BETEL 的另一個類別) 。就是設計內容相同、語言不同的英文／台灣中文語料，涵蓋不同框架（地理、政治、飲食、科技等語境。這裡就是 frame semantics 可以發揮的空間）。在「台灣」／"Taiwan" 的 token 位置抽取 L20（Qwen）或 L32（Gemma-12B）的 residual activation，交給 AV 產生描述，用 AR 的 MSE 過濾不可靠的解釋，再對描述做框架標註與分群。假說例如英文語境的向量描述更常落入 geopolitical frame，中文語境更常落入 in-group／地方生活 frame。進一步也可做因果驗證，把英文語境的「台灣向量」steering 注入中文語境，觀察生成行為是否偏移。

附檔：
- 16 組示範用的句對、標註 codebook 與分析計畫。（參考用，需要人的驗證！之後對每個框架 12 模板生出較完整的 (360 句?) 語料(CSV 含 entity、frame、language、site 等欄位）。這可以整到 BETEL。
- tokenizer 驗證腳本。因為要先確認「台灣」／"Taiwan" 在 Qwen 與 Gemma 中的 subtoken 切分與長度匹配是否可行。照理說先驗證切分再量產句子，順序上更合理。我的預期是中文句 token 數會系統性的少於英文句，少數的句對可能逼近或超出 ±20% 。如果 summary.md 顯示超過的集中在某些框架，我們再回去微調模板長度即可。週三下午meeting我們可以先看結果，再進行 360 句的量產。