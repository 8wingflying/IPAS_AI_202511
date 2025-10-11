## NLP 工具
- spaCy
- transformers

## NTLK模組
- NLTK (Natural Language Toolkit)
- NLTK是Python中最著名的自然語言處理(NLP)函式庫之一,提供了豐富的文本處理工具和語料庫。以下是主要模組的介紹:
- 核心模組
  - tokenize (分詞模組)  用於將文本切分成更小的單位(詞、句子等)
  - word_tokenize(): 將文本分割成單詞
  - sent_tokenize(): 將文本分割成句子
  - RegexpTokenizer: 使用正則表達式自定義分詞規則
  - 2. corpus (語料庫) 提供多種預建的語料庫資源 包含布朗語料庫、路透社新聞、電影評論等 可用於訓練和測試NLP模型
  - 3. stem (詞幹提取) 將單詞還原到詞幹形式
  - PorterStemmer: 最常用的英文詞幹提取器
  - LancasterStemmer: 更激進的詞幹提取
  - SnowballStemmer: 支援多種語言
  - 4. tag (詞性標註)標註單詞的詞性(名詞、動詞、形容詞等)
  - pos_tag(): 為單詞標註詞性  支援多種標註集(Penn Treebank等)
  - 5. chunk (組塊分析) 將詞性標註後的文本組合成更大的語法單位
  - 名詞短語提取 自定義組塊規則
  - 6. parse (句法分析) 分析句子的語法結構
  - 生成句法樹 依存句法分析
  - 7. sentiment (情感分析) 分析文本的情感傾向
  - SentimentIntensityAnalyzer: VADER情感分析器 適用於社交媒體文本
  - 8. probability (機率模組) 提供機率分佈和統計工具  頻率分佈 條件頻率分佈
  - 9. metrics (評估指標)  評估NLP模型性能  準確率、召回率、F-score  混淆矩陣
  - 10. classify (分類器) 文本分類工具
    - 樸素貝葉斯分類器
    - 決策樹分類器
    - 最大熵分類器
- collocations: 搭配詞提取
- ngrams: N-gram生成
- distance: 計算字串相似度
- translate: 機器翻譯相關工具
- draw: 可視化句法樹

## 詞幹提取 (Stemming) 
- 將單詞還原到其基本形式(詞幹)的過程,通過移除詞綴(前綴、後綴)來獲得單詞的核心部分。
```python
"running", "runs", "ran" → "run"
"better", "good" → 可能處理為相似形式
"connection", "connected", "connecting" → "connect"
```
- 目的:
  - 減少詞彙變化,將相似單詞統一處理
  - 降低文本特徵維度
  - 提高信息檢索和文本分析效率

## 範例
```python
import nltk

# 下載所需資源
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')

# 分詞
from nltk.tokenize import word_tokenize
text = "NLTK is a powerful NLP library."
tokens = word_tokenize(text)

# 詞性標註
from nltk import pos_tag
tagged = pos_tag(tokens)

# 停用詞過濾
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))
filtered = [w for w in tokens if w.lower() not in stop_words]

# 詞幹提取
from nltk.stem import PorterStemmer
stemmer = PorterStemmer()
stemmed = [stemmer.stem(w) for w in tokens]

```
##
- https://learning.oreilly.com/library/view/natural-language-processing/9781787285101/
- https://blog.csdn.net/qq_36070104/article/details/143240958
- https://cloud.tencent.com/developer/article/1765546
## 延伸閱讀
- NLTK 初學指南(一)：簡單易上手的自然語言工具箱－探索篇
- [Natural Language Processing: Python and NLTK](https://learning.oreilly.com/library/view/natural-language-processing/9781787285101/)
