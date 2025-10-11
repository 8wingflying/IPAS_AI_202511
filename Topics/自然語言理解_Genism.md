# 📚 **Gensim** 
> **Gensim** 是 NLP 中最強大的 **主題建模與語意分析工具之一**，  
> 支援：
> - 📖 文本前處理（Tokenization / Stopwords / Lemmatization）  
> - 🧮 語料轉換（BOW / TF-IDF / LSI / LDA / NMF）  
> - 🧠 詞向量模型（Word2Vec / FastText / Doc2Vec）  
> - 🔍 文件相似度與語意搜尋  
> - ⚙️ 高效能與記憶體優化（支援大型語料流式訓練）  

---


# 🧠 Gensim 各項功能與常用函數總覽表  

---

## 一、Gensim 概要（Overview）

| 模組名稱 | 中文說明 | 功能重點 |
|------------|------------|------------|
| `gensim.corpora` | 語料與字典模組 | 建立詞典、語料庫、Bag-of-Words 模型 |
| `gensim.models` | 模型模組 | Word2Vec、Doc2Vec、LDA、TF-IDF、LSI 等 |
| `gensim.similarities` | 相似度模組 | 文件相似度計算與檢索 |
| `gensim.parsing` | 文本前處理 | 分詞、去除停用詞、詞形還原 |
| `gensim.utils` | 工具模組 | 資料載入、tokenize、儲存模型 |
| `gensim.downloader` | 模型下載 | 預訓練詞向量（如 GloVe、word2vec-google-news） |

---

## 二、文本處理與語料建立（Text Preprocessing）

| 函數 / 類別 | 功能說明 | 範例 | 備註 |
|--------------|------------|-------|------|
| `simple_preprocess()` | 分詞 + 小寫 + 去除符號 | `gensim.utils.simple_preprocess("Hello World!")` | 輸出詞列表 |
| `remove_stopwords()` | 移除停用詞 | `from gensim.parsing.preprocessing import remove_stopwords` | 預設英文 |
| `strip_punctuation()` | 去除標點符號 | — | — |
| `strip_numeric()` | 去除數字 | — | — |
| `strip_short()` | 移除短詞（長度 ≤ N） | `strip_short(text, minsize=3)` | — |
| `preprocess_string()` | 一次完成清理流程 | `from gensim.parsing.preprocessing import preprocess_string` | 結合多個 filter |
| `split_sentences()` | 句子切割 | `split_sentences(text)` | — |

---

## 三、字典與語料庫（Dictionary & Corpus）

| 類別 / 函數 | 功能說明 | 範例 | 備註 |
|---------------|------------|-------|------|
| `Dictionary()` | 建立詞典（字→ID） | `id2word = corpora.Dictionary(texts)` | — |
| `.token2id` | 詞與索引映射 | `id2word.token2id` | — |
| `.doc2bow()` | 將文件轉為 Bag-of-Words | `id2word.doc2bow(["apple","banana"])` | 輸出 `(id, count)` |
| `.filter_extremes()` | 過濾高頻 / 低頻詞 | `no_below`, `no_above` | — |
| `.save()` / `.load()` | 儲存 / 載入詞典 | — | — |
| `MmCorpus()` / `BleiCorpus()` / `TextCorpus()` | 不同語料格式 | 用於大型語料訓練 | — |

---

## 四、TF-IDF 模型（Term Frequency–Inverse Document Frequency）

| 類別 / 函數 | 功能說明 | 範例 | 備註 |
|---------------|------------|-------|------|
| `TfidfModel()` | 建立 TF-IDF 模型 | `tfidf = models.TfidfModel(corpus)` | 訓練模型 |
| `.transform()` | 將語料轉為 TF-IDF | `tfidf[doc_bow]` | — |
| `.save()` / `.load()` | 模型儲存 / 載入 | — | — |

---

## 五、主題模型（Topic Modeling）

| 模型名稱 | 功能說明 | 常用參數 | 範例 |
|------------|------------|-------------|------|
| `LdaModel()` | Latent Dirichlet Allocation 主題模型 | `num_topics`, `passes`, `alpha`, `eta` | `lda = LdaModel(corpus, id2word, num_topics=5)` |
| `LdaMulticore()` | 多核心 LDA | `workers` | 提高運算效率 |
| `LsiModel()` | Latent Semantic Indexing（潛在語意索引） | `num_topics` | `lsi = LsiModel(tfidf_corpus, id2word)` |
| `HdpModel()` | Hierarchical Dirichlet Process | 自動決定主題數 | `hdp = HdpModel(corpus, id2word)` |
| `Nmf()` | 非負矩陣分解主題模型 | `num_topics`, `passes` | `nmf = Nmf(corpus, id2word)` |
| `.show_topics()` | 顯示主題字詞 | `num_words=10` | — |
| `.get_document_topics()` | 文件主題分佈 | — | — |

---

## 六、詞向量模型（Word Embedding Models）

| 模型名稱 | 功能說明 | 常用參數 | 範例 |
|------------|------------|-------------|------|
| `Word2Vec()` | 詞向量模型 | `vector_size`, `window`, `min_count`, `sg`, `epochs` | `w2v = Word2Vec(sentences, vector_size=100)` |
| `FastText()` | 字根詞向量模型 | `min_n`, `max_n` | `ft = FastText(sentences, vector_size=100)` |
| `Doc2Vec()` | 文件向量模型 | `dm`, `vector_size`, `window` | `d2v = Doc2Vec(docs, vector_size=100)` |
| `.train()` | 模型訓練 | `total_examples`, `epochs` | — |
| `.wv.most_similar()` | 相似詞查詢 | `positive`, `topn` | `w2v.wv.most_similar("king")` |
| `.wv.similarity()` | 詞距離 | `w2v.wv.similarity("car","automobile")` | — |
| `.save()` / `.load()` | 模型儲存 / 載入 | — | `.load("model.model")` |
| `.wv.save_word2vec_format()` | 輸出為 word2vec 格式 | — | — |

---

## 七、文件相似度與檢索（Document Similarity）

| 模組 / 類別 | 功能說明 | 範例 | 備註 |
|---------------|------------|-------|------|
| `SparseMatrixSimilarity()` | 稀疏矩陣相似度計算 | `index = SparseMatrixSimilarity(tfidf_corpus, num_features=1000)` | 快速 |
| `MatrixSimilarity()` | 稠密矩陣相似度 | — | 小型語料 |
| `Similarity()` | 自動索引 + 相似度查詢 | `index = Similarity(None, corpus_tfidf, num_features=500)` | 可儲存 |
| `.get_similarities()` | 回傳相似度分數 | — | — |
| `.save()` / `.load()` | 儲存索引 | — | — |

---

## 八、預訓練模型下載（Pretrained Models）

| 函數名稱 | 功能說明 | 範例 | 備註 |
|------------|------------|-------|------|
| `api.load()` | 載入預訓練詞向量 | `glove = api.load("glove-wiki-gigaword-100")` | 自動下載 |
| `glove.most_similar("king")` | 查詢相似詞 | — | — |
| 支援模型 | `"word2vec-google-news-300"`, `"fasttext-wiki-news-subwords-300"`, `"glove-twitter-25"` | — | — |

---

## 九、相似度與距離計算（Similarity Metrics）

| 函數名稱 | 功能說明 | 範例 | 備註 |
|------------|------------|-------|------|
| `similarity()` | 計算兩詞相似度 | `model.wv.similarity("car", "truck")` | — |
| `n_similarity()` | 多詞集合相似度 | `model.wv.n_similarity(['man'], ['king'])` | — |
| `doesnt_match()` | 找出不相關詞 | `model.wv.doesnt_match("breakfast cereal dinner lunch".split())` | — |
| `most_similar_cosmul()` | 餘弦相似度（cosmul） | `model.wv.most_similar_cosmul(positive=["woman", "king"], negative=["man"])` | — |

---

## 十、實用工具與序列化（Utilities）

| 函數 / 方法 | 功能說明 | 範例 | 備註 |
|---------------|------------|-------|------|
| `save()` / `load()` | 模型儲存 / 載入 | `model.save("model.model")` | — |
| `corpora.MmCorpus.serialize()` | 儲存語料 | — | — |
| `utils.simple_preprocess()` | 快速分詞 | — | NLP 前處理 |
| `logger` | 設定訓練日誌 | `import logging; logging.basicConfig(format='...')` | — |

---



✅ **典型範例：Word2Vec 訓練與相似度查詢**

```python
from gensim.models import Word2Vec
from gensim.utils import simple_preprocess

sentences = [
    simple_preprocess("Deep learning improves NLP models"),
    simple_preprocess("Natural language processing is powerful"),
    simple_preprocess("Word embeddings capture semantic meaning")
]

model = Word2Vec(sentences, vector_size=100, window=5, min_count=1, sg=1)
model.save("word2vec.model")

print(model.wv.most_similar("language"))
