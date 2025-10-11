# 📚 **Statsmodels** 
> **Statsmodels** 是 Python 的統計建模核心套件，  
> 提供 **迴歸分析、時間序列、假設檢定、ANOVA、非參數估計與多變量分析**。  
>  
> 它特別適合進行 **經濟學、社會科學、行為科學、金融統計** 等研究分析，  
> 並以「**可解釋性（Interpretability）**」與「**統計嚴謹性（Statistical Rigor）**」為主要特色。  

# 📊 Statsmodels 各項功能與常用函數總覽表  


---

## 一、Statsmodels 概要（Overview）

| 模組名稱 | 中文說明 | 功能重點 |
|-----------|------------|-----------|
| `statsmodels.api` | 主模組 | 匯入統一介面，支援多種模型 |
| `statsmodels.formula.api` | 公式介面 | 類似 R 語法 (`y ~ x1 + x2`) |
| `statsmodels.regression` | 迴歸分析 | OLS、GLS、WLS、Quantile 等 |
| `statsmodels.discrete` | 離散選擇模型 | Logit、Probit、Poisson、MNLogit |
| `statsmodels.tsa` | 時間序列分析 | ARIMA、SARIMA、VAR、季節分解 |
| `statsmodels.stats` | 統計檢定與推論 | T-test、ANOVA、正態性檢定等 |
| `statsmodels.multivariate` | 多變量分析 | MANOVA、PCA、因子分析 |
| `statsmodels.nonparametric` | 非參數統計 | Kernel Density、Lowess |
| `statsmodels.tools` | 工具模組 | 模型摘要、資料轉換、診斷工具 |

---

## 二、資料準備與匯入（Data Preparation）

| 功能類別 | 功能說明 | 常用函數 / 方法 | 備註 |
|------------|------------|------------------|------|
| 新增常數項 | 增加截距常數 | `add_constant(df)` | 用於回歸模型 |
| 樣本分割 | 分離訓練 / 測試資料 | `train_test_split()`（搭配 sklearn） | 常與模型驗證併用 |
| 資料格式轉換 | 將 DataFrame 轉 numpy | `.values` 或 `.to_numpy()` | 模型要求 numpy 格式 |
| 公式建模 | R 類語法建模 | `ols('y ~ x1 + x2', data=df)` | 使用 formula API |

---

## 三、線性迴歸模型（Linear Models）

| 功能類別 | 功能說明 | 常用函數 / 方法 | 備註 |
|------------|------------|------------------|------|
| 普通最小平方法 | Ordinary Least Squares | `OLS(y, X).fit()` | 最常見線性迴歸 |
| 加權最小平方法 | Weighted Least Squares | `WLS(y, X, weights=w).fit()` | 處理異方差 |
| 一般最小平方法 | Generalized LS | `GLS(y, X, sigma).fit()` | 處理自相關誤差 |
| 分位數迴歸 | Quantile Regression | `QuantReg(y, X).fit(q=0.5)` | 中位數迴歸 |
| 線性模型摘要 | 模型結果摘要 | `.summary()` | 輸出回歸表與統計指標 |

---

## 四、離散選擇模型（Discrete Choice Models）

| 模型類別 | 功能說明 | 常用函數 / 方法 | 備註 |
|------------|------------|------------------|------|
| 二元邏輯斯迴歸 | Logistic Regression | `Logit(y, X).fit()` | 適用於 0/1 類別 |
| Probit 模型 | 機率迴歸 | `Probit(y, X).fit()` | 使用常態分布 |
| Poisson 模型 | 計數資料模型 | `Poisson(y, X).fit()` | 適用於事件次數 |
| Multinomial Logit | 多分類模型 | `MNLogit(y, X).fit()` | 多類別分類 |
| 負二項模型 | 計數模型（過度分散） | `NegativeBinomial(y, X).fit()` | — |

---

## 五、時間序列分析（Time Series Analysis, `tsa`）

| 功能類別 | 功能說明 | 常用函數 / 方法 | 備註 |
|------------|------------|------------------|------|
| AR 模型 | 自迴歸 | `AR(y).fit()` | 單變量時間序列 |
| ARIMA 模型 | 差分與移動平均 | `ARIMA(y, order=(p,d,q)).fit()` | — |
| SARIMA 模型 | 季節性時間序列 | `SARIMAX(y, order=(p,d,q), seasonal_order=(P,D,Q,s))` | — |
| VAR 模型 | 多變量時間序列 | `VAR(df).fit(lags)` | 多維度時間依存 |
| 指數平滑 | 指數加權移動平均 | `SimpleExpSmoothing(y).fit()` | — |
| 季節分解 | 趨勢 / 季節 / 隨機成分 | `seasonal_decompose(y, model='additive')` | — |
| 單位根檢定 | ADF stationarity test | `adfuller(y)` | — |
| 自相關分析 | 自 / 偏自相關 | `acf(y)`, `pacf(y)` | — |

---

## 六、統計檢定與假設測試（Statistical Tests, `stats`）

| 功能類別 | 功能說明 | 常用函數 / 方法 | 備註 |
|------------|------------|------------------|------|
| 單樣本 t 檢定 | t-test | `ttest_1samp(a, popmean)` | — |
| 雙樣本 t 檢定 | t-test | `ttest_ind(a, b)` | — |
| 成對樣本 t 檢定 | Paired t-test | `ttest_rel(a, b)` | — |
| 方差分析 | ANOVA | `anova_lm(model)` | 需先建立線性模型 |
| 卡方檢定 | Chi-square test | `chisqprob(stat, df)` | 類別資料分析 |
| 正態性檢定 | Normality test | `jarque_bera(resid)`、`normal_ad()` | — |
| 自相關檢定 | Durbin-Watson | `durbin_watson(resid)` | 偵測序列相關 |
| 異方差檢定 | Breusch-Pagan | `het_breuschpagan(resid, exog)` | — |
| 模型共線性 | Variance Inflation Factor | `variance_inflation_factor(X, i)` | 檢查多重共線性 |

---

## 七、多變量分析（Multivariate Analysis）

| 功能類別 | 功能說明 | 常用函數 / 方法 | 備註 |
|------------|------------|------------------|------|
| MANOVA | 多變量方差分析 | `MANOVA.from_formula('y1 + y2 ~ group', data)` | — |
| PCA | 主成分分析 | `PCA(data, ncomp=2).fit()` | 資料降維 |
| 因子分析 | Factor Analysis | `Factor(data, n_factor=3).fit()` | 探索潛在變數 |
| CCA | Canonical Correlation | `CCA(x, y).fit()` | 尋找變數關聯 |

---

## 八、非參數模型（Nonparametric Methods）

| 功能類別 | 功能說明 | 常用函數 / 方法 | 備註 |
|------------|------------|------------------|------|
| Lowess 平滑 | 局部加權回歸 | `nonparametric.lowess(y, x, frac=0.2)` | 非線性趨勢平滑 |
| 核密度估計 | Kernel Density Estimation | `nonparametric.KDEUnivariate(data)` | 類似直方圖平滑化 |
| 離群值偵測 | Robust Regression | `RLM(y, X).fit()` | Robust Linear Model |
| 秩檢定 | Rank-based Test | `rankdata(a)` | 用於非正態資料 |

---

## 九、模型診斷與摘要（Model Diagnostics & Summaries）

| 功能類別 | 功能說明 | 常用函數 / 方法 | 備註 |
|------------|------------|------------------|------|
| 模型摘要 | 顯示完整統計表 | `.summary()` | 含 R²、F-test、AIC、BIC |
| 殘差分析 | 模型誤差分佈 | `.resid`、`.fittedvalues` | — |
| 預測 | 模型預測新資料 | `.predict(new_X)` | — |
| 模型指標 | 模型比較與評估 | `.aic`, `.bic`, `.rsquared_adj` | — |
| 假設檢定 | 統計推論 | `.t_test()`、`.f_test()` | — |

---

## 十、實用工具（Tools & Utilities）

| 功能類別 | 功能說明 | 常用函數 / 方法 | 備註 |
|------------|------------|------------------|------|
| 模型摘要轉表格 | 轉為 DataFrame | `summary2().tables[1]` | 可轉 CSV |
| 信賴區間 | 預測區間 | `.conf_int()` | 預設95% |
| 模型比較 | 比較多模型 | `anova_lm(model1, model2)` | 比較解釋力 |
| 自訂模型 | 繼承 `GenericLikelihoodModel` | 可建立新模型類型 | 進階應用 |

---



---

