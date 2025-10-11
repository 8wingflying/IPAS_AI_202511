# 📚 **Seaborn** 
> **Seaborn** 是建立在 **Matplotlib + Pandas** 之上的高階統計視覺化工具。  
> 它支援：
> - 自動統計匯總（如平均值、信賴區間）  
> - 分群顏色（hue）、行列分面（col, row）  
> - 內建樣式與調色盤  
> - 高度整合資料科學工作流程（EDA, ML 前分析）  
>  
> 廣泛應用於 **探索性資料分析（EDA）**、**機器學習前資料理解**、**報告可視化**。

---

# 🎨 Seaborn 各項功能與常用函數總覽表  


---

## 一、Seaborn 概要（Overview）

| 功能模組 | 中文說明 | 功能重點 |
|------------|------------|------------|
| `seaborn.relational` | 關聯圖模組 | 散點圖、折線圖（變數關係） |
| `seaborn.categorical` | 類別圖模組 | 條狀圖、盒鬚圖、群組比較 |
| `seaborn.distributions` | 分佈圖模組 | 直方圖、密度圖、KDE、ECDF |
| `seaborn.matrix` | 矩陣圖模組 | 熱圖、相關係數矩陣 |
| `seaborn.regression` | 迴歸圖模組 | 線性迴歸、區間信賴可視化 |
| `seaborn.axisgrid` | 複合繪圖模組 | FacetGrid, PairGrid 多子圖組合 |
| `seaborn.theme` | 視覺主題 | 改變整體風格與配色 |
| `seaborn.color_palette` | 顏色控制 | 定義調色盤 |
| `seaborn.objects` | 高階物件導向 API | Seaborn v0.12+ 新增的統一繪圖介面 |

---

## 二、關聯圖（Relational Plots）

| 函數名稱 | 圖表類型 | 功能說明 | 主要參數 | 範例 |
|------------|------------|------------|-------------|------|
| `sns.scatterplot()` | 散點圖 | 顯示變數間關係 | `x`, `y`, `hue`, `style`, `size` | 數值 vs 數值 |
| `sns.lineplot()` | 折線圖 | 顯示時間序列或連續關係 | `x`, `y`, `hue`, `estimator`, `ci` | 適用趨勢觀察 |
| `sns.relplot()` | 統一 API（scatter / line） | 可用 `kind='scatter'` 或 `'line'` | `col`, `row` | 可建立 Facet 多子圖 |

---

## 三、分佈圖（Distribution Plots）

| 函數名稱 | 圖表類型 | 功能說明 | 主要參數 | 備註 |
|------------|------------|------------|-------------|------|
| `sns.histplot()` | 直方圖 | 顯示資料分布頻率 | `bins`, `kde`, `hue` | 支援多群體比較 |
| `sns.kdeplot()` | 核密度估計圖 | 平滑分佈曲線 | `fill`, `bw_adjust`, `multiple` | — |
| `sns.ecdfplot()` | 累積分佈函數 | 顯示 CDF 曲線 | `stat`, `complementary` | — |
| `sns.distplot()` *(舊版)* | 混合分佈圖 | 已被 `histplot()` 取代 | — | 不建議新專案使用 |
| `sns.displot()` | 統一 API（hist / kde / ecdf） | 可指定 `kind` | `col`, `row`, `aspect` | 可建立多子圖 |

---

## 四、類別圖（Categorical Plots）

| 函數名稱 | 圖表類型 | 功能說明 | 主要參數 | 備註 |
|------------|------------|------------|-------------|------|
| `sns.barplot()` | 條狀圖（平均） | 顯示群組平均值與信賴區間 | `estimator`, `ci`, `hue` | 自動計算誤差 |
| `sns.countplot()` | 計數圖 | 顯示每類別樣本數 | `hue`, `order` | — |
| `sns.boxplot()` | 盒鬚圖 | 顯示中位數與離群值 | `hue`, `orient`, `palette` | 用於分群比較 |
| `sns.violinplot()` | 小提琴圖 | 顯示密度 + 統計摘要 | `split=True`, `inner='quartile'` | — |
| `sns.stripplot()` | 點狀圖 | 類別內隨機分佈樣本點 | `jitter`, `alpha` | — |
| `sns.swarmplot()` | 擁擠點圖 | 類別點自動分佈避免重疊 | — | 視覺清晰版 stripplot |
| `sns.catplot()` | 統一 API | 支援 bar/box/violin/strip 等 | `kind='bar'/'box'/...` | FacetGrid 版本 |

---

## 五、迴歸與關係圖（Regression Plots）

| 函數名稱 | 圖表類型 | 功能說明 | 主要參數 | 範例 |
|------------|------------|------------|-------------|------|
| `sns.regplot()` | 單變量線性迴歸圖 | 顯示散點與回歸線 | `x`, `y`, `order`, `ci` | — |
| `sns.lmplot()` | 線性迴歸多子圖 | FacetGrid 版本 | `hue`, `col`, `row` | 可分群繪製 |
| `sns.residplot()` | 殘差圖 | 顯示預測誤差 | `x`, `y`, `lowess` | 模型診斷用 |
| `sns.regressionplots` | 模組別名 | 含所有 `regplot/lmplot` 函數 | — | — |

---

## 六、矩陣與相關分析圖（Matrix & Correlation Plots）

| 函數名稱 | 圖表類型 | 功能說明 | 主要參數 | 備註 |
|------------|------------|------------|-------------|------|
| `sns.heatmap()` | 熱圖 | 顯示矩陣或相關係數強度 | `annot`, `cmap`, `vmin`, `vmax` | 常用於 `df.corr()` |
| `sns.clustermap()` | 層次叢集熱圖 | 類似 heatmap 並附層次聚類樹 | `method`, `metric`, `standard_scale` | — |
| `sns.matrixplot` | 模組別名 | 提供矩陣視覺化 API | — | — |

---

## 七、多變量關係圖（Multivariate / Grid Plots）

| 函數名稱 | 圖表類型 | 功能說明 | 主要參數 | 備註 |
|------------|------------|------------|-------------|------|
| `sns.pairplot()` | 成對關係圖 | 繪製所有變數的散點與分佈 | `hue`, `kind`, `diag_kind` | 常用於 EDA |
| `sns.jointplot()` | 雙變量分佈圖 | 顯示散點 + 邊際分佈 | `kind='scatter'/'kde'/'hex'` | — |
| `sns.PairGrid()` | 高階成對繪圖控制 | 可自訂上/下三角繪法 | `.map_upper()`, `.map_lower()` | 高彈性繪法 |
| `sns.FacetGrid()` | 多子圖繪圖架構 | 依欄位自動分群繪圖 | `.map()`, `.add_legend()` | 支援任意函數 |
| `sns.JointGrid()` | 雙變量繪圖架構 | 可自訂 joint/marginal 類型 | `.plot_joint()`, `.plot_marginals()` | — |

---

## 八、主題與風格設定（Themes & Styles）

| 函數名稱 | 功能說明 | 常用選項 / 參數 | 備註 |
|------------|------------|------------------|------|
| `sns.set_style()` | 設定整體風格 | `'white'`, `'dark'`, `'ticks'`, `'whitegrid'`, `'darkgrid'` | — |
| `sns.set_context()` | 調整字體與大小 | `'paper'`, `'notebook'`, `'talk'`, `'poster'` | 適合簡報或出版 |
| `sns.set_palette()` | 改變色盤 | `'deep'`, `'muted'`, `'bright'`, `'pastel'`, `'dark'` | — |
| `sns.color_palette()` | 取得顏色列表 | `sns.color_palette('coolwarm', 10)` | 用於自訂繪圖 |
| `sns.set_theme()` | 一次設定所有樣式 | `style`, `context`, `palette` | v0.11+ 推薦使用 |

---

## 九、顏色控制與美學（Color Palettes & Aesthetics）

| 函數名稱 | 功能說明 | 常用參數 | 備註 |
|------------|------------|-------------|------|
| `sns.palplot()` | 顯示色盤 | `sns.palplot(sns.color_palette('husl'))` | 顏色視覺化 |
| `sns.cubehelix_palette()` | 單色漸層 | `start`, `rot`, `dark`, `light` | 適合熱圖 |
| `sns.light_palette()` | 明亮色系 | `as_cmap=True` | — |
| `sns.dark_palette()` | 暗色系 | `reverse=True` | — |
| `sns.diverging_palette()` | 雙色漸層 | 例如：紅藍 | — |

---

## 十、高階物件導向繪圖（`seaborn.objects` — v0.12+）

| 功能類別 | 功能說明 | 函數 / 類別 | 範例語法 |
|------------|------------|------------------|-------------|
| 資料物件 | 建立 Seaborn 繪圖物件 | `sns.objects.Plot(data, x, y)` | `sns.objects.Plot(df, x='A', y='B').add(sns.objects.Point())` |
| 幾何圖層 | 加入繪圖元素 | `.add(sns.objects.Bar())`, `.add(Line())` | 類似 ggplot2 |
| 美學設定 | 控制顏色、樣式 | `.scale(color='category')` | — |
| 分面繪圖 | 拆分子圖 | `.facet(col='species')` | — |

---

## 十一、輔助功能與整合（Utilities & Integration）

| 功能類別 | 功能說明 | 常用函數 / 方法 | 備註 |
|------------|------------|------------------|------|
| 資料框整合 | 與 Pandas 無縫整合 | `sns.load_dataset('iris')` | 回傳 DataFrame |
| 圖表儲存 | 輸出成圖檔 | `plt.savefig('plot.png', dpi=300)` | 搭配 Matplotlib |
| 整合 NumPy | 可直接繪製 ndarray | `sns.histplot(np.random.randn(100))` | — |
| 與 matplotlib 互通 | 可使用 `plt.figure()`、`plt.subplot()` | — | — |

---

## 十二、典型繪圖流程（Quick Reference）

| 任務 | 對應函數 | 說明 |
|------|-----------|------|
| 顯示變數分佈 | `sns.histplot()`、`sns.kdeplot()` | 單變量分佈分析 |
| 類別比較 | `sns.boxplot()`、`sns.barplot()` | 比較群組間差異 |
| 關係探索 | `sns.scatterplot()`、`sns.lmplot()` | 兩變數線性關係 |
| 相關性分析 | `sns.heatmap(df.corr(), annot=True)` | 顯示變數關聯程度 |
| 多變量探索 | `sns.pairplot(df, hue='species')` | EDA 探索性分析 |
| 時間序列 | `sns.lineplot()` | 趨勢變化 |
| 統一視覺風格 | `sns.set_theme(style='whitegrid', palette='muted')` | 全局樣式控制 |

---



✅ **典型使用範例：**

```python
import seaborn as sns
import matplotlib.pyplot as plt
df = sns.load_dataset("tips")

sns.set_theme(style="whitegrid", palette="muted")
sns.boxplot(x="day", y="total_bill", hue="sex", data=df)
sns.despine()  # 移除上、右邊界線
plt.title("Daily Total Bill Distribution by Gender")
plt.show()
