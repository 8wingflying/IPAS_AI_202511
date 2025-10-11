# 📚 **Matplotlib** 
> **Matplotlib** 是 Python 最強大的低階繪圖引擎，  
> 支援從 **簡單折線圖 → 多子圖 → 3D → 動畫** 的完整繪圖流程。  
>  
> 特點包括：
> - 🎯 高客製化（從標籤、顏色、字型到多層物件）  
> - 📈 完全整合 **NumPy / Pandas / Seaborn / Plotly**  
> - 🧩 適用於 **EDA、科學模擬、工程圖表與學術論文可視化**  

---

# 📊 Matplotlib 各項功能與常用函數總覽表  


---

## 一、Matplotlib 概要（Overview）

| 模組 / 子套件 | 中文說明 | 功能重點 |
|----------------|------------|------------|
| `matplotlib.pyplot` | 高階繪圖介面 | 提供類似 MATLAB 的繪圖 API |
| `matplotlib.figure` | 圖形物件 | 控制整體 Figure（畫布）層級 |
| `matplotlib.axes` | 子圖軸物件 | 控制坐標軸與繪圖內容 |
| `matplotlib.lines`, `patches` | 圖形元素 | 控制線條、形狀、物件樣式 |
| `matplotlib.animation` | 動畫模組 | 建立動態視覺化效果 |
| `matplotlib.style` | 樣式模組 | 套用預設或自訂視覺風格 |
| `mpl_toolkits.mplot3d` | 三維繪圖模組 | 支援 3D 圖形與視角旋轉 |

---

## 二、基本繪圖指令（Basic Plot Functions）

| 函數名稱 | 圖表類型 | 功能說明 | 主要參數 | 範例 |
|------------|------------|------------|-------------|------|
| `plt.plot()` | 折線圖 | 繪製 x-y 座標連線 | `color`, `linestyle`, `marker`, `label` | 折線圖基礎 |
| `plt.scatter()` | 散點圖 | 顯示離散資料點 | `s`, `c`, `alpha`, `marker` | 分佈觀察 |
| `plt.bar()` | 長條圖 | 顯示類別資料 | `width`, `align`, `color` | 水平版用 `barh()` |
| `plt.hist()` | 直方圖 | 顯示數據分佈 | `bins`, `density`, `color` | 統計分析常用 |
| `plt.pie()` | 圓餅圖 | 顯示比例 | `labels`, `autopct`, `explode` | — |
| `plt.boxplot()` | 盒鬚圖 | 顯示中位數與離群值 | `vert`, `patch_artist` | — |
| `plt.violinplot()` | 小提琴圖 | 結合密度與統計資訊 | `showmeans`, `showmedians` | — |
| `plt.errorbar()` | 誤差圖 | 顯示誤差範圍 | `yerr`, `xerr`, `capsize` | 科學應用 |
| `plt.stem()` | 柱狀脈衝圖 | 顯示離散取樣信號 | — | 信號處理常用 |
| `plt.step()` | 階梯圖 | 顯示離散變化 | `where='mid'` | 時序變化 |

---

## 三、圖像結構與佈局（Figure & Axes Management）

| 函數名稱 | 功能說明 | 主要參數 | 備註 |
|------------|------------|-------------|------|
| `plt.figure()` | 建立畫布（Figure） | `figsize`, `dpi` | 整體圖形容器 |
| `plt.subplots()` | 建立子圖結構 | `nrows`, `ncols`, `sharex`, `sharey` | 常用於多圖佈局 |
| `plt.subplot()` | 單張子圖 | `plt.subplot(2,1,1)` | 快速分圖繪製 |
| `fig.add_subplot()` | 新增子圖 | `(行,列,索引)` | — |
| `fig.add_axes()` | 自訂軸位置 | `[x0, y0, w, h]` | 絕對定位 |
| `plt.tight_layout()` | 自動調整間距 | `pad`, `w_pad`, `h_pad` | 防止重疊 |
| `plt.subplots_adjust()` | 手動控制間距 | `left`, `right`, `top`, `bottom` | 自訂排版 |

---

## 四、標題、軸標籤與註解（Titles, Labels & Annotations）

| 函數名稱 | 功能說明 | 主要參數 | 範例 |
|------------|------------|-------------|------|
| `plt.title()` | 設定標題 | `fontsize`, `loc` | `plt.title("Sales Report")` |
| `plt.xlabel()` / `plt.ylabel()` | 設定軸標籤 | `labelpad`, `fontsize` | — |
| `plt.legend()` | 顯示圖例 | `loc`, `fontsize`, `frameon` | `plt.legend(['A','B'])` |
| `plt.grid()` | 顯示網格線 | `color`, `linestyle`, `alpha` | — |
| `plt.text()` | 插入文字 | `x`, `y`, `s`, `fontsize` | 座標定位文字 |
| `plt.annotate()` | 加入箭頭註解 | `xy`, `xytext`, `arrowprops` | 解釋重點 |
| `plt.axhline()` / `plt.axvline()` | 畫水平 / 垂直線 | `color`, `linestyle` | 對比參考線 |

---

## 五、樣式與顏色（Style & Color）

| 功能類別 | 功能說明 | 常用設定 / 函數 | 備註 |
|------------|------------|------------------|------|
| 顏色設定 | 指定線條或填色 | `color='r'`、`'#FF5733'`、`cmap='viridis'` | RGB / Hex / 名稱 |
| 線條樣式 | 改變線條形狀 | `'-'`, `'--'`, `':'`, `'-.'` | — |
| 標記樣式 | 點型設定 | `'o'`, `'^'`, `'s'`, `'x'`, `'D'` | — |
| 字型與大小 | 全域設定 | `plt.rcParams['font.size'] = 12` | 支援中文設定 |
| 樣式主題 | 預設風格 | `plt.style.use('ggplot')`, `'seaborn'`, `'dark_background'` | — |
| 調色盤 | 內建顏色組合 | `plt.colormaps()` | 影像或熱圖常用 |

---

## 六、進階視覺化（Advanced Plots）

| 函數名稱 | 圖表類型 | 功能說明 | 主要參數 |
|------------|------------|------------|-------------|
| `plt.contour()` | 等高線圖 | 用於 2D 曲面數據 | `levels`, `cmap` |
| `plt.contourf()` | 填色等高圖 | — | `alpha`, `linewidths` |
| `plt.imshow()` | 影像顯示 | 顯示矩陣或影像 | `cmap`, `interpolation` |
| `plt.matshow()` | 矩陣熱圖 | 類似 imshow | — |
| `plt.pcolor()` / `plt.pcolormesh()` | 色塊圖 | 用於網格資料 | `cmap` |
| `plt.quiver()` | 向量場 | 顯示方向性資料 | `U, V` 向量 |
| `plt.streamplot()` | 流線圖 | 動態流體視覺化 | `color`, `density` |
| `ax3d.plot_surface()` | 3D 曲面圖 | `rstride`, `cstride`, `cmap` |
| `ax3d.scatter3D()` | 3D 散點圖 | `depthshade=True` |
| `ax3d.plot_wireframe()` | 3D 網格 | 顯示結構化曲面 |

---

## 七、統計與時間序列繪圖（Statistical & Time Series）

| 函數名稱 | 功能說明 | 主要參數 | 備註 |
|------------|------------|-------------|------|
| `plt.boxplot()` | 盒鬚圖 | `notch`, `vert`, `patch_artist` | 統計摘要 |
| `plt.violinplot()` | 小提琴圖 | `showmeans`, `showmedians` | — |
| `plt.acorr()` / `plt.xcorr()` | 自 / 互相關函數圖 | — | 時序分析 |
| `plt.plot_date()` | 繪製時間序列 | `fmt='-'`, `xdate=True` | 結合日期軸 |
| `plt.fill_between()` | 填色範圍 | `y1`, `y2`, `color`, `alpha` | 信賴區間可視化 |

---

## 八、圖像輸出與控制（Saving & Display）

| 函數名稱 | 功能說明 | 主要參數 | 備註 |
|------------|------------|-------------|------|
| `plt.show()` | 顯示圖形 | — | 結尾指令 |
| `plt.savefig()` | 儲存圖形 | `fname`, `dpi`, `bbox_inches='tight'` | 儲存為 PNG/PDF/SVG |
| `plt.close()` | 關閉圖形視窗 | `plt.close('all')` | 避免記憶體累積 |
| `plt.figure(figsize=(w,h))` | 控制圖大小 | 單位為英吋 | — |

---

## 九、互動與動畫（Interaction & Animation）

| 函數名稱 | 功能說明 | 主要參數 | 備註 |
|------------|------------|-------------|------|
| `FuncAnimation()` | 建立動畫 | `frames`, `interval`, `repeat` | 需搭配更新函數 |
| `ArtistAnimation()` | 快速動畫製作 | — | 以物件清單為基礎 |
| `plt.pause()` | 暫停刷新 | 用於動態更新 | 實時視覺化 |
| `plt.ion()` / `plt.ioff()` | 開啟 / 關閉互動模式 | — | Jupyter 即時繪圖 |

---

## 十、中文與特殊設定（Localization & Config）

| 功能類別 | 功能說明 | 常用設定 / 範例 | 備註 |
|------------|------------|------------------|------|
| 中文字型設定 | 避免亂碼 | `plt.rcParams['font.sans-serif'] = ['Taipei Sans TC Beta']` | Windows: "Microsoft JhengHei" |
| 負號顯示修正 | 防止負號變方框 | `plt.rcParams['axes.unicode_minus'] = False` | — |
| 全域設定 | 儲存樣式偏好 | `matplotlibrc` 檔案 | — |

---



✅ **典型範例：多子圖折線圖**

```python
import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0, 10, 100)
y1, y2 = np.sin(x), np.cos(x)

fig, ax = plt.subplots(2, 1, figsize=(6, 6))
ax[0].plot(x, y1, label="sin(x)", color='blue')
ax[1].plot(x, y2, label="cos(x)", color='red', linestyle='--')

for a in ax:
    a.legend()
    a.grid(True)

plt.suptitle("Trigonometric Functions")
plt.tight_layout()
plt.show()
