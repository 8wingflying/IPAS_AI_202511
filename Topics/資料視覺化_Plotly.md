# 📚 **Plotly** 
> **Plotly** 是最強大的互動式資料視覺化框架之一，  
> 提供從 **資料分析 → 儀表板 → 地理資訊 → 3D 視覺化** 的完整工具鏈。  
>  
> 特點包括：
> - 🧠 高互動性（滑鼠懸停、縮放、篩選）  
> - 🌍 多平台支援（Jupyter、Web、Dash、Streamlit）  
> - 🎨 支援 2D / 3D / Mapbox / GeoJSON  
> - 🧩 可與 **Pandas / NumPy / TensorFlow / Scikit-learn** 整合  

---

# 📊 Plotly 各項功能與常用函數總覽表  


---

## 一、Plotly 概要（Overview）

| 模組名稱 | 中文說明 | 功能重點 |
|------------|------------|------------|
| `plotly.express` | 高階繪圖介面（簡潔版） | 一行指令產生互動圖表 |
| `plotly.graph_objects` | 低階圖形物件（高客製） | 建立複合式圖層與動畫 |
| `plotly.subplots` | 子圖管理模組 | 建立多圖版面 |
| `plotly.io` | 輸出與設定模組 | 匯出 HTML、PNG、PDF |
| `plotly.figure_factory` | 特殊統計圖表 | 直方圖、樹狀圖、群集熱圖 |
| `plotly.data` | 內建資料集 | Iris、Gapminder、Tips 等 |
| `dash`（擴展） | Web 應用框架 | 與 Plotly 整合建立儀表板 |

---

## 二、Plotly Express：高階繪圖介面（`plotly.express`）

| 函數名稱 | 圖表類型 | 功能說明 | 主要參數 | 範例 |
|------------|------------|------------|-------------|------|
| `px.scatter()` | 散點圖 | 顯示變數關係 | `x`, `y`, `color`, `size`, `hover_name` | 氣泡圖用途 |
| `px.line()` | 折線圖 | 顯示時間序列 / 趨勢 | `x`, `y`, `color` | — |
| `px.bar()` | 長條圖 | 類別資料比較 | `x`, `y`, `color`, `barmode` | — |
| `px.area()` | 區域圖 | 堆疊趨勢 | `x`, `y`, `color`, `groupnorm` | — |
| `px.histogram()` | 直方圖 | 顯示資料分佈 | `x`, `color`, `nbins`, `barmode` | — |
| `px.box()` | 盒鬚圖 | 顯示統計摘要 | `x`, `y`, `color`, `points` | — |
| `px.violin()` | 小提琴圖 | 顯示密度與離群值 | `x`, `y`, `box=True`, `points='all'` | — |
| `px.density_heatmap()` | 熱點密度圖 | 二維分佈熱度 | `x`, `y`, `nbinsx`, `nbinsy`, `color_continuous_scale` | — |
| `px.density_contour()` | 等高密度圖 | 類似 KDE 線條圖 | `contours_coloring` | — |
| `px.imshow()` | 影像 / 矩陣顯示 | 顯示影像 / 熱圖 | `color_continuous_scale` | — |
| `px.pie()` | 圓餅圖 | 顯示比例結構 | `values`, `names`, `hole` | `hole=0.4` 建立甜甜圈圖 |
| `px.sunburst()` | 層級圓圖 | 類別階層關係 | `path`, `values`, `color` | — |
| `px.treemap()` | 樹狀方塊圖 | 類別比例可視化 | `path`, `values` | — |
| `px.funnel()` | 漏斗圖 | 轉換流程可視化 | `x`, `y`, `color` | — |
| `px.parallel_coordinates()` | 平行座標圖 | 多維度資料比較 | `dimensions`, `color` | — |
| `px.parallel_categories()` | 類別流向圖 | 類別間關係流向 | `dimensions`, `color` | — |
| `px.scatter_3d()` | 3D 散點圖 | 三維資料視覺化 | `x`, `y`, `z`, `color`, `size` | — |
| `px.line_3d()` | 3D 折線圖 | 三維軌跡 / 曲線 | — | — |
| `px.choropleth()` | 區域地圖 | 顯示國家 / 區域值 | `locations`, `color`, `hover_name` | 地理資料常用 |
| `px.scatter_geo()` | 地理散點圖 | 全球地理位置分佈 | `lat`, `lon`, `color`, `size` | — |
| `px.bar_polar()` | 極座標長條圖 | 週期資料視覺化 | `r`, `theta`, `color` | 風向圖 |
| `px.line_polar()` | 極座標折線圖 | 方向性趨勢 | — | — |

---

## 三、Graph Objects：低階物件建構（`plotly.graph_objects`）

| 類別名稱 | 功能說明 | 範例 / 建立方法 |
|------------|------------|------------------|
| `go.Figure()` | 建立圖表物件 | `fig = go.Figure()` |
| `go.Scatter()` | 折線 / 散點資料 | `go.Scatter(x=..., y=..., mode='lines+markers')` |
| `go.Bar()` | 長條圖資料 | `go.Bar(x=..., y=...)` |
| `go.Box()` | 盒鬚圖資料 | `go.Box(y=..., name='A')` |
| `go.Heatmap()` | 熱圖資料 | `go.Heatmap(z=matrix)` |
| `go.Surface()` | 3D 曲面 | `go.Surface(z=z_data)` |
| `go.Mesh3d()` | 3D 網格 | `go.Mesh3d(x=..., y=..., z=...)` |
| `go.Pie()` | 圓餅圖 | `go.Pie(labels=..., values=...)` |
| `go.Indicator()` | KPI 指標儀表 | `go.Indicator(mode='gauge+number', value=70)` |
| `go.Waterfall()` | 瀑布圖 | `go.Waterfall(...)` |
| `go.Contour()` | 等高線圖 | `go.Contour(z=..., colorscale='Viridis')` |
| `go.Candlestick()` | K 線圖 | `go.Candlestick(open=..., high=..., low=..., close=...)` |
| `go.Table()` | 表格 | `go.Table(header=dict(values=[...]), cells=dict(values=[...]))` |

---

## 四、子圖與複合圖（`plotly.subplots`）

| 函數名稱 | 功能說明 | 主要參數 | 範例 |
|------------|------------|-------------|------|
| `make_subplots()` | 建立多子圖版面 | `rows`, `cols`, `shared_xaxes`, `subplot_titles` | `fig = make_subplots(rows=2, cols=2)` |
| `fig.add_trace()` | 加入子圖 | `row`, `col` | `fig.add_trace(go.Scatter(...), row=1, col=2)` |
| `fig.update_layout()` | 全域設定 | `title`, `width`, `height`, `showlegend` | — |
| `fig.update_xaxes()` / `update_yaxes()` | 設定軸範圍與標籤 | `range`, `title` | — |

---

## 五、樣式與互動設定（Style & Interactivity）

| 函數名稱 | 功能說明 | 主要參數 | 備註 |
|------------|------------|-------------|------|
| `fig.update_traces()` | 批次修改圖層屬性 | `marker`, `line`, `opacity` | — |
| `fig.update_layout()` | 修改全域設定 | `title`, `font`, `legend`, `template` | — |
| `fig.update_xaxes()` / `fig.update_yaxes()` | 控制軸屬性 | `range`, `showgrid`, `title` | — |
| `fig.add_annotation()` | 加入註解 | `text`, `x`, `y`, `arrowhead` | — |
| `fig.add_shape()` | 畫線 / 框 | `type='rect'`, `x0`, `x1`, `y0`, `y1` | — |
| `fig.add_image()` | 插入圖片 | `source`, `xref`, `yref`, `sizex` | — |
| `fig.add_hline()` / `fig.add_vline()` | 加水平 / 垂直線 | `x`, `y`, `line_color` | — |
| `fig.add_trace(go.Indicator(...))` | 加入儀表圖層 | — | 用於 KPI 或儀表板 |

---

## 六、顏色與主題模板（Colors & Themes）

| 功能類別 | 功能說明 | 常用設定 / 模板 | 備註 |
|------------|------------|------------------|------|
| 顏色範圍 | 連續變色 | `color_continuous_scale='Viridis' / 'Plasma' / 'Cividis'` | 熱圖常用 |
| 顏色分類 | 離散顏色 | `color_discrete_sequence=px.colors.qualitative.Pastel` | — |
| 主題模板 | 視覺主題 | `'plotly'`, `'seaborn'`, `'ggplot2'`, `'simple_white'`, `'presentation'` | — |
| 字型控制 | 全域字型 | `fig.update_layout(font=dict(family='Arial', size=14))` | — |

---

## 七、3D 圖形（3D Visualization）

| 函數名稱 | 圖表類型 | 功能說明 | 主要參數 |
|------------|------------|------------|-------------|
| `go.Scatter3d()` | 3D 散點圖 | 顯示三維資料點 | `x`, `y`, `z`, `marker` |
| `go.Surface()` | 3D 曲面圖 | 顯示連續表面 | `z`, `colorscale` |
| `go.Mesh3d()` | 3D 多面體 | 三維模型 | `i`, `j`, `k`, `opacity` |
| `go.Volume()` | 體積圖 | 體積密度可視化 | 醫學影像常用 |
| `px.scatter_3d()` / `px.line_3d()` | 快速建立 3D 視圖 | — |

---

## 八、地理與地圖視覺化（Geo & Mapbox）

| 函數名稱 | 圖表類型 | 功能說明 | 主要參數 |
|------------|------------|------------|-------------|
| `px.choropleth()` | 區域填色地圖 | 根據地理區域顯示變數 | `locations`, `color`, `hover_name` |
| `px.scatter_geo()` | 地理散點圖 | 顯示全球地點 | `lat`, `lon`, `color` |
| `px.density_mapbox()` | 地圖密度圖 | 以 Mapbox 底圖呈現熱度 | `lat`, `lon`, `z` |
| `px.line_geo()` | 地理路徑 | 顯示連線（航線） | — |
| `px.choropleth_mapbox()` | Mapbox 地圖填色圖 | `mapbox_style`, `zoom` | 高互動性地圖 |

---

## 九、輸出與整合（Export & Integration）

| 函數名稱 | 功能說明 | 主要參數 | 備註 |
|------------|------------|-------------|------|
| `fig.show()` | 顯示互動圖表 | 支援 Jupyter / Web | — |
| `fig.write_html()` | 輸出為 HTML | `file='chart.html'` | 可嵌入網頁 |
| `fig.write_image()` | 匯出靜態圖 | `format='png'/'pdf'/'svg'` | 需安裝 `kaleido` |
| `pio.show(fig)` | 控制顯示引擎 | — | — |
| `pio.renderers.default` | 設定輸出方式 | `'notebook'`, `'browser'`, `'vscode'` | — |

---

## 十、Dash 與應用擴展（Dash Integration）

| 功能類別 | 功能說明 | 常用模組 / 元件 | 備註 |
|------------|------------|------------------|------|
| 建立 Dash App | 互動式儀表板框架 | `from dash import Dash` | Plotly 官方支援 |
| 輸入元件 | 控制互動 | `dcc.Dropdown`, `dcc.Slider`, `dcc.Input` | — |
| 輸出元件 | 顯示圖表 | `dcc.Graph(figure=fig)` | — |
| 回呼函數 | 即時更新 | `@app.callback()` | — |

---


✅ **典型範例：互動式散點圖**

```python
import plotly.express as px
df = px.data.iris()
fig = px.scatter(df, x='sepal_width', y='sepal_length',
                 color='species', size='petal_length',
                 title='Iris Dataset Visualization')
fig.update_layout(template='plotly_dark')
fig.show()
