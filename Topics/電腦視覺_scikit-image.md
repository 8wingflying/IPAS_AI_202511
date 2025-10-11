# 📚 `scikit-image`
> `scikit-image` 是基於 NumPy / SciPy 的影像處理函式庫，  
> 功能涵蓋 **讀取、濾波、分割、特徵偵測、形態學、重建、復原與視覺化**。  
> 它是 Python 影像科學處理中最常與 **OpenCV、Pillow、matplotlib、NumPy** 搭配使用的套件之一，  
> 在 **電腦視覺、醫學影像、影像分析與AI資料前處理** 中皆有廣泛應用。



# 🧠 scikit-image 各項功能與常用函數總覽表  

---

## 一、影像讀取與輸出（Image I/O）

| 功能類別 | 英文模組 | 功能說明 | 常用函數 / 方法 | 備註 |
|------------|-----------|------------|------------------|------|
| 影像讀取 | `skimage.io` | 從檔案載入影像至 NumPy 陣列 | `io.imread(path)` | 回傳 ndarray |
| 顯示影像 | `skimage.io` | 顯示影像（使用 matplotlib） | `io.imshow(image)`、`io.show()` | — |
| 儲存影像 | `skimage.io` | 將影像寫入檔案 | `io.imsave(path, image)` | 支援 PNG, JPEG 等格式 |
| 視訊擷取 | `skimage.io` | 讀取影片影格序列 | `io.imread_collection()` | — |

---

## 二、顏色與資料轉換（Color & Data Conversion）

| 功能類別 | 英文模組 | 功能說明 | 常用函數 / 方法 | 備註 |
|------------|-----------|------------|------------------|------|
| 顏色空間轉換 | `skimage.color` | RGB ↔ Gray, HSV, LAB 等 | `rgb2gray()`、`rgb2hsv()`、`rgb2lab()` | 常用於前處理 |
| 反向轉換 | `skimage.color` | Gray / HSV / LAB → RGB | `gray2rgb()`、`lab2rgb()` | — |
| 浮點 / 整數轉換 | `skimage.util` | 改變影像資料型態 | `img_as_float()`、`img_as_ubyte()` | 自動正規化 |
| 色彩歸一化 | `skimage.exposure` | 調整亮度、對比、Gamma | `rescale_intensity()`、`adjust_gamma()` | — |

---

## 三、幾何變換（Geometric Transformations）

| 功能類別 | 英文模組 | 功能說明 | 常用函數 / 方法 | 備註 |
|------------|-----------|------------|------------------|------|
| 影像縮放 | `skimage.transform` | 調整影像尺寸 | `resize(image, output_shape)` | 預設抗鋸齒 |
| 旋轉影像 | `skimage.transform` | 按角度旋轉影像 | `rotate(image, angle)` | — |
| 平移 / 仿射變換 | `skimage.transform` | 幾何變換 | `warp(image, AffineTransform(...))` | — |
| 透視校正 | `skimage.transform` | 對應四點透視轉換 | `ProjectiveTransform()` | 可用於文件矯正 |
| 下取樣 / 重採樣 | `skimage.transform` | 降低解析度 | `pyramid_reduce()`、`rescale()` | — |

---

## 四、濾波與雜訊處理（Filtering & Denoising）

| 功能類別 | 英文模組 | 功能說明 | 常用函數 / 方法 | 備註 |
|------------|-----------|------------|------------------|------|
| 高斯模糊 | `skimage.filters` | 平滑影像以去除雜訊 | `gaussian(image, sigma)` | σ 控制模糊程度 |
| 中值濾波 | `skimage.filters.rank` | 去除椒鹽雜訊 | `median(image, selem)` | — |
| Sobel 邊緣 | `skimage.filters` | 偵測邊緣梯度 | `sobel(image)`、`scharr()` | — |
| 拉普拉斯濾波 | `skimage.filters` | 偵測高頻細節 | `laplace(image)` | — |
| 雜訊生成 | `skimage.util` | 加入模擬雜訊 | `random_noise(image, mode='gaussian')` | 用於測試 |
| 非區域均值去噪 | `skimage.restoration` | 高品質去雜訊 | `denoise_nl_means()` | 保留紋理細節 |

---

## 五、邊緣偵測與特徵分析（Edge & Feature Detection）

| 功能類別 | 英文模組 | 功能說明 | 常用函數 / 方法 | 備註 |
|------------|-----------|------------|------------------|------|
| Canny 邊緣偵測 | `skimage.feature` | 偵測邊緣 | `canny(image, sigma)` | 常用於物件辨識 |
| 角點偵測 | `skimage.feature` | 找出 Harris / Shi-Tomasi 角點 | `corner_harris()`、`corner_peaks()` | — |
| HOG 特徵 | `skimage.feature` | 提取方向梯度直方圖 | `hog(image)` | 應用於分類與辨識 |
| 區域特徵 | `skimage.feature` | 計算局部特徵描述子 | `local_binary_pattern()` | 紋理分析 |
| 影像模板匹配 | `skimage.feature` | 搜尋目標區域 | `match_template(image, template)` | — |

---

## 六、影像分割（Segmentation）

| 功能類別 | 英文模組 | 功能說明 | 常用函數 / 方法 | 備註 |
|------------|-----------|------------|------------------|------|
| 閾值分割 | `skimage.filters` | 基於亮度閾值分割 | `threshold_otsu()`、`threshold_li()` | 自動找閾值 |
| 分水嶺算法 | `skimage.segmentation` | 基於區域生長分割 | `watershed(image)` | 適用於細胞或顆粒影像 |
| SLIC 超像素 | `skimage.segmentation` | 超像素區域分割 | `slic(image, n_segments)` | 應用於影像壓縮 |
| 邊界遮罩 | `skimage.segmentation` | 視覺化邊界 | `mark_boundaries(image, segments)` | 顯示分割結果 |
| 主動輪廓 | `skimage.segmentation` | 動態輪廓模型 | `active_contour()` | 物件邊界追蹤 |

---

## 七、形態學運算（Morphology）

| 功能類別 | 英文模組 | 功能說明 | 常用函數 / 方法 | 備註 |
|------------|-----------|------------|------------------|------|
| 膨脹 / 侵蝕 | `skimage.morphology` | 二值影像區域擴張 / 收縮 | `dilation()`、`erosion()` | 基本操作 |
| 開 / 閉運算 | `skimage.morphology` | 去除雜訊、填補孔洞 | `opening()`、`closing()` | — |
| 骨架化 | `skimage.morphology` | 提取形狀主軸線 | `skeletonize()` | 應用於筆跡分析 |
| 區域填補 | `skimage.morphology` | 填補物件內部空洞 | `remove_small_holes()` | — |
| 結構元素 | `skimage.morphology` | 定義形態操作核 | `disk()`, `square()`, `rectangle()` | — |

---

## 八、影像測量與區域屬性（Region Properties & Measurement）

| 功能類別 | 英文模組 | 功能說明 | 常用函數 / 方法 | 備註 |
|------------|-----------|------------|------------------|------|
| 連通區域標記 | `skimage.measure` | 標記不同區域 | `label(image)` | 回傳標籤矩陣 |
| 區域屬性 | `skimage.measure` | 計算區域面積、周長、慣性等 | `regionprops(label_image)` | 回傳屬性字典 |
| 輪廓提取 | `skimage.measure` | 找出等值線 | `find_contours(image, level)` | — |
| 邊界框 | `skimage.measure` | 計算物件外接矩形 | `regionprops_table()` | 輸出 DataFrame 格式 |

---

## 九、復原與重建（Restoration & Reconstruction）

| 功能類別 | 英文模組 | 功能說明 | 常用函數 / 方法 | 備註 |
|------------|-----------|------------|------------------|------|
| 反卷積 | `skimage.restoration` | 去除模糊或運動模糊 | `richardson_lucy()` | 常用於天文影像 |
| 全變分去噪 | `skimage.restoration` | 保邊緣降噪 | `denoise_tv_chambolle()` | — |
| Poisson 重建 | `skimage.restoration` | 修補局部區域影像 | `inpaint_biharmonic()` | — |
| 權重去模糊 | `skimage.restoration` | 盲去模糊算法 | `unsupervised_wiener()` | — |

---

## 十、其他實用模組（Utilities & Visualization）

| 功能類別 | 英文模組 | 功能說明 | 常用函數 / 方法 | 備註 |
|------------|-----------|------------|------------------|------|
| 直方圖繪製 | `skimage.exposure` | 顯示亮度分布 | `histogram(image)`、`equalize_hist()` | — |
| 對比度拉伸 | `skimage.exposure` | 增強影像對比度 | `rescale_intensity()` | — |
| 隨機影像產生 | `skimage.data` | 內建測試影像 | `data.camera()`、`data.coins()` | 教學常用 |
| 區域視覺化 | `skimage.segmentation` | 顯示分割邊界 | `mark_boundaries()` | — |
| 資料格式轉換 | `skimage.util` | ndarray 與浮點 / 整數間轉換 | `img_as_ubyte()`、`img_as_float()` | — |

---



