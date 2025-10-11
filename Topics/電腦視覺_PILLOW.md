##📚 Pillow（Python Imaging Library, PIL
> Pillow 是 Python 影像處理的標準函式庫，  
> 支援 **影像載入、轉換、濾鏡、繪圖、批次處理與影像生成** 等功能。  
> 它與 NumPy、OpenCV、Matplotlib 等套件結合後，  
> 可廣泛應用於 **影像分析、AI 模型前處理、資料增強、視覺化生成** 等場景。

# 🖼️ Pillow功能與常用函數總覽表  

---

## 一、影像載入與儲存（Image I/O）

| 功能類別 | 功能說明 | 常用函數 / 方法 | 備註 |
|------------|------------|------------------|------|
| 讀取影像 | 從檔案載入影像物件 | `Image.open(filepath)` | 回傳 `PIL.Image.Image` 物件 |
| 儲存影像 | 將影像寫入檔案 | `image.save(filepath, format)` | 支援 PNG、JPEG、BMP、TIFF 等格式 |
| 影像顯示 | 使用預設影像檢視器開啟 | `image.show()` | 快速預覽結果 |
| 影像複製 | 建立影像副本 | `image.copy()` | 常用於非破壞性編輯 |

---

## 二、影像屬性與資訊（Image Properties）

| 功能類別 | 功能說明 | 常用屬性 / 方法 | 備註 |
|------------|------------|------------------|------|
| 影像尺寸 | 取得影像大小（寬、高） | `image.size` | 回傳 `(width, height)` |
| 模式資訊 | 影像顏色模式 | `image.mode` | 常見模式：`RGB`, `RGBA`, `L`, `CMYK` |
| 檔案格式 | 檔案類型（副檔名） | `image.format` | 例如 `JPEG`, `PNG` |
| 像素存取 | 讀取 / 修改像素 | `image.getpixel((x,y))`、`image.putpixel((x,y), value)` | — |

---

## 三、影像轉換（Image Conversion）

| 功能類別 | 功能說明 | 常用函數 / 方法 | 備註 |
|------------|------------|------------------|------|
| 顏色轉換 | 模式轉換，如灰階 | `image.convert(mode)` | 例：`image.convert('L')` 轉灰階 |
| 影像旋轉 | 按角度旋轉影像 | `image.rotate(angle, expand=True)` | `expand=True` 自動調整大小 |
| 翻轉影像 | 水平或垂直翻轉 | `ImageOps.flip()`、`ImageOps.mirror()` | 水平為 mirror，垂直為 flip |
| 調整大小 | 改變影像尺寸 | `image.resize((w, h), resample)` | 可指定 `Image.BICUBIC` |
| 裁切區域 | 擷取指定範圍 | `image.crop(box)` | `box=(left, top, right, bottom)` |

---

## 四、影像增強與調整（Enhancement & Adjustment）

| 功能類別 | 功能說明 | 常用模組 / 方法 | 備註 |
|------------|------------|------------------|------|
| 亮度調整 | 改變影像亮度 | `ImageEnhance.Brightness(image).enhance(factor)` | factor=1.0 為原亮度 |
| 對比度調整 | 改變影像對比度 | `ImageEnhance.Contrast(image).enhance(factor)` | — |
| 銳利化 | 強化影像邊緣 | `ImageEnhance.Sharpness(image).enhance(factor)` | — |
| 顏色飽和度 | 調整色彩強度 | `ImageEnhance.Color(image).enhance(factor)` | — |
| 自動對比 | 自動調整亮度與對比 | `ImageOps.autocontrast(image)` | — |

---

## 五、影像濾鏡與特效（Filtering & Effects）

| 功能類別 | 功能說明 | 常用函數 / 濾鏡 | 備註 |
|------------|------------|------------------|------|
| 模糊濾鏡 | 影像柔化 | `image.filter(ImageFilter.BLUR)` | — |
| 邊緣增強 | 強化輪廓邊緣 | `image.filter(ImageFilter.EDGE_ENHANCE)` | — |
| 輪廓偵測 | 偵測物體邊界 | `image.filter(ImageFilter.FIND_EDGES)` | — |
| 銳利化濾鏡 | 增加清晰度 | `image.filter(ImageFilter.SHARPEN)` | — |
| 自訂濾鏡 | 使用卷積核 | `image.filter(ImageFilter.Kernel(size, kernel))` | 可自行設計效果 |

---

## 六、幾何操作（Geometric Transformations）

| 功能類別 | 功能說明 | 常用函數 / 方法 | 備註 |
|------------|------------|------------------|------|
| 平移 / 仿射 | 影像幾何變換 | `image.transform(size, method, data)` | 支援 `AFFINE`, `PERSPECTIVE` |
| 翻轉與旋轉 | 快速方向操作 | `image.transpose()` | 參數如 `Image.ROTATE_90`, `Image.FLIP_LEFT_RIGHT` |
| 拼貼合成 | 合併多張影像 | `Image.new()` + `paste()` | — |
| 裁邊填色 | 增加邊框 | `ImageOps.expand(image, border, fill)` | — |

---

## 七、繪圖與文字（Drawing & Text）

| 功能類別 | 功能說明 | 常用模組 / 方法 | 備註 |
|------------|------------|------------------|------|
| 新增畫布 | 建立空白影像 | `Image.new(mode, size, color)` | — |
| 繪製形狀 | 畫線、矩形、圓形等 | `ImageDraw.Draw(image)` | 提供 `line()`, `rectangle()`, `ellipse()` |
| 加入文字 | 在影像上輸出文字 | `draw.text((x,y), text, fill, font)` | 需載入字型 |
| 載入字型 | 使用 TrueType 字型 | `ImageFont.truetype(path, size)` | 常搭配中文字體 |

---

## 八、通道與色彩操作（Channel & Color Operations）

| 功能類別 | 功能說明 | 常用函數 / 方法 | 備註 |
|------------|------------|------------------|------|
| 分離通道 | 分離 R、G、B 通道 | `r, g, b = image.split()` | 回傳三張灰階影像 |
| 合併通道 | 合併多個通道影像 | `Image.merge('RGB', (r, g, b))` | — |
| 灰階轉換 | RGB → 灰階 | `image.convert('L')` | — |
| 色彩反轉 | 將顏色取反 | `ImageOps.invert(image)` | 適用於灰階或RGB影像 |

---

## 九、批次處理與自動化（Batch & Automation）

| 功能類別 | 功能說明 | 常用技巧 / 函數 | 備註 |
|------------|------------|------------------|------|
| 批次讀檔 | 迭代資料夾影像 | `os.listdir()` + `Image.open()` | 常見於資料前處理 |
| 批次轉檔 | 轉換影像格式 | 迴圈 + `save()` | JPEG → PNG 轉換 |
| 縮圖處理 | 批量壓縮影像 | `image.thumbnail(size)` | 自動保持比例 |
| 影像拼接 | 水平或垂直排列 | `Image.new()` + `paste()` | 可建立照片牆效果 |

---

## 十、其他實用模組（Utilities & Helpers）

| 功能類別 | 功能說明 | 常用模組 / 方法 | 備註 |
|------------|------------|------------------|------|
| Exif 資訊 | 讀取照片拍攝資訊 | `image._getexif()` | 包含曝光時間、ISO等 |
| Alpha 混合 | 影像透明疊合 | `Image.blend(img1, img2, alpha)` | alpha=0.5 混合比例 |
| 合成遮罩 | 使用透明通道疊圖 | `image.paste(img, mask=mask_img)` | — |
| 降噪與濾鏡混合 | 結合 OpenCV / NumPy 使用 | `np.array(image)` ↔ `Image.fromarray()` | 可與 OpenCV 混合運算 |

---



---

