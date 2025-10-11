## OpenCV

# 🧩 OpenCV 各項功能與常用函數總覽表  
> **中英對照版教學講義**

---

## 一、影像讀取與顯示（Image I/O）

| 功能類別 | 功能說明 | 常用函數 / 方法 | 備註 |
|------------|------------|------------------|------|
| 讀取影像 | 從檔案載入影像資料 | `cv2.imread(filename, flags)` | `flags`: `cv2.IMREAD_COLOR`, `cv2.IMREAD_GRAYSCALE` |
| 顯示影像 | 在視窗中顯示影像 | `cv2.imshow(winname, image)` | 視窗名稱與影像矩陣 |
| 儲存影像 | 將影像寫入檔案 | `cv2.imwrite(filename, image)` | 支援 JPG、PNG、BMP |
| 錄影擷取 | 從攝影機或影片檔讀取影格 | `cv2.VideoCapture(source)` | 參數 0 表示內建攝影機 |
| 等待鍵盤事件 | 暫停程式直到按下任意鍵 | `cv2.waitKey(delay)` | 常與 `imshow()` 一起使用 |
| 關閉視窗 | 關閉所有 OpenCV 視窗 | `cv2.destroyAllWindows()` | — |

---

## 二、影像資訊與轉換（Image Properties & Conversion）

| 功能類別 | 功能說明 | 常用函數 / 方法 | 備註 |
|------------|------------|------------------|------|
| 影像尺寸 | 取得影像長寬與通道數 | `image.shape` | 回傳 (height, width, channels) |
| 類型轉換 | 改變影像的像素資料型態 | `image.astype(np.uint8)` | — |
| 顏色空間轉換 | RGB↔Gray、BGR↔HSV等 | `cv2.cvtColor(image, code)` | `cv2.COLOR_BGR2GRAY`, `cv2.COLOR_BGR2HSV` |
| 調整大小 | 改變影像尺寸 | `cv2.resize(image, dsize, fx, fy)` | 支援比例縮放 |
| 旋轉 / 翻轉 | 旋轉或鏡像影像 | `cv2.rotate()`、`cv2.flip(image, flipCode)` | flipCode=0垂直、1水平 |

---

## 三、影像處理（Image Processing）

| 功能類別 | 功能說明 | 常用函數 / 方法 | 備註 |
|------------|------------|------------------|------|
| 平滑與模糊 | 降噪與柔化影像 | `cv2.GaussianBlur()`、`cv2.medianBlur()`、`cv2.blur()` | 高斯、均值、中值濾波 |
| 邊緣偵測 | 偵測物體邊界 | `cv2.Canny(image, t1, t2)` | 雙閾值 Canny 邊緣 |
| 銳化 | 增強影像邊緣細節 | `cv2.filter2D(image, -1, kernel)` | 自訂卷積核 |
| 二值化 | 將影像轉為黑白 | `cv2.threshold(image, thresh, maxval, type)` | `cv2.THRESH_BINARY`, `cv2.THRESH_OTSU` |
| 自適應閾值 | 根據區域亮度設定閾值 | `cv2.adaptiveThreshold()` | 適合光照不均的影像 |
| 形態學運算 | 開、閉、膨脹、侵蝕等操作 | `cv2.morphologyEx()`、`cv2.erode()`、`cv2.dilate()` | 常用於降噪與區域處理 |

---

## 四、幾何變換（Geometric Transformations）

| 功能類別 | 功能說明 | 常用函數 / 方法 | 備註 |
|------------|------------|------------------|------|
| 平移 | 影像位移 | `cv2.warpAffine(image, M, dsize)` | `M` 為平移矩陣 |
| 旋轉 | 指定角度旋轉影像 | `cv2.getRotationMatrix2D()` + `warpAffine()` | — |
| 仿射變換 | 旋轉、縮放、平移綜合 | `cv2.getAffineTransform()` + `warpAffine()` | — |
| 透視變換 | 改變拍攝視角 | `cv2.getPerspectiveTransform()` + `warpPerspective()` | 常用於文件矯正 |

---

## 五、影像特徵（Feature Detection & Description）

| 功能類別 | 功能說明 | 常用函數 / 方法 | 備註 |
|------------|------------|------------------|------|
| 角點偵測 | 找出特徵角落 | `cv2.goodFeaturesToTrack()` | Shi-Tomasi 角點 |
| Harris角點 | 經典角點偵測法 | `cv2.cornerHarris()` | 早期常用 |
| SIFT / SURF | 不變特徵抽取 | `cv2.SIFT_create()`、`cv2.SURF_create()` | 對縮放與旋轉具穩定性 |
| ORB特徵 | 快速且免授權限制 | `cv2.ORB_create()` | 常用於實時應用 |
| 特徵匹配 | 比對兩張影像的特徵 | `cv2.BFMatcher()`、`cv2.FlannBasedMatcher()` | — |

---

## 六、物件偵測與輪廓分析（Object Detection & Contours）

| 功能類別 | 功能說明 | 常用函數 / 方法 | 備註 |
|------------|------------|------------------|------|
| 輪廓偵測 | 偵測二值影像的外框 | `cv2.findContours()` | 回傳輪廓座標 |
| 輪廓繪製 | 在影像上描繪輪廓 | `cv2.drawContours()` | 可指定顏色與粗細 |
| 邊界框與凸包 | 對物體建立邊界矩形或凸包 | `cv2.boundingRect()`、`cv2.convexHull()` | 用於物件分析 |
| 霍夫變換 | 偵測線條與圓形 | `cv2.HoughLines()`、`cv2.HoughCircles()` | 常用於車道與圓形物件 |

---

## 七、影片處理（Video Processing）

| 功能類別 | 功能說明 | 常用函數 / 方法 | 備註 |
|------------|------------|------------------|------|
| 讀取影片 | 開啟影片來源 | `cv2.VideoCapture()` | — |
| 取得影格 | 從影片逐格讀取 | `cap.read()` | 回傳 (ret, frame) |
| 寫出影片 | 儲存影片檔案 | `cv2.VideoWriter()` | 需指定編碼器與 FPS |
| 背景相減 | 動態物件偵測 | `cv2.createBackgroundSubtractorMOG2()` | 常用於監控應用 |

---

## 八、影像繪圖與文字（Drawing & Annotation）

| 功能類別 | 功能說明 | 常用函數 / 方法 | 備註 |
|------------|------------|------------------|------|
| 繪製直線 | 在影像上畫線 | `cv2.line(image, pt1, pt2, color, thickness)` | — |
| 繪製矩形 | 畫出框選區域 | `cv2.rectangle()` | — |
| 繪製圓形 | 在影像上畫圓 | `cv2.circle()` | — |
| 多邊形 | 畫出多邊形輪廓 | `cv2.polylines()` | — |
| 加入文字 | 顯示文字註記 | `cv2.putText(image, text, org, font, scale, color)` | 支援多種字型與大小 |

---

## 九、深度學習與神經網路（DNN Module）

| 功能類別 | 功能說明 | 常用函數 / 方法 | 備註 |
|------------|------------|------------------|------|
| 載入模型 | 匯入已訓練的 DNN 模型 | `cv2.dnn.readNetFromONNX()`、`cv2.dnn.readNetFromCaffe()` | 支援多種框架 |
| 影像輸入預處理 | 建立 blob 輸入張量 | `cv2.dnn.blobFromImage()` | 影像正規化與縮放 |
| 模型推論 | 前向傳遞進行預測 | `net.forward()` | 用於分類或偵測 |
| 物件偵測應用 | 常用於 YOLO、SSD、MobileNet | `cv2.dnn.readNet()` | 可結合 GPU 加速 |

---

## 十、攝影幾何與校正（Camera Calibration & Geometry）

| 功能類別 | 功能說明 | 常用函數 / 方法 | 備註 |
|------------|------------|------------------|------|
| 相機標定 | 校正鏡頭畸變與焦距 | `cv2.calibrateCamera()` | 使用棋盤格樣本 |
| 鏡頭校正 | 去除鏡頭畸變 | `cv2.undistort()` | — |
| 立體匹配 | 建立深度圖 | `cv2.StereoBM_create()`、`cv2.StereoSGBM_create()` | 用於3D重建 |
| 投影矩陣 | 對應世界座標與影像座標 | `cv2.projectPoints()` | — |

---

## 十一、其他實用模組（Utilities & Tools）

| 功能類別 | 功能說明 | 常用函數 / 方法 | 備註 |
|------------|------------|------------------|------|
| 計時 | 測量運算時間 | `cv2.getTickCount()`、`cv2.getTickFrequency()` | 常用於效能分析 |
| 滑桿控制 | 建立互動式參數控制 | `cv2.createTrackbar()` | GUI互動功能 |
| 影像混合 | 影像加權疊合 | `cv2.addWeighted()` | 混合兩張影像 |
| 通道操作 | 分離或合併 RGB 通道 | `cv2.split()`、`cv2.merge()` | — |

---

📚 **總結**
> OpenCV（Open Source Computer Vision Library）  
> 是最具代表性的電腦視覺函式庫之一，  
> 支援影像處理、特徵偵測、機器學習、深度學習整合，  
> 並廣泛應用於 **人工智慧、機器視覺、醫療影像、自駕車、監控分析** 等領域。

---


