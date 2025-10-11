# 📚 **SciPy（Scientific Python）** 
> **SciPy（Scientific Python）** 是 Python 科學運算的核心套件之一，  
> 基於 **NumPy**，整合了 **數值積分、最佳化、線性代數、統計分析、訊號與影像處理** 等模組。  
> 它被廣泛應用於 **AI建模、工程模擬、物理計算、生物資訊、金融數據分析** 等領域。

# ⚙️ SciPy 各項功能與常用函數總覽表  


---

## 一、SciPy 概要（Overview）:重要模組

| 模組 | 中文說明 | 功能重點 |
|------|-----------|-----------|
| `scipy.cluster` | 叢集分析 | k-means、層次式分群 |
| `scipy.constants` | 科學常數 | 提供物理與數學常數 |
| `scipy.fft` / `scipy.fftpack` | 傅立葉轉換 | 頻譜分析與訊號處理 |
| `scipy.integrate` | 積分與微分方程 | 數值積分、ODE求解 |
| `scipy.interpolate` | 插值法 | 一維與多維插值 |
| `scipy.io` | 檔案輸入輸出 | MATLAB、WAV、CSV等格式 |
| `scipy.linalg` | 線性代數 | 矩陣分解、特徵值、逆矩陣 |
| `scipy.ndimage` | 影像與多維資料處理 | 濾波、邊緣、旋轉、卷積 |
| `scipy.optimize` | 最佳化 | 極值、最小化、曲線擬合 |
| `scipy.signal` | 訊號處理 | 濾波器設計、頻譜分析 |
| `scipy.sparse` | 稀疏矩陣 | 優化儲存與運算大型矩陣 |
| `scipy.spatial` | 空間與幾何運算 | KD-Tree、距離、凸包 |
| `scipy.special` | 特殊函數 | Gamma、Beta、Bessel 等函數 |
| `scipy.stats` | 統計分析 | 機率分布、假設檢定、迴歸分析 |

---

## 二、科學常數模組（`scipy.constants`）

| 功能類別 | 功能說明 | 常用函數 / 參數 | 備註 |
|------------|------------|------------------|------|
| 物理常數 | 取得科學常數 | `constants.pi`, `constants.g`, `constants.h`, `constants.c` | 圓周率、重力常數、普朗克常數、光速 |
| 單位轉換 | 單位間換算 | `constants.convert_temperature(val, 'C', 'K')` | 攝氏 ↔ 開氏 |
| 常數查詢 | 搜尋常數 | `dir(constants)` | 顯示所有常數 |

---

## 三、線性代數模組（`scipy.linalg`）

| 功能類別 | 功能說明 | 常用函數 / 方法 | 備註 |
|------------|------------|------------------|------|
| 矩陣乘法 | 矩陣內積 | `linalg.blas.dgemm(a, b)` 或 `np.dot(a,b)` | — |
| 反矩陣 | 計算逆矩陣 | `linalg.inv(a)` | — |
| 行列式 | 計算 determinant | `linalg.det(a)` | — |
| 解線性方程 | 求解 Ax = b | `linalg.solve(A, b)` | — |
| 特徵值分解 | 求 eigenvalue / vector | `linalg.eig(a)` | — |
| 奇異值分解 | SVD 分解 | `linalg.svd(a)` | 用於降維 |
| QR / LU 分解 | 矩陣分解 | `linalg.qr(a)`、`linalg.lu(a)` | — |
| 範數計算 | 向量 / 矩陣範數 | `linalg.norm(a, ord=2)` | — |

---

## 四、積分與微分方程（`scipy.integrate`）

| 功能類別 | 功能說明 | 常用函數 / 方法 | 備註 |
|------------|------------|------------------|------|
| 定積分 | 計算 f(x) 的積分 | `integrate.quad(func, a, b)` | 回傳積分值與誤差 |
| 二重積分 | 二維積分 | `integrate.dblquad(func, ax, bx, ay, by)` | — |
| 多重積分 | 多變數積分 | `integrate.nquad(func, ranges)` | — |
| 常微分方程 | ODE 求解 | `integrate.solve_ivp(f, t_span, y0)` | 支援 Runge-Kutta |
| 數值積分 | 利用樣本積分 | `integrate.simps(y, x)`、`trapz(y, x)` | Simpson / Trapezoidal |

---

## 五、最佳化模組（`scipy.optimize`）

| 功能類別 | 功能說明 | 常用函數 / 方法 | 備註 |
|------------|------------|------------------|------|
| 單變數極值 | 函數最小化 | `optimize.minimize_scalar(f)` | — |
| 多變數極值 | 多維最小化 | `optimize.minimize(f, x0)` | 支援 BFGS, Nelder-Mead |
| 曲線擬合 | 最小平方擬合 | `optimize.curve_fit(func, xdata, ydata)` | 回傳參數與誤差 |
| 根求解 | 求方程式根 | `optimize.root(f, x0)` | — |
| 約束最佳化 | 含邊界條件 | `optimize.minimize(f, bounds=...)` | 用於經濟學模型 |

---

## 六、統計與機率（`scipy.stats`）

| 功能類別 | 功能說明 | 常用函數 / 方法 | 備註 |
|------------|------------|------------------|------|
| 機率分布 | 各類連續與離散分布 | `stats.norm`, `stats.poisson`, `stats.uniform` | 可抽樣與求 PDF |
| 機率密度 | 機率分布函數 | `.pdf(x)`, `.cdf(x)` | 機率密度與累積分布 |
| 抽樣 | 隨機生成樣本 | `.rvs(size=n)` | — |
| 假設檢定 | t檢定、卡方檢定等 | `ttest_ind(a,b)`, `chi2_contingency()` | 統計推論 |
| 相關係數 | 變數關係分析 | `pearsonr(x,y)`, `spearmanr(x,y)` | — |
| 敘述統計 | 平均、變異、峰度 | `describe(data)`、`skew()`、`kurtosis()` | — |

---

## 七、訊號處理（`scipy.signal`）

| 功能類別 | 功能說明 | 常用函數 / 方法 | 備註 |
|------------|------------|------------------|------|
| 濾波器設計 | FIR / IIR 濾波器 | `butter()`, `cheby1()`, `firwin()` | 濾波器係數生成 |
| 濾波運算 | 套用濾波器 | `lfilter(b,a,x)`、`filtfilt()` | 時域濾波 |
| 卷積運算 | 訊號卷積 | `convolve(x, y)` | — |
| 自相關 | 分析訊號延遲關係 | `correlate(x, y)` | — |
| 頻譜分析 | 功率與頻率分析 | `welch(x)`, `periodogram()` | — |
| 峰值偵測 | 找出波峰位置 | `find_peaks(x)` | 常用於感測訊號 |

---

## 八、插值與曲線擬合（`scipy.interpolate`）

| 功能類別 | 功能說明 | 常用函數 / 方法 | 備註 |
|------------|------------|------------------|------|
| 一維插值 | 線性或多項式插值 | `interp1d(x, y, kind='linear')` | 支援 spline, cubic |
| 二維插值 | 曲面插值 | `interp2d(x, y, z)` | — |
| 樣條插值 | B-spline 擬合 | `UnivariateSpline(x, y, s)` | — |
| 最近鄰插值 | 基於距離 | `griddata(points, values, xi, method='nearest')` | 常用於地理資料 |

---

## 九、影像與多維資料處理（`scipy.ndimage`）

| 功能類別 | 功能說明 | 常用函數 / 方法 | 備註 |
|------------|------------|------------------|------|
| 濾波器 | 高斯 / 中值 / Sobel | `gaussian_filter()`, `median_filter()`, `sobel()` | 常用於影像平滑 |
| 幾何變換 | 旋轉 / 平移 / 縮放 | `rotate()`, `shift()`, `zoom()` | — |
| 卷積運算 | 任意核卷積 | `convolve()`、`correlate()` | — |
| 邊緣偵測 | 偵測梯度變化 | `prewitt()`, `sobel()` | — |
| 物件標記 | 找出區域連通物 | `label()`、`find_objects()` | — |

---

## 十、稀疏矩陣（`scipy.sparse`）

| 功能類別 | 功能說明 | 常用函數 / 方法 | 備註 |
|------------|------------|------------------|------|
| 稀疏矩陣格式 | 高效儲存稀疏資料 | `csr_matrix()`, `csc_matrix()` | 行壓縮 / 列壓縮格式 |
| 稀疏運算 | 矩陣乘法 / 轉置 | `.dot()`, `.T`, `.toarray()` | 節省記憶體 |
| 稀疏解方程 | 解大型線性方程 | `spsolve(A, b)` | 用於科學模擬 |
| 稀疏統計 | 統計分析 | `.count_nonzero()` | 非零元素計算 |

---

## 十一、空間與距離（`scipy.spatial`）

| 功能類別 | 功能說明 | 常用函數 / 方法 | 備註 |
|------------|------------|------------------|------|
| 距離計算 | 向量距離度量 | `distance.euclidean(u, v)`、`cityblock()`、`cosine()` | 支援多種距離 |
| KD-Tree | 快速最近鄰查詢 | `KDTree(data)`、`query(x)` | — |
| Delaunay 三角化 | 幾何分割 | `Delaunay(points)` | 用於3D建模 |
| 凸包計算 | Convex Hull | `ConvexHull(points)` | 計算包覆邊界 |

---

## 十二、特殊函數（`scipy.special`）

| 功能類別 | 功能說明 | 常用函數 / 方法 | 備註 |
|------------|------------|------------------|------|
| Gamma / Beta | 數學特殊函數 | `gamma(x)`, `beta(a,b)` | — |
| Bessel 函數 | 工程振動分析 | `jn(n, x)` | — |
| 誤差函數 | 機率統計應用 | `erf(x)`, `erfc(x)` | — |
| 累積分布函數 | 常用於統計 | `ndtr(x)` | 常見於高斯分布 |
| 組合運算 | nCr, nPr | `comb(n, k)`、`perm(n, k)` | — |

---



---

