##  📚 **NumPy（Numerical Python）** 
> **NumPy（Numerical Python）** 是 Python 科學運算的核心套件，  
> 支援高效的多維陣列運算、線性代數、統計分析與隨機模擬。  
> 它是 **Pandas、scikit-learn、TensorFlow、PyTorch** 等套件的基礎，  
> 被廣泛應用於 **AI訓練、資料分析、信號處理、影像計算與工程模擬** 等領域。

---
# 🔢 NumPy 各項功能與常用函數總覽表  
> **中英對照版教學講義**

---

## 一、陣列建立與初始化（Array Creation）

| 功能類別 | 功能說明 | 常用函數 / 方法 | 備註 |
|------------|------------|------------------|------|
| 建立陣列 | 將列表轉為陣列 | `np.array([1,2,3])` | 最基本方式 |
| 建立零矩陣 | 產生全 0 陣列 | `np.zeros((rows, cols))` | 預設型別 float |
| 建立一矩陣 | 產生全 1 陣列 | `np.ones((rows, cols))` | — |
| 建立空矩陣 | 建立未初始化陣列 | `np.empty((rows, cols))` | 內容為隨機值 |
| 建立等差數列 | 均勻分布數列 | `np.arange(start, stop, step)` | 類似 range() |
| 建立等距點數 | 固定數量的等距點 | `np.linspace(start, stop, num)` | 常用於繪圖 |
| 建立隨機數 | 生成隨機陣列 | `np.random.rand()`, `np.random.randint()` | 支援多維 |
| 單位矩陣 | 生成對角為 1 的矩陣 | `np.eye(n)` | 常用於線性代數 |

---

## 二、陣列屬性與結構（Array Attributes & Structure）

| 功能類別 | 功能說明 | 常用屬性 / 方法 | 備註 |
|------------|------------|------------------|------|
| 陣列形狀 | 查看維度 | `arr.shape` | 回傳 tuple |
| 陣列大小 | 元素總數 | `arr.size` | — |
| 陣列維度 | 幾維陣列 | `arr.ndim` | — |
| 資料型別 | 元素型態 | `arr.dtype` | e.g. int32, float64 |
| 改變形狀 | 重塑陣列結構 | `arr.reshape(r, c)` | 不改變資料內容 |
| 攤平成一維 | 將多維轉一維 | `arr.flatten()` | 常用於預測模型輸入 |
| 型別轉換 | 改變資料型態 | `arr.astype(np.float32)` | — |

---

## 三、索引與切片（Indexing & Slicing）

| 功能類別 | 功能說明 | 常用函數 / 方法 | 備註 |
|------------|------------|------------------|------|
| 基本索引 | 按位置取值 | `arr[0,1]` | 取第0列第1欄 |
| 區間切片 | 取多筆資料 | `arr[1:4]`, `arr[:, 0]` | Python風格 |
| 高維切片 | 多維範圍選取 | `arr[0:2, 1:3]` | 行列選擇 |
| 布林索引 | 條件過濾 | `arr[arr > 5]` | 回傳符合條件元素 |
| 花式索引 | 指定索引列表 | `arr[[0,2,4]]` | 可搭配多維使用 |
| 反向索引 | 從尾端取值 | `arr[-1]`, `arr[::-1]` | 支援負索引 |

---

## 四、數學與統計運算（Mathematical & Statistical Operations）

| 功能類別 | 功能說明 | 常用函數 / 方法 | 備註 |
|------------|------------|------------------|------|
| 基本加減乘除 | 元素運算 | `arr1 + arr2`, `arr * 2` | 向量化運算 |
| 開根號 | 平方根運算 | `np.sqrt(arr)` | — |
| 指數與對數 | e 與 log 運算 | `np.exp(arr)`, `np.log(arr)` | — |
| 絕對值 | 取元素絕對值 | `np.abs(arr)` | — |
| 加總 | 所有元素總和 | `np.sum(arr)` | `axis=0/1` 控制方向 |
| 平均 | 計算平均值 | `np.mean(arr)` | — |
| 標準差 | 數值離散程度 | `np.std(arr)` | — |
| 最小 / 最大值 | 取極值 | `np.min(arr)`, `np.max(arr)` | — |
| 百分位 | 計算分位數 | `np.percentile(arr, 75)` | 常用於統計分析 |

---

## 五、線性代數（Linear Algebra）

| 功能類別 | 功能說明 | 常用函數 / 方法 | 備註 |
|------------|------------|------------------|------|
| 矩陣相乘 | 內積 / 點積 | `np.dot(a, b)`、`a @ b` | — |
| 轉置 | 行列互換 | `a.T` | — |
| 反矩陣 | 求矩陣逆 | `np.linalg.inv(a)` | 需為方陣 |
| 行列式 | 計算 determinant | `np.linalg.det(a)` | — |
| 特徵值 / 向量 | Eigen decomposition | `np.linalg.eig(a)` | — |
| 解聯立方程 | 求解 Ax = b | `np.linalg.solve(A, b)` | — |
| 奇異值分解 | SVD 分解 | `np.linalg.svd(a)` | 用於降維與壓縮 |

---

## 六、陣列運算與邏輯比較（Array Operations & Logic）

| 功能類別 | 功能說明 | 常用函數 / 方法 | 備註 |
|------------|------------|------------------|------|
| 元素比較 | 比較大小 | `arr > 0`、`arr1 == arr2` | 回傳布林矩陣 |
| 全部條件 | 所有值皆符合 | `np.all(condition)` | — |
| 任一條件 | 只要有一值符合 | `np.any(condition)` | — |
| 陣列加總 | 按軸合計 | `arr.sum(axis=0)` | — |
| 最大值索引 | 回傳極值位置 | `np.argmax(arr)`、`np.argmin(arr)` | — |
| 排序 | 陣列排序 | `np.sort(arr)`、`arr.argsort()` | — |
| 唯一值 | 找出不重複值 | `np.unique(arr)` | 可搭配 return_counts=True |

---

## 七、隨機數與統計分布（Random Numbers & Distributions）

| 功能類別 | 功能說明 | 常用函數 / 方法 | 備註 |
|------------|------------|------------------|------|
| 均勻分布 | 產生 0~1 隨機值 | `np.random.rand(shape)` | — |
| 整數亂數 | 產生整數 | `np.random.randint(low, high, size)` | — |
| 常態分布 | 平均與標準差 | `np.random.normal(mean, std, size)` | — |
| 二項分布 | 成功次數模擬 | `np.random.binomial(n, p, size)` | — |
| 抽樣 | 隨機取樣 | `np.random.choice(arr, size)` | — |
| 設定種子 | 控制隨機結果 | `np.random.seed(value)` | 確保重現性 |

---

## 八、陣列操作與結合（Array Manipulation）

| 功能類別 | 功能說明 | 常用函數 / 方法 | 備註 |
|------------|------------|------------------|------|
| 串接 | 陣列合併 | `np.concatenate([a,b], axis=0)` | — |
| 垂直合併 | 上下堆疊 | `np.vstack([a,b])` | — |
| 水平合併 | 左右堆疊 | `np.hstack([a,b])` | — |
| 分割 | 拆分陣列 | `np.split(arr, n)`、`np.hsplit()` | — |
| 重塑 | 改變維度 | `arr.reshape((r, c))` | 常見於影像/AI前處理 |
| 轉置 | 行列互換 | `arr.T` | — |
| 堆疊多維 | 建立多層矩陣 | `np.stack([a,b], axis=0)` | — |

---

## 九、邏輯、條件與遮罩運算（Conditional & Masking）

| 功能類別 | 功能說明 | 常用函數 / 方法 | 備註 |
|------------|------------|------------------|------|
| 條件選擇 | if-else 方式 | `np.where(condition, x, y)` | — |
| 遮罩運算 | 根據條件篩選元素 | `arr[mask]` | mask 為布林陣列 |
| 條件統計 | 統計符合條件數 | `(arr > 0).sum()` | 常用於分類統計 |

---

## 十、其他實用工具（Utilities & Integration）

| 功能類別 | 功能說明 | 常用函數 / 方法 | 備註 |
|------------|------------|------------------|------|
| 生成索引網格 | 建立 X, Y 座標矩陣 | `np.meshgrid(x, y)` | 常用於繪圖 |
| 向量化函數 | 將自訂函式向量化 | `np.vectorize(func)` | 提升效能 |
| 儲存檔案 | 儲存 NumPy 陣列 | `np.save('file.npy', arr)` | 二進位格式 |
| 載入檔案 | 讀取 .npy 檔 | `np.load('file.npy')` | — |
| 結合 Pandas | 轉為 DataFrame | `pd.DataFrame(np_array)` | 整合分析流程 |

---



