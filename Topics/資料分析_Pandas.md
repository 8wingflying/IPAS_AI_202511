# 📚 **Pandas** 
> **Pandas** 是 Python 的資料分析核心套件，結合 **NumPy** 的運算效率與 **SQL / Excel** 的操作便利性。  
> 它支援 **資料清理、分析、統計、群組、轉換與視覺化**，  
> 並廣泛應用於 **AI資料前處理、商業分析、金融資料、時間序列建模** 等領域。

# 🐼 Pandas 各項功能與常用函數總覽表  

---

## 一、資料結構（Data Structures）

| 功能類別 | 英文名稱 | 功能說明 | 常用函數 / 方法 | 備註 |
|------------|-----------|------------|------------------|------|
| 一維序列 | `Series` | 儲存一維帶索引的資料 | `pd.Series(data, index)` | 可視為進階版的 NumPy array |
| 二維表格 | `DataFrame` | 儲存帶標籤的行列資料 | `pd.DataFrame(data, columns, index)` | 最常用的 Pandas 結構 |
| 索引物件 | `Index` | 行列標籤的集合 | `df.index`, `df.columns` | 支援重新命名與操作 |
| 多層索引 | `MultiIndex` | 建立階層式索引 | `pd.MultiIndex.from_tuples()` | 用於群組與交叉表 |

---

## 二、資料讀取與輸出（Input / Output, I/O）

| 功能類別 | 英文名稱 | 功能說明 | 常用函數 / 方法 | 備註 |
|------------|-----------|------------|------------------|------|
| 讀取 CSV | Read CSV File | 從 CSV 載入資料 | `pd.read_csv('file.csv')` | 最常用的讀取方式 |
| 儲存 CSV | Save CSV File | 將 DataFrame 寫入 CSV | `df.to_csv('out.csv', index=False)` | — |
| 讀取 Excel | Read Excel | 載入 Excel 檔 | `pd.read_excel('file.xlsx', sheet_name)` | 需安裝 openpyxl |
| 儲存 Excel | Save Excel | 寫入 Excel 檔 | `df.to_excel('out.xlsx')` | — |
| 讀取 JSON | Read JSON | 載入 JSON 格式資料 | `pd.read_json('file.json')` | — |
| 讀取 SQL | SQL Query | 從資料庫載入資料 | `pd.read_sql(query, conn)` | 支援 SQLite / MySQL 等 |
| 讀取剪貼簿 | Clipboard | 直接從剪貼簿貼上 | `pd.read_clipboard()` | 快速載入表格資料 |

---

## 三、資料檢視與摘要（Data Inspection & Summary）

| 功能類別 | 功能說明 | 常用函數 / 方法 | 備註 |
|------------|------------|------------------|------|
| 顯示前幾筆 | 顯示前 N 行 | `df.head(n)` | 預設 5 行 |
| 顯示後幾筆 | 顯示最後 N 行 | `df.tail(n)` | — |
| 檢視維度 | 顯示行列數 | `df.shape` | 回傳 (rows, cols) |
| 檢視欄位資訊 | 顯示欄位型態 | `df.info()` | 含欄位名稱與記憶體用量 |
| 敘述統計 | 取得平均、標準差等 | `df.describe()` | 僅適用於數值欄 |
| 檢視欄位名 | 顯示所有欄位名稱 | `df.columns` | — |
| 檢視索引 | 顯示索引內容 | `df.index` | — |

---

## 四、資料選取與篩選（Selection & Filtering）

| 功能類別 | 功能說明 | 常用函數 / 方法 | 備註 |
|------------|------------|------------------|------|
| 按欄位選取 | 選取單一欄位 | `df['column']` 或 `df.column` | 回傳 Series |
| 多欄選取 | 選取多個欄位 | `df[['col1','col2']]` | 回傳 DataFrame |
| 條件篩選 | 篩選符合條件資料 | `df[df['col'] > 10]` | — |
| 索引選取 | 按標籤 / 位置選取 | `df.loc[]`, `df.iloc[]` | loc 為標籤、iloc 為位置 |
| 切片取樣 | 區間選取 | `df[5:10]` | 與 Python list 相同 |
| 取唯一值 | 找出不重複資料 | `df['col'].unique()`、`nunique()` | — |
| 排序 | 排序資料 | `df.sort_values(by='col')` | 可升降冪排序 |

---

## 五、資料清理（Data Cleaning）

| 功能類別 | 功能說明 | 常用函數 / 方法 | 備註 |
|------------|------------|------------------|------|
| 缺失值檢查 | 檢查空值 | `df.isnull()`、`df.notnull()` | 回傳布林矩陣 |
| 缺失值刪除 | 移除空值列或欄 | `df.dropna(axis=0)` | `axis=1` 為欄 |
| 缺失值填補 | 以特定值填補 | `df.fillna(value)` | 可用平均或中位數 |
| 重複值檢查 | 找出重複列 | `df.duplicated()` | — |
| 重複值刪除 | 移除重複資料 | `df.drop_duplicates()` | — |
| 欄位重新命名 | 改變欄位名稱 | `df.rename(columns={'old':'new'})` | — |
| 資料型別轉換 | 改變欄位型態 | `df['col'].astype(float)` | — |

---

## 六、資料運算與轉換（Data Transformation）

| 功能類別 | 功能說明 | 常用函數 / 方法 | 備註 |
|------------|------------|------------------|------|
| 新增欄位 | 建立新欄 | `df['new'] = ...` | — |
| 欄位運算 | 向量化數學運算 | `df['col'] * 2` | 自動逐列運算 |
| Apply 函數 | 對欄位套用自訂函式 | `df['col'].apply(func)` | — |
| Map 對應 | 對值進行轉換 | `df['col'].map({'A':1,'B':2})` | — |
| Lambda 運算 | 匿名函式快速運算 | `df.apply(lambda x: x**2)` | — |
| 資料正規化 | 標準化或縮放 | `(df - df.mean()) / df.std()` | — |

---

## 七、資料群組與彙總（Grouping & Aggregation）

| 功能類別 | 功能說明 | 常用函數 / 方法 | 備註 |
|------------|------------|------------------|------|
| 群組運算 | 依欄位分組 | `df.groupby('col')` | — |
| 群組統計 | 平均 / 計數 / 總和 | `.mean()`, `.count()`, `.sum()` | 可鏈式使用 |
| 多層群組 | 多欄位分組 | `df.groupby(['A','B']).agg(...)` | — |
| 樞紐分析表 | Pivot Table | `pd.pivot_table(df, values, index, columns, aggfunc)` | 與 Excel 類似 |
| 交叉表 | Crosstab | `pd.crosstab(df.A, df.B)` | 頻率統計 |

---

## 八、合併與串接（Merging & Concatenation）

| 功能類別 | 功能說明 | 常用函數 / 方法 | 備註 |
|------------|------------|------------------|------|
| 串接資料 | 垂直或水平合併 | `pd.concat([df1, df2], axis=0)` | axis=1 為橫向 |
| 合併資料 | 類似 SQL Join | `pd.merge(df1, df2, on='key')` | 可指定 `how='inner'/'outer'` |
| 加入索引 | 以索引對齊合併 | `df1.join(df2)` | index 對齊 |

---

## 九、時間序列（Time Series）

| 功能類別 | 功能說明 | 常用函數 / 方法 | 備註 |
|------------|------------|------------------|------|
| 日期轉換 | 字串轉日期 | `pd.to_datetime(df['date'])` | 支援多格式 |
| 設定時間索引 | 將日期設為索引 | `df.set_index('date')` | — |
| 重新取樣 | 改變時間頻率 | `df.resample('M').mean()` | 月、週、日等 |
| 時間差計算 | 日期運算 | `df['diff'] = df['date2'] - df['date1']` | 回傳 timedelta |
| 移動平均 | 平滑化序列 | `df['col'].rolling(window=3).mean()` | 常用於趨勢分析 |

---

## 十、統計與數據分析（Statistical Functions）

| 功能類別 | 功能說明 | 常用函數 / 方法 | 備註 |
|------------|------------|------------------|------|
| 平均值 | 求平均 | `df['col'].mean()` | — |
| 中位數 | 求中位數 | `df['col'].median()` | — |
| 標準差 | 求變異程度 | `df['col'].std()` | — |
| 相關係數 | 計算欄位相關性 | `df.corr()` | — |
| 共變異數 | 求欄位共變異 | `df.cov()` | — |
| 統計摘要 | 描述性統計 | `df.describe()` | — |

---

## 十一、輸出與視覺化（Output & Visualization）

| 功能類別 | 功能說明 | 常用函數 / 方法 | 備註 |
|------------|------------|------------------|------|
| 基本繪圖 | 快速可視化 | `df.plot(kind='line/bar/hist')` | 基於 Matplotlib |
| 箱型圖 | 檢查離群值 | `df.boxplot(column='col')` | — |
| 直方圖 | 顯示分布 | `df['col'].hist()` | — |
| 散點圖 | 變數關係 | `df.plot.scatter(x='col1', y='col2')` | — |

---


---

