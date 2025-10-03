## 教學錄影有程式碼
- 教學錄影1:附有Python程式碼的教學
- 教學錄影2:講解EDA技術與結果(不含程式碼)

## Iris資料集的EDA
- 1. 載入資料	檢視前幾筆資料、型態
- 2. 統計摘要	基本統計量 (平均、標準差、四分位數)
- 3. 單變量分析	直方圖 / 箱型圖觀察分布
- 4. 雙變量分析	散佈圖檢查兩特徵關係
- 5. 多變量分析	成對圖 (pairplot)、相關係數熱圖
- 6. 異常值檢查	箱型圖 / IQR
- 7. 結論	總結發現
#### 1. 載入資料	檢視前幾筆資料、型態
#### 2. 統計摘要	基本統計量 (平均、標準差、四分位數)
#### 3. 單變量分析	直方圖 / 箱型圖觀察分布
#### 4. 雙變量分析	散佈圖檢查兩特徵關係
#### 5. 多變量分析	成對圖 (pairplot)、相關係數熱圖
#### 6. 異常值檢查	箱型圖 / IQR
#### 7. 結論	總結發現

```python
# 匯入套件
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

# 1. 載入資料
iris = sns.load_dataset("iris")

# 2. 檢視基本資訊
print("===== 資料前五筆 =====")
print(iris.head())
print("\n===== 資料型態與缺失值 =====")
print(iris.info())

# 3. 統計摘要
print("\n===== 基本統計量 =====")
print(iris.describe())

# 4. 缺失值檢查
print("\n===== 缺失值統計 =====")
print(iris.isnull().sum())

# 5. 單變量分析：直方圖 + 箱型圖
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
sns.histplot(iris['sepal_length'], kde=True)
plt.title("Sepal Length 分布")

plt.subplot(1, 2, 2)
sns.boxplot(x="species", y="petal_length", data=iris)
plt.title("不同品種的 Petal Length 分布")
plt.show()

# 6. 雙變量分析：散佈圖
plt.figure(figsize=(6, 5))
sns.scatterplot(x="sepal_length", y="sepal_width", hue="species", data=iris)
plt.title("Sepal Length vs Sepal Width")
plt.show()

# 7. 多變量分析：成對圖
sns.pairplot(iris, hue="species")
plt.show()

# 8. 多變量分析：相關性熱圖
plt.figure(figsize=(6, 5))
sns.heatmap(iris.corr(), annot=True, cmap="coolwarm")
plt.title("特徵相關性熱圖")
plt.show()

# 9. 異常值檢查 (以 sepal_width 為例，利用 Z-score)
z_scores = stats.zscore(iris['sepal_width'])
outliers = iris[(abs(z_scores) > 3)]
print("\n===== 異常值檢查 (sepal_width) =====")
print(outliers)
```
