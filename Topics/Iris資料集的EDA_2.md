## Iris資料集的EDA
```python
# =========================================================
# 📘 Exploratory Data Analysis (EDA) 常用圖形實作手冊
# Dataset: Iris (鳶尾花資料集)
# Author: T Ben
# =========================================================

# 匯入常用套件
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from math import pi
import numpy as np
from pandas.plotting import parallel_coordinates

# 設定風格
sns.set(style="whitegrid", font="Microsoft JhengHei", font_scale=1.1)
plt.rcParams['axes.unicode_minus'] = False

# ---------------------------------------------------------
# 1️⃣ 載入資料集
# ---------------------------------------------------------
df = sns.load_dataset('iris')
print("✅ 資料集基本資訊：")
print(df.info())
print("\n📋 前五筆資料：")
display(df.head())

# ---------------------------------------------------------
# 2️⃣ 直方圖 (Histogram)
# ---------------------------------------------------------
plt.figure(figsize=(6,4))
sns.histplot(data=df, x='sepal_length', kde=True, color='skyblue')
plt.title('Histogram of Sepal Length (花萼長度直方圖)')
plt.show()

# ---------------------------------------------------------
# 3️⃣ 箱型圖 (Box Plot)
# ---------------------------------------------------------
plt.figure(figsize=(6,4))
sns.boxplot(data=df, x='species', y='sepal_width', palette='Set2')
plt.title('Boxplot of Sepal Width by Species (依花種比較花萼寬度)')
plt.show()

# ---------------------------------------------------------
# 4️⃣ 小提琴圖 (Violin Plot)
# ---------------------------------------------------------
plt.figure(figsize=(6,4))
sns.violinplot(data=df, x='species', y='petal_length', palette='muted')
plt.title('Violin Plot of Petal Length by Species (依花種比較花瓣長度)')
plt.show()

# ---------------------------------------------------------
# 5️⃣ 散佈圖 (Scatter Plot)
# ---------------------------------------------------------
plt.figure(figsize=(6,4))
sns.scatterplot(data=df, x='sepal_length', y='petal_length', hue='species', s=80)
plt.title('Scatter Plot: Sepal vs Petal Length (花萼長度 vs 花瓣長度)')
plt.show()

# ---------------------------------------------------------
# 6️⃣ 成對關係圖 (Pair Plot)
# ---------------------------------------------------------
sns.pairplot(df, hue='species', corner=True, diag_kind='kde')
plt.suptitle('Pair Plot of Iris Dataset (成對關係圖)', y=1.02)
plt.show()

# ---------------------------------------------------------
# 7️⃣ 熱度圖 (Heatmap)
# ---------------------------------------------------------
corr = df.corr(numeric_only=True)
plt.figure(figsize=(6,5))
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Heatmap (變數相關係數熱度圖)')
plt.show()

# ---------------------------------------------------------
# 8️⃣ 密度圖 (Density Plot)
# ---------------------------------------------------------
plt.figure(figsize=(6,4))
sns.kdeplot(data=df, x='sepal_length', hue='species', fill=True, common_norm=False)
plt.title('Density Plot of Sepal Length (花萼長度密度分布圖)')
plt.show()

# ---------------------------------------------------------
# 9️⃣ 折線圖 (Line Plot)
# ---------------------------------------------------------
plt.figure(figsize=(6,4))
sns.lineplot(data=df.sort_values('sepal_length'), x='sepal_length', y='petal_length')
plt.title('Line Plot: Sepal vs Petal Length (花萼長度與花瓣長度折線關係)')
plt.show()

# ---------------------------------------------------------
# 🔟 雷達圖 (Radar Chart)
# ---------------------------------------------------------
avg = df.groupby('species').mean(numeric_only=True)
labels = avg.columns.tolist()
num_vars = len(labels)
angles = np.linspace(0, 2*pi, num_vars, endpoint=False).tolist()
angles += angles[:1]

fig, ax = plt.subplots(figsize=(6,6), subplot_kw=dict(polar=True))
for i, row in avg.iterrows():
    values = row.tolist()
    values += values[:1]
    ax.plot(angles, values, label=i)
    ax.fill(angles, values, alpha=0.25)
ax.set_xticks(angles[:-1])
ax.set_xticklabels(labels)
plt.title('Radar Chart for Iris Species (雷達圖: 各花種平均特徵)')
plt.legend()
plt.show()

# ---------------------------------------------------------
# 11️⃣ 平行座標圖 (Par

```
