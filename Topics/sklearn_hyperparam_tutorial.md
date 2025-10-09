# 📚 scikit-learn 超參數搜尋程式比較教學筆記

本筆記深度比較 scikit-learn 支援的超參數搜尋方法（Hyperparameter Optimization Methods），包含說明、範例程式、可能輸出並附上中文評論。

---

## 👉 第一部分　Grid Search (網格搜尋)

### 概念
Grid Search 是最基礎的超參數尋索方法，它會將所有可能的參數組合全部測試，擔保每個超參數的全面搜尋。

### Python 範例
```python
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.datasets import load_iris

X, y = load_iris(return_X_y=True)

# 定義模型
model = SVC()

# 定義搜尋參數空間
param_grid = {
    'C': [0.1, 1, 10],
    'kernel': ['linear', 'rbf'],
    'gamma': ['scale', 'auto']
}

grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid_search.fit(X, y)

print("\n🔍 最佳參數:", grid_search.best_params_)
print("⭐ 最佳模型測試正確率:", grid_search.best_score_)
```

### 評論
- **優點**: 簡單、可重現結果
- **缺點**: 計算成本高，易對高維資料發生維度災難

---

## 🎡 第二部分　Random Search (隨機搜尋)

### 概念
Random Search 從參數空間中隨機抽取組合，能夠以較低的計算成本獲得良好結果，適合大型模型。

### Python 範例
```python
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_wine
from scipy.stats import randint

X, y = load_wine(return_X_y=True)
model = RandomForestClassifier()

param_dist = {
    'n_estimators': randint(50, 300),
    'max_depth': randint(3, 15),
    'min_samples_split': randint(2, 10)
}

random_search = RandomizedSearchCV(model, param_distributions=param_dist, n_iter=20, cv=5, scoring='accuracy', n_jobs=-1, random_state=42)
random_search.fit(X, y)

print("\n🔍 最佳參數:", random_search.best_params_)
print("⭐ 最佳模型正確率:", random_search.best_score_)
```

### 評論
- **優點**: 效率高、適用性強
- **缺點**: 結果具隨機性，不一定是全域最佳解

---

## ⚙️ 第三部分　Halving Random Search (漸進式隨機搜尋)

### 概念
逐步減少不優的試驗組合，把計算資源分配給表現較好的參數。
屬於 **Successive Halving** 類方法，是現代自動化模型搜尋的基礎技術。

### Python 範例
```python
from sklearn.experimental import enable_halving_search_cv  # 必須先啟用實驗模組
from sklearn.model_selection import HalvingRandomSearchCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.datasets import load_breast_cancer
from scipy.stats import uniform, randint

X, y = load_breast_cancer(return_X_y=True)
model = GradientBoostingClassifier()

param_dist = {
    'n_estimators': randint(50, 300),
    'learning_rate': uniform(0.01, 0.2),
    'max_depth': randint(2, 8)
}

halving_search = HalvingRandomSearchCV(model, param_distributions=param_dist, factor=3, resource='n_estimators', max_resources=300, random_state=42, scoring='accuracy', cv=5)
halving_search.fit(X, y)

print("\n🔍 最佳參數:", halving_search.best_params_)
print("⭐ 最佳模型正確率:", halving_search.best_score_)
```

### 評論
- **優點**: 有效利用資源，調參速度快
- **缺點**: 需要詳細設定 factor 與 resource

---

## 🧬 第四部分　BayesSearchCV (赫葉斯優化)

### 概念
使用赫葉斯理論建立代理模型，逐步根據模型預測測試方向，能在較少試驗中找到近佳解。

### Python 範例
```python
# pip install scikit-optimize
from skopt import BayesSearchCV
from sklearn.svm import SVC
from sklearn.datasets import load_iris

X, y = load_iris(return_X_y=True)

search_spaces = {
    'C': (1e-3, 1e+2, 'log-uniform'),
    'gamma': (1e-4, 1e-1, 'log-uniform'),
    'kernel': ['rbf', 'poly']
}

opt = BayesSearchCV(
    SVC(),
    search_spaces,
    n_iter=25,
    scoring='accuracy',
    cv=5,
    n_jobs=-1,
    random_state=42
)

opt.fit(X, y)
print("\n🔍 最佳參數:", opt.best_params_)
print("⭐ 最佳模型正確率:", opt.best_score_)
```

### 評論
- **優點**: 效率高、在少數次調參中即可推近最佳值
- **缺點**: 需額外套件、理論難度高

---

## 📊 總結比較表

| 方法 | 模組 | 搜尋模式 | 優點 | 缺點 | 適用場景 |
|------|------|-----------|------|------|------------|
| GridSearchCV | `model_selection` | 完全網格搜尋 | 簡單、可重現 | 計算成本高 | 小型模型 |
| RandomizedSearchCV | `model_selection` | 隨機抽樣 | 效率高 | 隨機性高 | 大型資料 |
| HalvingRandomSearchCV | `experimental` | 漸進式自適應 | 資源利用效率高 | 設定複雜 | AutoML 應用 |
| BayesSearchCV | `skopt` | 模型預測 | 快速收斂 | 額外套件 | 高精準調參 |

---

## 📚 學習建議

- 推薦從 **RandomizedSearchCV** 開始，推起漸進式搜尋方法。
- 當參數太多時，避免 Grid Search，優先考慮 Random Search 或 Bayesian Optimization。
- 在 AutoML 環境，最推薦 HalvingRandomSearchCV + Optuna / Ray Tune 等工具。

---

🔗 **可用產出**:
- `.ipynb` 版: 可直接執行且看輸出
- `.md` 版: 適合教學與正式教程文件

---

> 更進階版本：可以增加 **Optuna 與 Ray Tune 實例比較版** ，包含新一代 AutoML 調參架構。

