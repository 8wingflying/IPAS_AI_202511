# 📚 **Imbalanced-learn**
> **Imbalanced-learn** 是處理分類任務中資料不平衡問題的專用工具，  
> 提供 **過採樣、欠採樣、混合採樣、集成取樣與專用評估指標**。  
>  
> 常與 **scikit-learn**、**Pandas**、**NumPy** 結合，  
> 廣泛應用於 **欺詐偵測、醫療診斷、稀有事件預測、信用風險分析** 等領域。  

---

# ⚖️ imbalanced-learn 各項功能與常用函數總覽表  

---

## 一、Imbalanced-learn 概要（Overview）

| 模組名稱 | 中文說明 | 功能重點 |
|-----------|------------|-----------|
| `imblearn.over_sampling` | 過採樣（Over-sampling） | 增加少數類樣本數量 |
| `imblearn.under_sampling` | 欠採樣（Under-sampling） | 減少多數類樣本數量 |
| `imblearn.combine` | 結合採樣方法 | 同時使用過採樣與欠採樣 |
| `imblearn.ensemble` | 集成取樣（Ensemble Sampling） | 與集成學習模型結合 |
| `imblearn.pipeline` | 管線化整合（Pipeline） | 串聯取樣與模型訓練流程 |
| `imblearn.metrics` | 評估指標 | 適用於不平衡分類的專用指標 |
| `imblearn.datasets` | 測試資料集 | 內建不平衡樣本集供實驗使用 |

---

## 二、過採樣方法（Over-sampling Methods）

| 方法名稱 | 全名 / 英文對應 | 功能說明 | 常用參數 | 函數範例 |
|------------|----------------|------------|-------------|-------------|
| `RandomOverSampler` | 隨機過採樣 | 隨機重複少數類樣本 | `sampling_strategy`, `random_state` | `RandomOverSampler().fit_resample(X, y)` |
| `SMOTE` | Synthetic Minority Over-sampling Technique | 以 KNN 為基礎生成少數類新樣本 | `k_neighbors`, `sampling_strategy` | `SMOTE().fit_resample(X, y)` |
| `BorderlineSMOTE` | 邊界式 SMOTE | 僅對決策邊界附近樣本進行生成 | `kind='borderline-1'/'borderline-2'` | `BorderlineSMOTE().fit_resample(X, y)` |
| `SVMSMOTE` | SVM-SMOTE | 使用 SVM 找出支撐向量樣本產生新資料 | `m_neighbors`, `svm_estimator` | `SVMSMOTE().fit_resample(X, y)` |
| `ADASYN` | Adaptive Synthetic Sampling | 針對難分類樣本自動生成更多樣本 | `n_neighbors`, `sampling_strategy` | `ADASYN().fit_resample(X, y)` |
| `KMeansSMOTE` | K-means SMOTE | 結合 K-means 聚類產生多樣化樣本 | `kmeans_args`, `cluster_balance_threshold` | `KMeansSMOTE().fit_resample(X, y)` |

---

## 三、欠採樣方法（Under-sampling Methods）

| 方法名稱 | 全名 / 英文對應 | 功能說明 | 常用參數 | 函數範例 |
|------------|----------------|------------|-------------|-------------|
| `RandomUnderSampler` | 隨機欠採樣 | 隨機刪除多數類樣本 | `sampling_strategy`, `random_state` | `RandomUnderSampler().fit_resample(X, y)` |
| `NearMiss` | 近鄰欠採樣 | 根據 KNN 距離選取最具代表性的樣本 | `version=1/2/3`, `n_neighbors` | `NearMiss(version=1).fit_resample(X, y)` |
| `TomekLinks` | Tomek 連結法 | 移除重疊樣本以清理決策邊界 | `sampling_strategy` | `TomekLinks().fit_resample(X, y)` |
| `EditedNearestNeighbours (ENN)` | 編輯最近鄰法 | 移除分類不確定的樣本 | `n_neighbors` | `EditedNearestNeighbours().fit_resample(X, y)` |
| `CondensedNearestNeighbour (CNN)` | 壓縮最近鄰法 | 保留最具代表性樣本 | `n_neighbors` | `CondensedNearestNeighbour().fit_resample(X, y)` |
| `OneSidedSelection (OSS)` | 單邊選擇法 | 結合 Tomek Links 與 CNN | `n_neighbors`, `n_seeds_S` | `OneSidedSelection().fit_resample(X, y)` |
| `ClusterCentroids` | 聚類中心取樣 | 用 K-means 壓縮多數樣本 | `n_clusters` | `ClusterCentroids().fit_resample(X, y)` |

---

## 四、結合方法（Combine Sampling Methods）

| 方法名稱 | 功能說明 | 原理 | 函數範例 |
|------------|------------|-----------|-----------|
| `SMOTEENN` | 先 SMOTE 過採樣再 ENN 淨化 | 結合合成樣本與噪聲移除 | `SMOTEENN().fit_resample(X, y)` |
| `SMOTETomek` | 先 SMOTE 再 Tomek 清理樣本 | 同時強化邊界與平衡 | `SMOTETomek().fit_resample(X, y)` |

---

## 五、集成取樣方法（Ensemble-based Resampling）

| 方法名稱 | 全名 / 英文對應 | 功能說明 | 常用參數 | 函數範例 |
|------------|----------------|------------|-------------|-------------|
| `BalancedBaggingClassifier` | 平衡 Bagging 分類器 | 在每個 Bagging 迭代中進行再取樣 | `n_estimators`, `base_estimator` | `BalancedBaggingClassifier().fit(X, y)` |
| `BalancedRandomForestClassifier` | 平衡隨機森林 | 對每棵樹進行欠採樣 | `n_estimators`, `max_depth` | `BalancedRandomForestClassifier().fit(X, y)` |
| `EasyEnsembleClassifier` | 簡易集成法 | 建立多個子集進行欠採樣後集成 | `n_estimators` | `EasyEnsembleClassifier().fit(X, y)` |
| `RUSBoostClassifier` | 欠採樣 + AdaBoost | 結合 Boosting 與隨機欠採樣 | `n_estimators`, `learning_rate` | `RUSBoostClassifier().fit(X, y)` |

---

## 六、評估指標模組（Metrics for Imbalanced Data）

| 指標名稱 | 功能說明 | 常用函數 / 方法 | 備註 |
|------------|------------|------------------|------|
| 分類報告 | 評估 Precision、Recall、F1 | `classification_report_imbalanced(y_true, y_pred)` | 類似 sklearn 的報告但更詳細 |
| G-mean | 幾何平均 | `geometric_mean_score(y_true, y_pred)` | 適合不平衡分類 |
| 平衡準確度 | Balanced Accuracy | `balanced_accuracy_score(y_true, y_pred)` | 避免偏向多數類 |
| 指標矩陣 | 結合多項指標 | `sensitivity_specificity_support()` | 同時計算敏感度與特異度 |
| ROC-AUC | 曲線下面積 | `roc_auc_score(y_true, y_score)` | 可針對每一類別計算 |
| F-beta | 調整權重的 F 分數 | `fbeta_score(y_true, y_pred, beta)` | beta > 1 強調 recall |

---

## 七、管線與整合（Pipelines & Integration）

| 功能類別 | 功能說明 | 常用函數 / 方法 | 備註 |
|------------|------------|------------------|------|
| 管線化整合 | 將取樣與模型訓練結合 | `Pipeline(steps=[('sampling', SMOTE()), ('model', classifier)])` | 自動處理資料不平衡 |
| 與 sklearn 相容 | 完全兼容 sklearn API | `.fit()`, `.predict()`, `.score()` | 可與 GridSearchCV 併用 |
| 交叉驗證 | 平衡交叉驗證抽樣 | `RepeatedStratifiedKFold()` | 確保每折類別比例相同 |

---

## 八、資料集模組（Datasets）

| 功能類別 | 功能說明 | 常用函數 / 方法 | 備註 |
|------------|------------|------------------|------|
| 內建資料集 | 不平衡樣本測試 | `make_imbalance(X, y, sampling_strategy)` | 將平衡資料變成不平衡 |
| 人工資料產生 | 模擬不平衡資料 | `make_classification(weights=[0.9,0.1])` | 搭配 sklearn 使用 |

---



✅ **常見組合範例：**

```python
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# 建立管線：過採樣 + 模型
model = Pipeline(steps=[
    ('oversample', SMOTE(random_state=42)),
    ('classifier', RandomForestClassifier())
])

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y)
model.fit(X_train, y_train)
print(model.score(X_test, y_test))
```
