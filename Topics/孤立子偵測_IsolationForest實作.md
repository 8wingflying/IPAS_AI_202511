## GEMINI實作
- 以物件導向（Object-Oriented Programming, OOP）方式實作 Isolation Forest 異常檢測演算法的範例程式碼。
- 這個實作範例將包含兩個主要類別(Class)：
  - IsolationTree：用來建立單棵隨機隔離樹。
  - IsolationForest：用來集成多棵隔離樹，計算最終的異常分數。
```python
import numpy as np
from typing import List, Union

class IsolationTree:
    """
    Isolation Forest 的單棵隔離樹 (Isolation Tree, iTree)。
    
    透過遞迴地隨機切分數據，將異常點快速隔離。
    """
    def __init__(self, max_depth: int):
        # 樹的最大深度，限制樹的大小，提高效率
        self.max_depth = max_depth
        # 儲存節點的結構：None 表示尚未訓練或到達葉節點
        self.split_feature = None  # 用來切分的特徵索引
        self.split_value = None    # 用來切分的特徵值
        self.left = None           # 左子樹 (遞迴的 IsolationTree 實例)
        self.right = None          # 右子樹 (遞迴的 IsolationTree 實例)
        self.n_samples = 0         # 節點包含的樣本數

    def fit(self, X: np.ndarray, current_depth: int):
        """
        訓練隔離樹。
        
        Args:
            X (np.ndarray): 節點上的數據集 (樣本 x 特徵)。
            current_depth (int): 當前節點的深度。
        """
        self.n_samples = X.shape[0]
        
        # --- 停止條件 ---
        # 1. 樣本數為 1 或 0 (已完全隔離)
        # 2. 達到最大深度限制
        # 3. 數據集所有維度上的值都相同 (無法再切分)
        if self.n_samples <= 1 or current_depth >= self.max_depth or np.all(X == X[0, :]):
            return

        # --- 隨機切分 ---
        n_features = X.shape[1]
        
        # 隨機選擇一個特徵進行切分 (Isolation)
        self.split_feature = np.random.randint(0, n_features)
        feature_data = X[:, self.split_feature]
        
        # 隨機選擇一個切分點 (在該特徵的 min/max 之間)
        min_val = np.min(feature_data)
        max_val = np.max(feature_data)
        
        # 如果特徵數據都一樣，無法切分，則直接返回
        if min_val == max_val:
            return
            
        self.split_value = np.random.uniform(min_val, max_val)
        
        # 根據切分值將數據分成左右兩部分
        left_mask = feature_data < self.split_value
        right_mask = feature_data >= self.split_value
        
        X_left = X[left_mask]
        X_right = X[right_mask]
        
        # --- 遞迴建樹 ---
        # 處理左子樹
        self.left = IsolationTree(self.max_depth)
        self.left.fit(X_left, current_depth + 1)
        
        # 處理右子樹
        self.right = IsolationTree(self.max_depth)
        self.right.fit(X_right, current_depth + 1)

    def get_path_length(self, x: np.ndarray, current_depth: int) -> float:
        """
        計算單個樣本在當前樹中的路徑長度 (Path Length)。
        
        Args:
            x (np.ndarray): 單個數據點。
            current_depth (int): 當前遞迴深度。
            
        Returns:
            float: 從根節點到葉節點的路徑長度。
        """
        # 如果達到葉節點，返回當前深度加上一個調整值
        if self.left is None and self.right is None:
            # c(n) 是用於路徑長度估計的調整項（論文中提到）
            # 這裡為簡化，直接返回當前深度，但在實際應用中應加入調整項
            return current_depth 
        
        # 根據切分點遞迴地向下搜尋
        if x[self.split_feature] < self.split_value:
            return self.left.get_path_length(x, current_depth + 1)
        else:
            return self.right.get_path_length(x, current_depth + 1)

class IsolationForest:
    """
    Isolation Forest 異常檢測器。
    
    集成多棵 iTree，並計算異常分數。
    """
    def __init__(self, n_estimators: int = 100, max_samples: int = 256):
        """
        Args:
            n_estimators (int): 森林中樹的數量。
            max_samples (int): 每棵樹訓練時使用的樣本子集大小。
        """
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.trees: List[IsolationTree] = []
        
        # 最大深度 H 的計算公式：ceil(log2(max_samples))
        # 這是為了限制樹的高度，因為異常點通常在淺層被隔離
        self.max_depth = int(np.ceil(np.log2(max_samples)))
        
        # 用於計算異常分數的標準化因子 (用於平均路徑長度)
        self.c = self._average_path_length(max_samples)


    def _average_path_length(self, n: int) -> float:
        """
        計算給定樣本數 n 時的平均路徑長度調整因子 c(n)。
        H(i) 是調和數（Harmonic Number）的估計。
        """
        if n > 2:
            return 2.0 * (np.log(n - 1) + 0.5772156649) - (2.0 * (n - 1) / n)
        elif n == 2:
            return 1.0
        else: # n <= 1
            return 0.0

    def fit(self, X: np.ndarray):
        """
        訓練 Isolation Forest。
        
        Args:
            X (np.ndarray): 訓練數據集 (樣本 x 特徵)。
        """
        n_rows = X.shape[0]
        if self.max_samples > n_rows:
            sample_size = n_rows
        else:
            sample_size = self.max_samples
            
        self.trees = []
        
        for _ in range(self.n_estimators):
            # 隨機抽取子樣本 (子採樣)
            indices = np.random.choice(n_rows, size=sample_size, replace=False)
            X_subset = X[indices]
            
            # 訓練單棵隔離樹
            tree = IsolationTree(self.max_depth)
            tree.fit(X_subset, 0)
            self.trees.append(tree)

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """
        計算每個樣本的異常分數（Anomaly Score）。
        
        Args:
            X (np.ndarray): 待預測的數據集。
            
        Returns:
            np.ndarray: 異常分數，值越接近 1 表示越異常。
        """
        if not self.trees:
            raise RuntimeError("Model is not fitted yet.")

        # 1. 計算所有樣本在所有樹中的平均路徑長度 E(h(x))
        avg_path_lengths = np.zeros(X.shape[0])
        
        for i, x in enumerate(X):
            path_lengths = []
            for tree in self.trees:
                # 取得路徑長度。注意：我們傳入的 x 必須與訓練時的數據維度一致
                path_lengths.append(tree.get_path_length(x, 0))
            avg_path_lengths[i] = np.mean(path_lengths)

        # 2. 標準化分數 (Score)
        # score = 2 ** (-E(h(x)) / c)
        scores = 2 ** (-avg_path_lengths / self.c)
        
        return scores

# --- 範例使用 ---
if __name__ == '__main__':
    # 創建一個包含正常點和少量異常點的數據集
    rng = np.random.RandomState(42)
    
    # 正常點 (密集區域)
    X_normal = 0.3 * rng.randn(80, 2) + 2
    
    # 異常點 (稀疏區域)
    X_outliers = rng.uniform(low=-4, high=4, size=(20, 2))
    
    X_combined = np.r_[X_normal, X_outliers]
    
    # 混合數據集並隨機打亂
    rng.shuffle(X_combined)
    
    # 初始化並訓練 Isolation Forest
    # n_estimators=100 (100 棵樹)
    # max_samples=256 (每棵樹使用 256 個子樣本)
    model = IsolationForest(n_estimators=100, max_samples=100)
    print("--- 開始訓練 Isolation Forest ---")
    model.fit(X_combined)
    print("訓練完成。")
    
    # 進行預測，獲得異常分數
    anomaly_scores = model.decision_function(X_combined)
    
    # 設定閾值 (例如，取 top 20% 作為異常點)
    threshold = np.percentile(anomaly_scores, 80)
    
    # 判斷結果
    predictions = anomaly_scores > threshold
    
    # 輸出結果
    print(f"\n平均異常分數 (前 5 筆): {anomaly_scores[:5]}")
    print(f"異常分數閾值 (Top 20%): {threshold:.4f}")
    
    # 統計被標記為異常的點
    n_anomalies = np.sum(predictions)
    print(f"\n總樣本數: {len(X_combined)}")
    print(f"預測為異常點的數量 (分數 > 閾值): {n_anomalies}")
    
    # 輸出最異常的 5 個樣本的分數
    sorted_indices = np.argsort(anomaly_scores)[::-1] # 降序排序
    top_5_indices = sorted_indices[:5]
    
    print("\n--- 最異常的 5 個樣本 ---")
    for idx in top_5_indices:
        print(f"樣本索引 {idx}: 分數={anomaly_scores[idx]:.4f} (原始座標={X_combined[idx]})")
```
