# 神經網路
- 使用感知機模型建構簡單邏輯閘運作
- 使用多層感知機建構NOR邏輯閘運作
- 監督式(機器學習)
  - 使用多層感知機進行分類
  - 使用多層感知機進行分類

## 使用感知機模型建構簡單邏輯閘運作
```python

```
## 使用多層感知機建構NOR邏輯閘運作
# 監督式(機器學習)
#### 使用感知機進行分類 
- 使用物件導向技術撰寫感知機類別class Perceptron
```python
import numpy as np

class Perceptron:
    """
    簡單感知機 (Perceptron) 模型實作
    """

    def __init__(self, learning_rate=0.1, n_iters=100):
        self.lr = learning_rate
        self.n_iters = n_iters
        self.activation_func = self._unit_step
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        """
        訓練 感知機模型
        X: (樣本數, 特徵數)
        y: 標籤(0 或 1)
        """
        n_samples, n_features = X.shape
        # 初始化權重與偏置 ==> 全部設為0
        self.weights = np.zeros(n_features)
        self.bias = 0

        # 將標籤轉為 -1 / 1（感知機更新規則）
        y_ = np.where(y <= 0, -1, 1)

        for _ in range(self.n_iters):
            for idx, x_i in enumerate(X):
                # 預測
                linear_output = np.dot(x_i, self.weights) + self.bias
                y_pred = self.activation_func(linear_output)
                # 更新規則：w ← w + lr * (y_true - y_pred) * x
                update = self.lr * (y_[idx] - y_pred)
                self.weights += update * x_i
                self.bias += update

    def predict(self, X):
        """
        使用訓練好的模型進行  預測
        """
        linear_output = np.dot(X, self.weights) + self.bias
        y_pred = self.activation_func(linear_output)
        return np.where(y_pred > 0, 1, 0)

    def _unit_step(self, x):
        """
        激活函數 Activation function ==> 使用階梯函數Step Function
        """
        return np.where(x >= 0, 1, -1)

# === 測試資料：邏輯 AND 運算 ===
X = np.array([[0,0], [0,1], [1,0], [1,1]])
y = np.array([0, 0, 0, 1])  # AND 運算結果

# 建立並訓練感知機模型
clf = Perceptron(learning_rate=0.1, n_iters=10)
clf.fit(X, y)

# 預測與結果
predictions = clf.predict(X)
print("預測結果:", predictions)
print("真實標籤:", y)
print("權重:", clf.weights)
print("偏置:", clf.bias)
```
```
預測結果: [0 0 0 1]
真實標籤: [0 0 0 1]
權重: [0.4 0.2]
偏置: -0.4000000000000001
```
#### 使用多層感知機進行XOR分類
```python
## 載入套件
#from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
#from sklearn import svm

## 輸入資料
y = [0, 1, 1, 0]
X = [[0, 0], [0, 1], [1, 0], [1, 1]]

## 建立 分類器clf 
clf = MLPClassifier(solver='lbfgs', activation='logistic', hidden_layer_sizes=(2,), max_iter=100, random_state=20)

## 訓練 分類器 ==> 使用fit()函數
clf.fit(X, y)

## 預測 ==> 使用predict()函數
predictions = clf.predict(X)

## 計算評估指標 ==> 使用score()函數
print('Accuracy: %s' % clf.score(X, y))

for i, p in enumerate(predictions[:10]):
    print('True: %s, Predicted: %s' % (y[i], p))
```
