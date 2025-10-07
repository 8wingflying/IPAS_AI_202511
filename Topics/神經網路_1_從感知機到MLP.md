# 神經網路
- 單純感知機模型(沒有學習功能)
  - 使用感知機模型建構簡單邏輯閘運作
  - 使用多層感知機建構NOR邏輯閘運作
- 監督式(機器學習) ==> 看看機器是如何`學習`
  - 使用多層感知機進行分類
  - 使用多層感知機進行回歸

# 單純感知機模型(沒有學習功能)
## 使用感知機模型建構簡單邏輯閘運作
- AND 邏輯閘的感知機模型
```python
import numpy as np


def AND(x1, x2):
    x = np.array([x1, x2])
    w = np.array([0.6, 0.4])
    b = -0.7
    tmp = np.sum(w*x) + b
    if tmp <= 0:
        return 0
    else:
        return 1

if __name__ == '__main__':
    for xs in [(0, 0), (1, 0), (0, 1), (1, 1)]:
        y = AND(xs[0], xs[1])
        print(str(xs) + " -> " + str(y))
```
- NAND 邏輯閘的感知機模型
```python
import numpy as np


def NAND(x1, x2):
    x = np.array([x1, x2])
    w = np.array([-0.6, -0.6])
    b = 0.8
    tmp = np.sum(w*x) + b
    if tmp <= 0:
        return 0
    else:
        return 1

if __name__ == '__main__':
    for xs in [(0, 0), (1, 0), (0, 1), (1, 1)]:
        y = NAND(xs[0], xs[1])
        print(str(xs) + " -> " + str(y))
```
- OR 邏輯閘的感知機模型
```python
import numpy as np


def OR(x1, x2):
    x = np.array([x1, x2])
    w = np.array([0.5, 0.5])
    b = -0.4
    tmp = np.sum(w*x) + b
    if tmp <= 0:
        return 0
    else:
        return 1

if __name__ == '__main__':
    for xs in [(0, 0), (1, 0), (0, 1), (1, 1)]:
        y = OR(xs[0], xs[1])
        print(str(xs) + " -> " + str(y))
```
- NOR 邏輯閘的感知機模型
```python
import numpy as np


def NOR(x1, x2):
    x = np.array([x1, x2])
    w = np.array([-0.5, -0.5])
    b = 0.4
    tmp = np.sum(w*x) + b
    if tmp <= 0:
        return 0
    else:
        return 1

if __name__ == '__main__':
    for xs in [(0, 0), (1, 0), (0, 1), (1, 1)]:
        y = NOR(xs[0], xs[1])
        print(str(xs) + " -> " + str(y))
```
## 使用多層感知機(MLP)建構NOR邏輯閘運作
```python

def XOR(x1, x2):
    s1 = NAND(x1, x2)
    s2 = OR(x1, x2)
    y = AND(s1, s2)
    return y

if __name__ == '__main__':
    for xs in [(0, 0), (1, 0), (0, 1), (1, 1)]:
        y = XOR(xs[0], xs[1])
        print(str(xs) + " -> " + str(y))
```
# 監督式(機器學習) ==> 分類--迴歸分析(Regression Analysis）
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
#### 使用感知機進行迴歸分析(Regression Analysis）
- CHATGPT:簡單建立線性回歸的資料 再使用python物件導向方式撰寫感知機 並分析資料
  - 建立簡單的線性資料集（例如 𝑦=3𝑥+5+噪音)
  - 以物件導向（OOP）方式實作感知機（Perceptron）模型進行迴歸分析
  - 分析訓練結果與可視化預測
```python
## 一、建立線性資料
import numpy as np
import matplotlib.pyplot as plt

# 產生線性資料
np.random.seed(42)
X = np.linspace(0, 10, 50).reshape(-1, 1)
y = 3 * X + 5 + np.random.randn(50, 1) * 2  # y = 3x + 5 + 噪音

plt.scatter(X, y, color='blue', label='Data')
plt.title("Linear Data Example")
plt.xlabel("X")
plt.ylabel("y")
plt.legend()
plt.show()

## 二、以物件導向方式實作感知機 (Perceptron Regression)

class PerceptronRegressor:
    """
    一個簡單的感知機迴歸模型 (類似線性迴歸)
    y_pred = w * x + b
    使用梯度下降 (Gradient Descent) 更新權重
    """
    def __init__(self, lr=0.01, epochs=1000):
        self.lr = lr
        self.epochs = epochs
        self.w = None
        self.b = None
        self.loss_history = []

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.w = np.zeros((n_features, 1))
        self.b = 0

        for epoch in range(self.epochs):
            y_pred = np.dot(X, self.w) + self.b
            error = y_pred - y
            loss = np.mean(error ** 2)

            # 記錄損失
            self.loss_history.append(loss)

            # 更新參數
            dw = (2 / n_samples) * np.dot(X.T, error)
            db = (2 / n_samples) * np.sum(error)
            self.w -= self.lr * dw
            self.b -= self.lr * db

    def predict(self, X):
        return np.dot(X, self.w) + self.b

## 三、訓練與分析

# 初始化並訓練模型
model = PerceptronRegressor(lr=0.01, epochs=1000)
model.fit(X, y)

# 預測
y_pred = model.predict(X)

# 顯示訓練結果
print(f"權重 w = {model.w.flatten()[0]:.3f}, 偏差 b = {model.b:.3f}")

# 可視化結果
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.scatter(X, y, label='Data')
plt.plot(X, y_pred, color='red', label='Model Prediction')
plt.title("Perceptron Regression Result")
plt.xlabel("X")
plt.ylabel("y")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(model.loss_history)
plt.title("Loss Curve")
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.show()
```
## 多層感知機的權重更新機制 ==> 反向傳播(backpropagation)
