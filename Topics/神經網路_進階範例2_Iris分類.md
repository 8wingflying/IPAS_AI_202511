# 神經網路_進階範例2_Iris分類
- 使用 Python 手刻多層感知機（MLP, Multi-Layer Perceptron）**進行 Iris 資料集分類的完整範例，包含：
  - ✅ 使用物件導向 (OOP) 撰寫
  - ✅ 實作 Forward / Backpropagation
  - ✅ 分類結果與準確率分析
  - ✅ 可視化訓練 Loss 曲線
```python
# 一、載入資料與前處理
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import matplotlib.pyplot as plt

# 載入 Iris 資料集
iris = load_iris()
X = iris.data       # 特徵: sepal length, sepal width, petal length, petal width
y = iris.target.reshape(-1, 1)  # 類別: 0, 1, 2

# One-Hot 編碼 (轉成三類)
encoder = OneHotEncoder(sparse_output=False)
y = encoder.fit_transform(y)

# 標準化特徵
scaler = StandardScaler()
X = scaler.fit_transform(X)

# 分割訓練/測試資料
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 🧠 二、以物件導向撰寫多層感知機（含反向傳播）
class MLP:
    """
    多層感知機 (Multi-Layer Perceptron) for Classification
    使用 Sigmoid 隱藏層 + Softmax 輸出層
    """
    def __init__(self, input_size, hidden_size, output_size, lr=0.01, epochs=1000):
        self.lr = lr
        self.epochs = epochs

        # 權重初始化 (Xavier initialization)
        self.W1 = np.random.randn(input_size, hidden_size) * np.sqrt(1. / input_size)
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) * np.sqrt(1. / hidden_size)
        self.b2 = np.zeros((1, output_size))

        self.loss_history = []

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    
    def sigmoid_derivative(self, a):
        return a * (1 - a)
    
    def softmax(self, z):
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)
    
    def cross_entropy_loss(self, y_true, y_pred):
        n = y_true.shape[0]
        return -np.sum(y_true * np.log(y_pred + 1e-8)) / n
    
    def forward(self, X):
        # Forward propagation
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = self.sigmoid(self.z1)
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = self.softmax(self.z2)
        return self.a2

    def backward(self, X, y_true):
        # Backpropagation
        n = X.shape[0]
        delta2 = self.a2 - y_true
        dW2 = np.dot(self.a1.T, delta2) / n
        db2 = np.sum(delta2, axis=0, keepdims=True) / n

        delta1 = np.dot(delta2, self.W2.T) * self.sigmoid_derivative(self.a1)
        dW1 = np.dot(X.T, delta1) / n
        db1 = np.sum(delta1, axis=0, keepdims=True) / n

        # 更新權重
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2

    def fit(self, X, y):
        for epoch in range(self.epochs):
            y_pred = self.forward(X)
            loss = self.cross_entropy_loss(y, y_pred)
            self.loss_history.append(loss)
            self.backward(X, y)

            if epoch % 100 == 0:
                print(f"Epoch {epoch}, Loss = {loss:.4f}")

    def predict(self, X):
        y_pred = self.forward(X)
        return np.argmax(y_pred, axis=1)

# 🧮 三、訓練模型與評估準確率
# 建立模型
mlp = MLP(input_size=4, hidden_size=8, output_size=3, lr=0.05, epochs=1000)
mlp.fit(X_train, y_train)

# 預測
y_pred = mlp.predict(X_test)
y_true = np.argmax(y_test, axis=1)

accuracy = np.mean(y_pred == y_true)
print(f"分類準確率: {accuracy*100:.2f}%")

# 📈 四、視覺化損失變化
plt.plot(mlp.loss_history, label="Training Loss")
plt.title("MLP Training Loss (Iris Classification)")
plt.xlabel("Epoch")
plt.ylabel("Cross-Entropy Loss")
plt.legend()
plt.show()
```
