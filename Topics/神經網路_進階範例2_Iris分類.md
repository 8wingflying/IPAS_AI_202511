# 神經網路_進階範例2_Iris分類
- 使用 Python 手刻多層感知機（MLP, Multi-Layer Perceptron）進行 Iris 資料集分類的完整範例，包含：
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
## 比較 Sigmoid、Tanh、ReLU 三種激活函數在 Iris 資料集分類的表現
- 顯示每種激活函數的：
  - 🧠 收斂過程（Loss 曲線）
  - 📊 最終準確率比較
```python
#一、資料準備（與前相同）
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# 載入 Iris 資料集
iris = load_iris()
X = iris.data
y = iris.target.reshape(-1, 1)

# One-Hot 編碼
encoder = OneHotEncoder(sparse_output=False)
y = encoder.fit_transform(y)

# 標準化
scaler = StandardScaler()
X = scaler.fit_transform(X)

# 資料分割
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#🧠 二、建立通用 MLP 類別（可切換激活函數）
class MLP_Activation:
    """
    多層感知機 (MLP) for Iris Classification
    可切換激活函數: 'sigmoid' / 'tanh' / 'relu'
    """
    def __init__(self, input_size, hidden_size, output_size, activation='sigmoid', lr=0.05, epochs=1000):
        self.lr = lr
        self.epochs = epochs
        self.activation_name = activation

        # 初始化權重
        self.W1 = np.random.randn(input_size, hidden_size) * np.sqrt(1. / input_size)
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) * np.sqrt(1. / hidden_size)
        self.b2 = np.zeros((1, output_size))

        self.loss_history = []

    # === 激活函數與導數 ===
    def _activate(self, z):
        if self.activation_name == 'sigmoid':
            return 1 / (1 + np.exp(-z))
        elif self.activation_name == 'tanh':
            return np.tanh(z)
        elif self.activation_name == 'relu':
            return np.maximum(0, z)
        else:
            raise ValueError("Unknown activation function")

    def _activate_derivative(self, a):
        if self.activation_name == 'sigmoid':
            return a * (1 - a)
        elif self.activation_name == 'tanh':
            return 1 - a ** 2
        elif self.activation_name == 'relu':
            return (a > 0).astype(float)

    def _softmax(self, z):
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    def _cross_entropy(self, y_true, y_pred):
        n = y_true.shape[0]
        return -np.sum(y_true * np.log(y_pred + 1e-8)) / n

    # === Forward / Backward ===
    def forward(self, X):
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = self._activate(self.z1)
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = self._softmax(self.z2)
        return self.a2

    def backward(self, X, y_true):
        n = X.shape[0]
        delta2 = self.a2 - y_true
        dW2 = np.dot(self.a1.T, delta2) / n
        db2 = np.sum(delta2, axis=0, keepdims=True) / n

        delta1 = np.dot(delta2, self.W2.T) * self._activate_derivative(self.a1)
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
            loss = self._cross_entropy(y, y_pred)
            self.loss_history.append(loss)
            self.backward(X, y)

    def predict(self, X):
        y_pred = self.forward(X)
        return np.argmax(y_pred, axis=1)

#🧮 三、訓練三種模型（Sigmoid / Tanh / ReLU）
activations = ['sigmoid', 'tanh', 'relu']
results = {}

for act in activations:
    print(f"\n=== 使用 {act} 激活函數訓練中 ===")
    model = MLP_Activation(input_size=4, hidden_size=8, output_size=3, activation=act, lr=0.05, epochs=1000)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_true = np.argmax(y_test, axis=1)
    acc = np.mean(y_pred == y_true)
    results[act] = {"model": model, "accuracy": acc}
    print(f"{act} 準確率: {acc*100:.2f}%")

# 📈 四、視覺化 Loss 曲線比較
plt.figure(figsize=(10, 6))
for act in activations:
    plt.plot(results[act]['model'].loss_history, label=f"{act} (acc={results[act]['accuracy']*100:.1f}%)")

plt.title("MLP Activation Function Comparison (Iris Classification)")
plt.xlabel("Epoch")
plt.ylabel("Cross-Entropy Loss")
plt.legend()
plt.show()
```
- ReLU 通常在隱藏層表現最佳，因為梯度不易消失。
- Tanh 在中小型資料集表現穩定，且較 Sigmoid 收斂快。
- Sigmoid 在多層時常有梯度消失問題，但適合初學展示非線性轉換概念。
