## Day 4：NumPy 基本方法说明

### 1. 数组创建

#### 1.1 `np.array()`
**作用**：把 Python 的列表、元组等序列数据转换成 NumPy 数组（`ndarray`），是最基础的数组构造方式。

**基本用法**：

```python
import numpy as np

# 一维数组
a = np.array([1, 2, 3])

# 二维数组（矩阵）
matrix = np.array([[1, 2, 3],
                   [4, 5, 6]])

# 指定数据类型
float_arr = np.array([1, 2, 3], dtype=float)
```

#### 1.2 `np.arange()`
**作用**：生成等差序列的一维数组，类似 Python 的内置 `range()`，但返回的是 NumPy 数组。

**参数**：
- `start`：起始值，默认 0。
- `stop`：终止值（不包含）。
- `step`：步长，默认 1。

**基本用法**：

```python
import numpy as np

arr = np.arange(0, 10, 2)  # [0, 2, 4, 6, 8]
```

#### 1.3 `np.linspace()`
**作用**：在给定区间内生成指定个数的等间距点，常用于画图、数值实验。

**参数**：
- `start`：起始值。
- `stop`：结束值（默认包含）。
- `num`：要生成的点的个数，默认 50。

**基本用法**：

```python
import numpy as np

arr = np.linspace(0, 1, 5)  # [0.  , 0.25, 0.5 , 0.75, 1.  ]
```

---

### 2. 向量化操作

**概念**：
在 NumPy 中，对数组进行加减乘除等运算时，可以直接写成“数组 和 数组”的形式，而不用写显式的 for 循环，这种写法叫**向量化（vectorization）**。

**好处**：
- 写法简洁。
- 底层使用 C 实现，通常比 Python 循环快很多。

**示例：收益率向量化计算**（对应路线图中的练习 3）：

```python
import numpy as np

prices = np.array([100, 102, 101, 105, 103])
# prices[1:] = 从索引 1 开始到结尾
# prices[:-1] = 从“第 1 个”开始，到“倒数第 2 个”为止（不包含最后一个）。
# 相邻价格差 diff = 后一天价格 - 前一天价格
diff = prices[1:] - prices[:-1]

# 向量化计算收益率
returns = diff / prices[:-1]
```

这里 `prices[1:]` 和 `prices[:-1]` 都是数组，`diff / prices[:-1]` 这一行没有任何显式 for 循环，但会对每一个元素做除法运算，得到逐日收益率数组。

---

### 3. 常用统计函数

以下函数都是对数组数据进行统计计算，支持通过 `axis` 参数在不同行/列方向上聚合。

#### 3.1 `np.mean()`
**作用**：计算平均值（均值）。

```python
import numpy as np

data = np.array([1, 2, 3, 4, 5])
mean_val = np.mean(data)  # 3.0
```

#### 3.2 `np.std()`
**作用**：计算标准差，用来衡量数据波动程度，在量化中常用来衡量波动率。

```python
std_val = np.std(data)
```

#### 3.3 `np.max()` / `np.min()`
**作用**：分别计算数组中的最大值和最小值。

```python
max_val = np.max(data)
min_val = np.min(data)
```

#### 3.4 在练习中的应用

对应路线图中的练习 2：

```python
import numpy as np

data = np.random.normal(loc=0, scale=1, size=100)

print(f"均值: {np.mean(data):.3f}")
print(f"标准差: {np.std(data):.3f}")
print(f"最大值: {np.max(data):.3f}")
print(f"最小值: {np.min(data):.3f}")
```

---

### 4. 随机数生成

#### 4.1 `np.random.normal()`
**作用**：生成服从**正态分布（高斯分布）**的随机数，常用于模拟收益率、噪声等。

**参数**：
- `loc`：均值（μ）。
- `scale`：标准差（σ），控制波动大小。
- `size`：生成的样本数量或形状。

**基本用法**（对应练习 1）：

```python
import numpy as np

data = np.random.normal(loc=0, scale=1, size=100)
```

#### 4.2 `np.random.uniform()`
**作用**：生成服从**均匀分布**的随机数，即在区间 `[low, high)` 上每个值出现的概率相同。

**参数**：
- `low`：下界，默认 0.0。
- `high`：上界（不包含），默认 1.0。
- `size`：生成的样本数量或形状。

**基本用法**：

```python
import numpy as np

u = np.random.uniform(low=0, high=1, size=100)  # 100 个 [0,1) 上均匀分布的数
```

---

### 5. 小结

- `np.array()`：把 Python 序列转换成 NumPy 数组，是一切的基础。
- `np.arange()`：按步长生成等差数组，类似 `range()`。
- `np.linspace()`：在区间内生成固定个数的等间距点，常用于画图。
- 向量化操作：直接在数组之间做运算，写法简洁、运行高效。
- `mean`、`std`、`max`、`min`：常用统计量，用于描述数据的“中心”和“波动范围”。
- `np.random.normal()`：生成正态分布随机数，常用于收益率、噪声模拟。
- `np.random.uniform()`：生成均匀分布随机数，适合做简单的随机模拟或初始化。

