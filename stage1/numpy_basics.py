import numpy as np

# ============================================================
# 1. 创建数组
# ============================================================
print("=" * 50)
print("1. 创建数组")					

# 从列表创建
a = np.array([1, 2, 3, 4, 5])						#建立一维数组赋值给a
print("一维数组:", a)
# 创建全零、全一数组
zeros = np.zeros((3, 4))          # 3行4列			#应该是用python语法，调用函数
ones = np.ones((2, 3))            # 2行3列
print("全零数组:\n", zeros)
print("全一数组:\n", ones)

# 创建等差数列
range_arr = np.arange(0, 10, 2)   # 从0到10，步长2			#调用arange函数，
lin_arr = np.linspace(0, 1, 5)    # 从0到1，等分5个点			#调用linspace函数，将数值差平分成最后一个参数份，然后输出
print("arange:", range_arr)
print("linspace:", lin_arr)

# 随机数组
rand_arr = np.random.rand(2, 3)   # 2行3列，0到1均匀分布			#调用random函数
print("随机数组:\n", rand_arr)

# ============================================================
# 2. 数组属性
# ============================================================
print("\n" + "=" * 50)
print("2. 数组属性")

arr = np.array([[1, 2, 3], [4, 5, 6]])
print("数组:\n", arr)
print("形状 (shape):", arr.shape)      # (2, 3)			#结构体数组，调用函数成员
print("维度 (ndim):", arr.ndim)        # 2
print("元素个数 (size):", arr.size)    # 6
print("数据类型 (dtype):", arr.dtype)  # int64

# ============================================================
# 3. 索引和切片
# ============================================================
print("\n" + "=" * 50)
print("3. 索引和切片")

arr = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print("原数组:\n", arr)

# 取单个元素
print("arr[0, 1]:", arr[0, 1])        # 第0行第1列 -> 2

# 取一行
print("arr[1]:", arr[1])              # 第1行 -> [4, 5, 6]

# 取一列
print("arr[:, 2]:", arr[:, 2])        # 所有行第2列 -> [3, 6, 9]

# 切片
print("arr[0:2, 1:3]:\n", arr[0:2, 1:3])  # 前两行，后两列

# 布尔索引
mask = arr > 5
print("arr > 5 的掩码:\n", mask)
print("arr[arr > 5]:", arr[arr > 5])  # 所有大于5的元素

# ============================================================
# 4. 形状操作
# ============================================================
print("\n" + "=" * 50)
print("4. 形状操作")

arr = np.arange(12)
print("原数组:", arr)

# reshape
reshaped = arr.reshape(3, 4)
print("reshape(3, 4):\n", reshaped)

# 转置
transposed = reshaped.T
print("转置:\n", transposed)

# 展平
flattened = reshaped.flatten()
print("展平:", flattened)

# 增加维度
expanded = arr[np.newaxis, :]         # 变成 (1, 12)
print("增加维度后的形状:", expanded.shape)

# ============================================================
# 5. 广播机制
# ============================================================
print("\n" + "=" * 50)
print("5. 广播机制")

a = np.array([[1, 2, 3], [4, 5, 6]])
b = np.array([10, 20, 30])

# b 会广播到每一行
result = a + b
print("a + b:\n", result)

# 标量广播
print("a * 2:\n", a * 2)

# 列向量广播
c = np.array([[100], [200]])
print("a + c:\n", a + c)

# ============================================================
# 6. 矩阵运算
# ============================================================
print("\n" + "=" * 50)
print("6. 矩阵运算")

A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

# 逐元素乘法
print("逐元素乘法 A * B:\n", A * B)

# 矩阵乘法
print("矩阵乘法 A @ B:\n", A @ B)

# 转置
print("A 的转置:\n", A.T)

# 逆矩阵
print("A 的逆矩阵:\n", np.linalg.inv(A))

# 行列式
print("A 的行列式:", np.linalg.det(A))

# ============================================================
# 7. 统计操作
# ============================================================
print("\n" + "=" * 50)
print("7. 统计操作")

data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print("数据:\n", data)

print("总和:", np.sum(data))
print("按行求和:", np.sum(data, axis=1))
print("按列求和:", np.sum(data, axis=0))
print("均值:", np.mean(data))
print("标准差:", np.std(data))
print("最大值:", np.max(data))
print("最小值:", np.min(data))
print("最大值索引:", np.argmax(data))
print("最小值索引:", np.argmin(data))

# 归一化 (0-1)
normalized = (data - np.min(data)) / (np.max(data) - np.min(data))
print("归一化:\n", normalized)

# ============================================================
# 8. 常用函数
# ============================================================
print("\n" + "=" * 50)
print("8. 常用函数")

x = np.array([0, np.pi/2, np.pi])
print("sin(x):", np.sin(x))
print("cos(x):", np.cos(x))
print("exp(x):", np.exp(x))
print("sqrt(x):", np.sqrt(np.array([1, 4, 9, 16])))

# 拼接
a = np.array([1, 2])
b = np.array([3, 4])
print("拼接:", np.concatenate([a, b]))

# 堆叠
print("垂直堆叠:\n", np.vstack([a, b]))
print("水平堆叠:", np.hstack([a, b]))

# ============================================================
# 9. 随机数
# ============================================================
print("\n" + "=" * 50)
print("9. 随机数")

np.random.seed(42)  # 设置随机种子，保证可复现

print("均匀分布:", np.random.rand(5))
print("标准正态分布:", np.random.randn(5))
print("随机整数:", np.random.randint(0, 10, size=5))
print("随机选择:", np.random.choice([1, 2, 3, 4, 5], size=3))

# ============================================================
# 10. 实际应用示例：简单线性回归的梯度计算
# ============================================================
print("\n" + "=" * 50)
print("10. 实际应用：手动计算线性回归梯度")

# 数据：y = 2x + 1
x = np.array([1, 2, 3, 4, 5])
y = np.array([3, 5, 7, 9, 11])

# 初始化参数
w = 0.0
b = 0.0
lr = 0.01

# 手动梯度下降
for epoch in range(100):
    y_pred = w * x + b
    loss = np.mean((y_pred - y) ** 2)
    
    # 计算梯度
    dw = np.mean(2 * (y_pred - y) * x)
    db = np.mean(2 * (y_pred - y))
    
    # 更新参数
    w -= lr * dw
    b -= lr * db

print(f"训练后 w = {w:.4f}, b = {b:.4f}")
print(f"期望 w = 2.0, b = 1.0")

print("\n" + "=" * 50)
print("练习 1 完成！")

# ============================================================
# 11.创建5*5的单位矩阵
# ============================================================
identity = np.eye(5)
print("5*5单位矩阵:\n", identity)

# ============================================================
# 12.计算两个向量的点积
# ============================================================
v1 = np.array([1,2,3])
v2 = np.array([4,5,6])
dot = np.dot(v1,v2)
print("v1:",v1)
print("v2:",v2)
print("点积:",dot)

# ============================================================
# 13.找出数组中所以偶数
# ============================================================
arr = np.array([1,2,3,4,5,6,7,8,9,10])
mask = arr % 2 == 0
print("原数组:", arr)
print("布尔掩码:", mask)
print("所有偶数:", arr[mask])
