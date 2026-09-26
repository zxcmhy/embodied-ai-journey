#1.创建张量
import torch

a = torch.tensor([1,2,3,4,5])
b = torch.zeros((3,4))
c = torch.ones((2,3))
d = torch.arange(0,10,2)
e = torch.eye(3)

print(a)
print(b)
print(c)
print(d)
print(e)

#2.自动求导
x = torch.tensor([3.0], requires_grad=True)
y = x**2+2*x
y.backward()
print("dy/dx:", x.grad)

#3.手动线性回归（改小学习率）
x_data = torch.tensor([[1.0],[2.0],[3.0],[4.0],[5.0]])
y_data = torch.tensor([[5.0],[8.0],[11.0],[14.0],[17.0]])

w = torch.tensor([[0.0]], requires_grad=True)
b = torch.tensor([[0.0]], requires_grad=True)
lr = 0.005

# 先确认形状
y_pred_check = x_data @ w + b
print("y_pred.shape:", y_pred_chek.shape)

for epoch in range(5000):
	y_pred = x_data @ w + b
	loss = ((y_pred - y_data)**2).mean()
	loss.backward()
	with torch.no_grad():
		w -=lr * w.grad
		b -=lr * b.grad
	w.grad.zero_()
	b.grad.zero_()

print(f"w = {w.item():.4f}, b = {b.item():.4f}")
print(f"期望 w = 3.0, b = 2.0")