import torch
import torch.nn as nn
import torch.optim as optim

model = nn.Linear(1, 1)
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

x = torch.tensor([[1.0], [2.0], [3.0], [4.0]])
y = torch.tensor([[3.0], [5.0], [7.0], [9.0]])

for epoch in range(1000):
    y_pred = model(x)
    loss = criterion(y_pred, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

print("预测 x=5:", model(torch.tensor([[5.0]])).item())
print("期望值: 11.0")
print("最终 loss:", loss.item())