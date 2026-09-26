import torch
import torch.nn as nn
import torch.optim as optim

x_data = torch.tensor([[1.0],[2.0],[3.0],[4.0],[5.0]])
y_data = torch.tensor([[5.0],[8.0],[11.0],[14.0],[17.0]])

model = nn.Linear(1,1)
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr = 0.01)

for epoch in range(5000):
	y_pred = model(x_data)
	loss = criterion(y_pred, y_data)
	optimizer.zero_grad()
	loss.backward()
	optimizer.step()

w = model.weight.item()
b = model.bias.item()
print(f"w = {w:.4f}, b = {b:.4f}")