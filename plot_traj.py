import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns
sns.set_theme()

import torch 
import torch.nn as nn 
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
torch.set_default_dtype(torch.float32)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device} device")

# GRU Baseline
class ToyGRU(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(ToyGRU, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.gru = nn.GRU(input_size=hidden_size, hidden_size=hidden_size, num_layers=2,batch_first=True)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, output_size)
        self.activation = nn.ReLU()
    
    def forward(self, x):
        batch_size = x.shape[0]
        out = self.activation(self.linear1(x))
        _, hn = self.gru(out.view(batch_size, 1, -1))
        out = self.activation(self.linear2(hn[0]))
        out = self.linear3(out)
        return out
   
# ToyGRU Inputs
input_size = 12
hidden_size = 512
output_size = 1

model = ToyGRU(input_size, hidden_size, output_size).to(device)
model.load_state_dict(torch.load('./weights_gru.pth'))
model.eval()

# Inference
test_data = np.load("./test_data.npy")

# Load the Test Dataset
test_set = torch.from_numpy(test_data)

# Feature and Ground Truth Matrix
X = test_set[:, :-1].to(device)
y = test_set[:, -1].reshape(-1, 1).to(device)

with torch.no_grad():
    test_pred = model(X)

t = np.linspace(0, 1, X.shape[0])

plt.figure( figsize=(18, 12))
plt.plot(t, y.cpu().numpy(), label="Ground Truth", color="blue")
plt.plot(t, test_pred.cpu().numpy(), label="Predictions", color="orange")
plt.legend(loc=1 ,prop={'size': 12})
plt.xlim(0, 0.05)
# plt.ylim(-10, 10)
plt.xlabel("t")
plt.ylabel("Angular Velocity")
plt.show()