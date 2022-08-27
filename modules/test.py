from model import LSTM
import torch
sample_input = torch.zeros(20, 200, dtype=torch.int)
sample_len = [200]*20

print("Loading Model...\n")
model = LSTM(200, 512, 300, 2, 2, True, 0.5, 5)

print("Doing Forward Pass..\n")
output = model(sample_input, sample_len)
print(output.shape)