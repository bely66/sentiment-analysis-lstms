import torchtext
import torch
from torch import optim, nn
import numpy as np

from src.utils import train, evaluate, count_parameters, initialize_weights
from src.data_loading import get_data_loaders
from modules.model import LSTM



train_dataloader, valid_dataloader, test_dataloader, vocab = get_data_loaders()

vocab_size = len(vocab)
embedding_dim = 300
hidden_dim = 300
output_dim = 2
n_layers = 2
bidirectional = True
dropout_rate = 0.5
pad_index = vocab['<pad>']

model = LSTM(vocab_size, embedding_dim, hidden_dim, output_dim, n_layers, bidirectional, dropout_rate, 
             pad_index)

print(f'The model has {count_parameters(model):,} trainable parameters')

model.apply(initialize_weights)

# Saving Vocabulary is important because the process isn't deterministic
torch.save(vocab, "lstm_vocab.pth")


vectors = torchtext.vocab.FastText()
pretrained_embedding = vectors.get_vecs_by_tokens(vocab.get_itos())
model.embeddings.weight.data = pretrained_embedding

lr = 5e-4

optimizer = optim.Adam(model.parameters(), lr=lr)
criterion = nn.CrossEntropyLoss()

device = torch.device('mps' if torch.cuda.is_available() else 'cpu')

model = model.to(device)
criterion = criterion.to(device)

n_epochs = 10
best_valid_loss = float('inf')

train_losses = []
train_accs = []
valid_losses = []
valid_accs = []

for epoch in range(n_epochs):

    train_loss, train_acc = train(train_dataloader, model, criterion, optimizer, device)
    valid_loss, valid_acc = evaluate(valid_dataloader, model, criterion, device)

    train_losses.extend(train_loss)
    train_accs.extend(train_acc)
    valid_losses.extend(valid_loss)
    valid_accs.extend(valid_acc)
    
    epoch_train_loss = np.mean(train_loss)
    epoch_train_acc = np.mean(train_acc)
    epoch_valid_loss = np.mean(valid_loss)
    epoch_valid_acc = np.mean(valid_acc)
    
    if epoch_valid_loss < best_valid_loss:
        best_valid_loss = epoch_valid_loss
        with torch.no_grad():
            torch.save(model, 'lstm.pth')
            scripted_model = torch.jit.script(model)
            scripted_model.save('lstm_scripted.pt')
    
    print(f'epoch: {epoch+1}')
    print(f'train_loss: {epoch_train_loss:.3f}, train_acc: {epoch_train_acc:.3f}')
    print(f'valid_loss: {epoch_valid_loss:.3f}, valid_acc: {epoch_valid_acc:.3f}')



