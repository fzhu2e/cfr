import torch
from torch import nn
from torch.utils import data
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from . import utils

def sliding_tensor(input_tensor, window=3, step=1):
    input = input_tensor.numpy()
    input_win = np.zeros([(input.shape[0]-window+1)//step, window, input.shape[1]])
    for idx, i in enumerate(np.arange(window, input.shape[0], step)):
        input_win[idx] = input[i-window:i]

    return torch.from_numpy(input_win)

    
def mk_tensors(list_of_series):
    ''' Make tensors for features and labels

    Args:
        list_of_series (list): a list of arrays

    '''
    features = torch.tensor(np.array(list_of_series)).T
    scaler = MinMaxScaler()
    scaler.fit(features)
    features = scaler.transform(features)
    features = torch.from_numpy(features).float()

    return features, scaler

def split_data(features, labels, train_frac=0.6, valid_frac=0.2, batch_size=1):
    train_size = int(train_frac * features.shape[0])
    train_data = data.TensorDataset(features[:train_size], labels[:train_size])
    train_loader = data.DataLoader(train_data, shuffle=True, batch_size=batch_size, drop_last=True)

    valid_size = int(0.2 * features.shape[0])
    valid_f = features[train_size:train_size+valid_size]
    valid_l = labels[train_size:train_size+valid_size]

    test_f = features[train_size+valid_size:]
    test_l = labels[train_size+valid_size:]

    output_dict = {
        'train_loader': train_loader,
        'valid_f': valid_f,
        'valid_l': valid_l,
        'test_f': test_f,
        'test_l': test_l,
    }
    return output_dict

def get_device(verbose=True):
    # If we have a GPU available, we'll set our device to GPU. We'll use this device variable later in our code.
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    if verbose: print('device:', device)
    return device
    
class LinearNet(nn.Module):
    def __init__(self, input_size=None, output_size=None, device='cpu'):
        super(LinearNet, self).__init__()
        self.device = device
        self.fc = nn.Linear(input_size, output_size)

    def forward(self, x):
        out = self.fc(x.to(self.device))
        return out
    

class GRUNet(nn.Module):
    def __init__(self, input_size=None, hidden_size=None, output_size=None, num_layers=None, drop_prob=0.2, device='cpu'):
        super(GRUNet, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.device = device
        
        # input_size: batch size x seq length x n of feature 
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, dropout=drop_prob)

        self.fc = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
        
    def forward(self, x, h):
        out, h = self.gru(x, h)
        out = self.fc(self.relu(out))
        return out, h
    
    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        hidden = weight.new(self.num_layers, batch_size, self.hidden_size).zero_().to(self.device)
        return hidden


class LSTMNet(nn.Module):
    def __init__(self, input_size=None, hidden_size=None, output_size=None, num_layers=None, drop_prob=0.2, device='cpu'):
        super(LSTMNet, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.device = device
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=drop_prob)
        self.fc = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
        
    def forward(self, x, h):
        out, h = self.lstm(x, h)
        out = self.fc(self.relu(out))
        return out, h
    
    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        hidden = (weight.new(self.num_layers, batch_size, self.hidden_size).zero_().to(self.device),
                  weight.new(self.num_layers, batch_size, self.hidden_size).zero_().to(self.device))
        return hidden
        

def train_model(train_loader, valid_f, valid_l, lr, model_type,
                hidden_size=256, output_size=1, num_layers=2,
                criterion=None, optimizer=None, save_path=None, verbose=True, max_epochs=100):

    device = get_device()
    batch_size, _, input_size = next(iter(train_loader))[0].shape

    model_dict = {
        'GRU': GRUNet(input_size, hidden_size=hidden_size, output_size=output_size, num_layers=num_layers, device=device),
        'LSTM': LSTMNet(input_size, hidden_size=hidden_size, output_size=output_size, num_layers=num_layers, device=device),
        'Linear': LinearNet(input_size, output_size=output_size, device=device),
    }
    model = model_dict[model_type]
    model.to(device)

    criterion = nn.MSELoss() if criterion is None else criterion
    optimizer = torch.optim.Adam(model.parameters(), lr=lr) if optimizer is None else optimizer

    model.train()
    if verbose: print(f'Starting training of "{model_type}" model.')

    # Start training loop
    train_loss = []
    valid_loss = []
    min_valid_loss = 1e9

    for epoch in range(1, max_epochs+1):
        if model_type in ['GRU', 'LSTM']:
            h = model.init_hidden(batch_size)

        avg_loss = 0.
        counter = 0
        for x, label in train_loader:
            counter += 1
            if model_type == 'GRU':
                h = h.data
            elif model_type == 'LSTM':
                h = tuple([e.data for e in h])
            model.zero_grad()
            
            if model_type in ['GRU', 'LSTM']:
                out, h = model(x.to(device).float(), h)
            else:
                out = model(x.to(device).float())

            loss = criterion(out[:,-1], label.to(device).float()[:,-1])
            loss.backward()
            optimizer.step()
            avg_loss += loss.item()

        avg_loss = avg_loss/len(train_loader)
        train_loss.append(avg_loss)

        if model_type in ['GRU', 'LSTM']:
            h = model.init_hidden(valid_f.shape[0])
            out, h = model(valid_f.to(device).float(), h)
        else:
            out = model(valid_f.to(device).float())

        loss = criterion(out[:,-1], valid_l.to(device).float()[:,-1]).item()
        valid_loss.append(loss)

        if loss < min_valid_loss:
            min_valid_loss = np.copy(loss)
            optim_model = model
            optim_epoch = epoch

        print(f'Epoch {epoch}/{max_epochs} - Train loss: {avg_loss}, Valid loss: {loss}')

    print('Optimal epoch:', optim_epoch)

    output_dict = {
        'optim_model': optim_model,
        'train_loss': train_loss,
        'valid_loss': valid_loss,
        'optim_epoch': optim_epoch,
    }

    if save_path is not None:
        torch.save([optim_model, train_loss, valid_loss, optim_epoch], save_path)


    return output_dict

def eval_model(model, features, labels, scaler, model_type=None):
    device = get_device()
    model.eval()
    if model_type in ['GRU', 'LSTM']:
        h = model.init_hidden(features.shape[0])
        out, h = model(features.float().to(device), h)
    else:
        out = model(features.float().to(device))
    
    pred = scaler.inverse_transform(out[:, -1].detach().cpu().numpy()).squeeze()
    truth = scaler.inverse_transform(labels[:, -1].detach().cpu().numpy()).squeeze()

    r = np.corrcoef(truth, pred)[1, 0]
    ce = utils.coefficient_efficiency(truth, pred)

    output_dict = {
        'pred': pred,
        'truth': truth,
        'corr': r,
        'CE': ce,
    }

    return output_dict