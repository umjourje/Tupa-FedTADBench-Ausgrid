import pandas as pd
import numpy as np

import torch
import torch.nn as nn
import torchmetrics

from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

import lightning as L
import pytorch_lightning as pl

from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score
from sklearn.preprocessing import MinMaxScaler



class LSTMAE_Lightning(pl.LightningModule):
    def __init__(self, n_features: int, hidden_size: int,
                 n_layers: tuple, use_bias: tuple, dropout: tuple):
        super().__init__()
        self.n_features = n_features
        self.hidden_size = hidden_size

        self.n_layers = n_layers
        self.use_bias = use_bias
        self.dropout = dropout

        self.encoder = nn.LSTM(self.n_features, self.hidden_size, batch_first=True,
                               num_layers=self.n_layers[0], bias=self.use_bias[0], dropout=self.dropout[0])
        self.decoder = nn.LSTM(self.n_features, self.hidden_size, batch_first=True,
                               num_layers=self.n_layers[1], bias=self.use_bias[1], dropout=self.dropout[1])
        self.hidden2output = nn.Linear(self.hidden_size, self.n_features)

        self.automatic_optimization = False
        
        self.scores = []
        self.ys = []

    def forward(self, batch, Pi=None, priors_corr=None, prior_test=None):
        batch_size = batch.shape[0]

        enc_hidden = self._init_hidden(batch_size)  # initialization with zero
        _, enc_hidden = self.encoder(batch.float(), enc_hidden)  # .float() here or .double() for the model

        dec_hidden = enc_hidden

        output = torch.zeros_like(batch)
        for i in reversed(range(batch.shape[1])):
            output[:, i, :] = self.hidden2output(dec_hidden[0][0, :])

            if self.training:
                _, dec_hidden = self.decoder(batch[:, i].unsqueeze(1).float(), dec_hidden)
            else:
                _, dec_hidden = self.decoder(output[:, i].unsqueeze(1), dec_hidden)

        output_flatten = output.view((output.shape[0], output.shape[1] * output.shape[2]))
        batch_flatten = batch.view((batch.shape[0], batch.shape[1] * batch.shape[2]))
        
        rec_err = torch.abs(output_flatten ** 2 - batch_flatten ** 2)
        rec_err = torch.sum(rec_err, dim=1)
        
        output = output[:, -1, :]
        return enc_hidden[1][-1], rec_err, output

    def _init_hidden(self, batch_size):
        return (torch.zeros(self.n_layers[0], batch_size, self.hidden_size),
                torch.zeros(self.n_layers[0], batch_size, self.hidden_size))

    def training_step(self, batch, batch_idx):
        opt = self.optimizers()
        opt.zero_grad()                       #### Trainer
        
        x, y = batch
        x = x.to(self.device)
        y = y.to(self.device)
        feature, logits, output = self.forward(x)
        
        loss = F.mse_loss(output, y)
        loss.backward()                             #### Trainer
        opt.step()
        
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        x = x.to(self.device)
        y = y.to(self.device)
        feature, logits, output = self.forward(x)
        loss = F.mse_loss(feature, y)
        
        # accu = accuracy_score(output, y)
        self.log('train_val_loss', loss)
        # self.log('train_val_accuracy', accu)
        return loss
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        x = x.to(self.device)
        y = y.to(self.device)

        feature, logits, output = self.forward(x)
        loss = F.mse_loss(output, y)

    
        self.scores.append(logits.detach().cpu().numpy())
        self.ys.append(y.detach().cpu().numpy())
        
        if (len(y.unique())==1):
            self.ys.append(np.array(1.))

        scores = np.concatenate(self.scores, axis=0)
        ys = np.concatenate(self.ys, axis=0)

        print("----------------------------------------------------")
        print("shape ys: ", ys.shape)
        print("y_true: ", y.unique())

        # if len(scores.shape) == 2:
        #     scores = np.squeeze(scores, axis=1)
        # if len(ys.shape) == 2:
        #     ys = np.squeeze(ys, axis=1)

        auc_roc = roc_auc_score(ys, scores)
        ap = average_precision_score(ys, scores)
        accu = accuracy_score(output, y)

        self.log('test_loss', loss, prog_bar=True)
        self.log('test_auc_roc', auc_roc)
        self.log('test_avg_precision', ap)
        self.log('test_accuracy', accu)
        # self.log('test_accuracy', accu, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters())
    
    def backward(self, loss):
        loss.backward()

    def on_before_zero_grad(self, *args, **kwargs):
        # Ensure that the hidden state is on the same device as the model
        self.hidden2output = self.hidden2output.to(self.device)

    def on_train_batch_start(self, *args, **kwargs):
        # Ensure that the hidden state is on the same device as the model
        self.hidden2output = self.hidden2output.to(self.device)


class Ausgrid_Dataset(Dataset):

    def __init__(self, dataidxs=None, train_val_test='train', transform=None, target_transform=None, download=False, window_len=5, scaler=MinMaxScaler(feature_range=(0, 1)), client=0):

        self.dataidxs = dataidxs
        self.trainValTest = train_val_test
        self.transform = transform
        self.target_transform = target_transform
        self.download = download
        self.window_len = window_len
        self.scaler = scaler
        self.client_num = client # 10 max for ausgrid

        self.data, self.target = self.__build_truncated_dataset__()

    def __build_truncated_dataset__(self):

        base_path = '/home/labnet/Documents/JulianaPiaz/split_dataset'
            
        if (self.trainValTest=='train'):
            data = pd.read_csv(base_path + f'/train_{self.client_num}.csv')
            data = data.values.astype(np.float32)
            data = np.nan_to_num(data)
            data = self.scaler.fit_transform(data)
            target = data.copy()
        elif(self.trainValTest=='val'):
            data = pd.read_csv(base_path + f'/val_{self.client_num}.csv')
            data = data.values.astype(np.float32)
            data = np.nan_to_num(data)
            data = self.scaler.transform(data)
            val_target_path = f'/home/labnet/Documents/JulianaPiaz/split_dataset/val_label_{self.client_num}.csv'
            target_csv = pd.read_csv(val_target_path)
            target = target_csv.values
            target = target.astype(np.float32)
        elif(self.trainValTest=='test'):
            data = pd.read_csv(base_path + f'/test_{self.client_num}.csv')
            data = data.values.astype(np.float32)
            data = np.nan_to_num(data)
            data = self.scaler.transform(data)
            test_target_path = f'/home/labnet/Documents/JulianaPiaz/split_dataset/test_label_{self.client_num}.csv'
            target_csv = pd.read_csv(test_target_path)
            target = target_csv.values
            target = target.astype(np.float32)
        else:
            print("Error loading Ausgrid Dataset.")
            return 17

        if self.dataidxs:
            data = data[self.dataidxs]
            target = target[self.dataidxs]

        return data, target

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        if index + 1 - self.window_len < 0:
            data = self.data[0: index + 1]
            delta = self.window_len - data.shape[0]
            data0 = data[0]
            if len(data0.shape) == 1:
                data0 = data0[np.newaxis, :]
            data0 = np.repeat(data0, delta, axis=0)
            # print(data0.shape, data.shape)
            data = np.concatenate((data0, data), axis=0)
        else:
            data = self.data[index + 1 - self.window_len: index + 1]
        target = self.target[index]

        return data, target

    def __len__(self):
        return self.data.shape[0]


def load_data(node_id, batch_size, window_len):
    
    train_data = Ausgrid_Dataset(train_val_test='train', window_len=window_len, client=node_id)
    val_data = Ausgrid_Dataset(train_val_test='val', window_len=window_len, client=node_id)
    test_data = Ausgrid_Dataset(train_val_test='test', window_len=window_len, client=node_id)

    trainloader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        # persistent_workers=True,
        # prefetch_factor=4,
        num_workers=2,
        # drop_last=True
        drop_last=False
    )

    valoader = DataLoader(
        val_data,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
        # persistent_workers=True,
        # prefetch_factor=4,
        num_workers=2,
        drop_last=False
    )

    testloader = DataLoader(
        test_data,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
        # persistent_workers=True,
        # prefetch_factor=4,
        num_workers=2,
        drop_last=False
    )
    return trainloader, valoader, testloader



def main() -> None:
    """Centralized training."""

    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    checkpoint_path = '/home/labnet/Documents/JulianaPiaz/quickstart-pytorch-lightning/checkpoints/model/'

    args = {'n_features':7, 'dataset_name': 'ausgrid', 'epochs': 1, 'batch_size': 64, 
            'lr': 0.001, 'hidden_size': 32, 'n_layers': (2, 2), 'use_bias': (True, True), 
            'dropout': (0, 0), 'criterion': nn.MSELoss(), 
            'random_seed': 42, 'window_len': 30}
    
    # Load data
    train_loader, val_loader, test_loader = load_data(0, args['batch_size'], args['window_len'])

    # Load model
    # model = LSTMAE_Lightning.load_from_checkpoint(checkpoint_path+"checkpoint.ckpt")
    model = LSTMAE_Lightning(n_features=7, hidden_size=args['hidden_size'],
                   n_layers=args['n_layers'], use_bias=args['use_bias'], 
                   dropout=args['dropout'])


    # Train
    trainer = pl.Trainer(max_epochs=args['epochs'], accelerator='cpu', default_root_dir=checkpoint_path)
    trainer.fit(model, train_loader)

    # Validation
    return_val = trainer.validate(model, val_loader)
    
    print("=======================================================")
    print(return_val)
    print("=======================================================")
    
    # Test
    return_test = trainer.test(model, test_loader)

    print("=======================================================")
    print(return_test)
    print("=======================================================")
    

if __name__ == '__main__':
    main()
