import pandas as pd
import numpy as np

import torch
import torch.nn as nn
import torchmetrics

from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

import lightning as L
import pytorch_lightning as pl
from lightning.pytorch.loggers import TensorBoardLogger


from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score
from sklearn.preprocessing import MinMaxScaler



class LSTMAE_Lightning(pl.LightningModule):
    def __init__(self, n_features: int, hidden_size: int,
                 n_layers: tuple, use_bias: tuple, 
                 dropout: tuple):
        super().__init__()
        self.n_features = n_features
        self.hidden_size = hidden_size

        self.n_layers = n_layers
        self.use_bias = use_bias
        self.dropout = dropout

        self.encoder = nn.LSTM(self.n_features, self.hidden_size, batch_first=True, device='cpu',
                               num_layers=self.n_layers[0], bias=self.use_bias[0], dropout=self.dropout[0])
        self.decoder = nn.LSTM(self.n_features, self.hidden_size, batch_first=True, device='cpu',
                               num_layers=self.n_layers[1], bias=self.use_bias[1], dropout=self.dropout[1])
        self.hidden2output = nn.Linear(self.hidden_size, self.n_features, device='cpu')

        self.automatic_optimization = True
        
        self.scores = []
        self.ys = []
        self.training_step_outputs = []
        self.test_step_outputs = []
        self.output_model = None

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
        x, y = batch

        feature, logits, output = self.forward(x)
        loss = F.mse_loss(output, y)
        loss_bin = F.binary_cross_entropy(torch.sigmoid(output), torch.sigmoid(y))
        
        self.training_step_outputs.append(loss)
        self.log('train_loss_step', loss)
        self.log('train_BCE_loss', loss_bin)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch

        feature, logits, output = self.forward(x)
        loss = F.mse_loss(output, y)
        loss_bin = F.binary_cross_entropy(torch.sigmoid(logits), torch.sigmoid(y))
        
        self.scores.append(logits.detach().cpu().numpy())
        self.ys.append(y.detach().cpu().numpy())
        

        self.logger.experiment.add_scalar('val_loss', loss)
        self.logger.experiment.add_scalar('val_loss_bin', loss_bin)
        return loss, loss_bin
    
    def test_step(self, batch, batch_idx):
        x, y = batch

        feature, logits, output = self.forward(x)
        loss = F.mse_loss(output, y)
        # loss_bin = F.binary_cross_entropy(torch.sigmoid(output), torch.sigmoid(y))

        self.test_step_outputs.append(loss)
        self.scores.append(logits.detach().cpu().numpy())
        self.ys.append(y.detach().cpu().numpy())

        self.log('test_loss', loss, prog_bar=True)
        self.logger.experiment.add_scalar('test_loss', loss)
        # self.logger.experiment.add_scalar('test_loss_bin', loss_bin)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)
    
    def backward(self, loss):
        loss.backward()

    def on_before_zero_grad(self, *args, **kwargs):
        # Ensure that the hidden state is on the same device as the model
        self.hidden2output = self.hidden2output

    def on_train_batch_start(self, *args, **kwargs):
        # Ensure that the hidden state is on the same device as the model
        self.hidden2output = self.hidden2output

    
    def on_train_epoch_end(self):
        epoch_mean = torch.stack(self.training_step_outputs).mean()
        tensorboard_logs = {'train_loss_1':epoch_mean, 'step':self.current_epoch}
        
        #self.log("training_epoch_mean", epoch_mean)
        #self.logger.experiment.add_scalar("tb_training_epoch_mean", epoch_mean, self.current_epoch)
        self.training_step_outputs.clear()
        
        #print('++++++++++++++++++++++++++++++++++++++')
        #print('época atual:', self.current_epoch)
        #print('++++++++++++++++++++++++++++++++++++++')
        return {'loss':epoch_mean, 'log':tensorboard_logs}
        

    def on_validation_epoch_end(self):
        scores = np.concatenate(self.scores, axis=0)
        ys = np.concatenate(self.ys, axis=0)

        if len(scores.shape) == 2:
            scores = np.squeeze(scores, axis=1)
        if len(ys.shape) == 2:
            ys = np.squeeze(ys, axis=1)

        auc_roc = roc_auc_score(ys, scores)
        ap = average_precision_score(ys, scores)

        val_loss_mean = torch.stack(self.test_step_outputs).mean()
        tensorboard_logs = {'val_loss':val_loss_mean, 'step':self.current_epoch}
        
        self.logger.experiment.add_scalar('val_auc_roc', auc_roc)
        self.logger.experiment.add_scalar('val_avg_precision', ap)

        tensorboard_logs = {'val_auc_roc':auc_roc, 'val_avg_precision': ap, 'step':self.current_epoch}
        return {'val_auc':auc_roc, 'log':tensorboard_logs}


    def on_test_end(self):
        scores = np.concatenate(self.scores, axis=0)
        ys = np.concatenate(self.ys, axis=0)

        if len(scores.shape) == 2:
            scores = np.squeeze(scores, axis=1)
        if len(ys.shape) == 2:
            ys = np.squeeze(ys, axis=1)

        auc_roc = roc_auc_score(ys, scores)
        ap = average_precision_score(ys, scores)

        test_loss_mean = torch.stack(self.test_step_outputs).mean()
        tensorboard_logs = {'test_loss':test_loss_mean, 'step':self.current_epoch}
        
        self.logger.experiment.add_scalar('test_auc_roc', auc_roc)
        self.logger.experiment.add_scalar('test_avg_precision', ap)

        tensorboard_logs = {'test_auc_roc':auc_roc, 'test_avg_precision': ap, 'step':self.current_epoch}
        return {'test_auc':auc_roc, 'log':tensorboard_logs}


    def calc_metrics(self):

        scores = np.concatenate(self.scores, axis=0)
        ys = np.concatenate(self.ys, axis=0)

        if len(scores.shape) == 2:
            scores = np.squeeze(scores, axis=1)
        if len(ys.shape) == 2:
            ys = np.squeeze(ys, axis=1)

        auc_roc = roc_auc_score(ys, scores)
        ap = average_precision_score(ys, scores)

        return auc_roc, ap, scores

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

        base_path = '/home/labnet/Documents/JulianaPiaz/split_dataset_normal'
            
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
            val_target_path = f'/home/labnet/Documents/JulianaPiaz/split_dataset_normal/val_label_{self.client_num}.csv'
            target_csv = pd.read_csv(val_target_path)
            target = target_csv.values
            target = target.astype(np.float32)
        elif(self.trainValTest=='test'):
            data = pd.read_csv(base_path + f'/test_{self.client_num}.csv')
            data = data.values.astype(np.float32)
            data = np.nan_to_num(data)
            data = self.scaler.transform(data)
            test_target_path = f'/home/labnet/Documents/JulianaPiaz/split_dataset_normal/test_label_{self.client_num}.csv'
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

    # device_obj = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # checkpoint_path = '/home/labnet/Documents/JulianaPiaz/quickstart-pytorch-lightning/checkpoints/model/'

    args = {'n_features':7, 'dataset_name': 'ausgrid', 'epochs': 10, 'batch_size': 32, 
            'lr': 0.001, 'hidden_size': 32, 'n_layers': (2, 2), 'use_bias': (True, True), 
            'dropout': (0, 0), 'criterion': nn.MSELoss(), 
            'random_seed': 42, 'window_len': 30}
    
    # Load data
    train_loader, val_loader, test_loader = load_data(9, args['batch_size'], args['window_len'])

    # Load model
    # model = LSTMAE_Lightning.load_from_checkpoint(checkpoint_path+"checkpoint.ckpt")
    model = LSTMAE_Lightning(n_features=7, hidden_size=args['hidden_size'],
                   n_layers=args['n_layers'], use_bias=args['use_bias'], 
                   dropout=args['dropout'])


    # Instantiate Logger and Trainer

    logger_tensorboard = TensorBoardLogger("tb_logs", name="trial_centralized")
    
    trainer = pl.Trainer(max_epochs=args['epochs'], accelerator='cpu', logger=logger_tensorboard)
    
    # Train
    trainer.fit(model, train_loader)

    # Validation
    # trainer.validate(model, val_loader)
    
    # Test
    trainer.test(model, test_loader)

    run_id = 30
    # Metrics
    model_save_path = '/home/labnet/Documents/JulianaPiaz/quickstart-pytorch-lightning/logs_metrics/models/' + args['dataset_name'] + '_centralized_model_run_client9_' + str(run_id) + '.pth'
    score_save_path = '/home/labnet/Documents/JulianaPiaz/quickstart-pytorch-lightning/logs_metrics/scores/' + args['dataset_name'] + '_centralized_score_run_' + str(run_id) + '.npy'

    # get metrics
    auc_roc_metric, avg_precicion_metric, scores_epoch = model.calc_metrics()
    print("=======================================================")
    print('auc-roc: ' + str(auc_roc_metric) + ' auc_pr: ' + str(avg_precicion_metric))
    print("=======================================================")
    
    best_auc_roc = 0
    best_ap = 0

    if auc_roc_metric > best_auc_roc:
            best_auc_roc = auc_roc_metric
            best_ap = avg_precicion_metric
            torch.save(model.state_dict(), model_save_path)
            np.save(score_save_path, scores_epoch)
            print(' update')
    else:
        print()
    

if __name__ == '__main__':
    main()
