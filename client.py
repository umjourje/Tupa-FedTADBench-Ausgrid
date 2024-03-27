import argparse
from collections import OrderedDict
from typing import Dict, List, Tuple

import pytorch_lightning as pl
from lightning.pytorch.loggers import TensorBoardLogger

import torch
from datasets.utils.logging import disable_progress_bar

import flwr as fl
import numpy as np
import mnist
import ausgrid

disable_progress_bar()


class FlowerClient(fl.client.NumPyClient):
    def __init__(self, model, train_loader, val_loader, test_loader, logging):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.logger_tensorboard = logging

    def get_parameters(self, config) -> List[np.ndarray]:
        # Return model parameters as a list of NumPy ndarrays
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters: List[np.ndarray]) -> None:
        # Set model parameters from a list of NumPy ndarrays
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
    
        trainer = pl.Trainer(max_epochs=1, accelerator='cpu', logger=self.logger_tensorboard, default_root_dir='/home/labnet/Documents/JulianaPiaz/quickstart-pytorch-lightning/checkpoints/model/')
        trainer.fit(self.model, self.train_loader,)

        return self.get_parameters(config={}), 55000, {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)

        trainer = pl.Trainer(accelerator='cpu', logger=self.logger_tensorboard)
        results = trainer.test(self.model, self.test_loader)
        print("-------------------RESULTADOS-------------------")
        print(results)

        loss = results[0]["test_loss"]

        return loss, 10000, {"loss": loss}



def main() -> None:
    parser = argparse.ArgumentParser(description="Flower")
    parser.add_argument(
        "--node-id",
        type=int,
        choices=range(0, 10),
        required=True,
        help="Specifies the artificial data partition",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        required=False,
        help="Specifies the NN batch partition",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        required=False,
        help="Specifies the NN window partition",
    )

    args = parser.parse_args()
    node_id = args.node_id
    #batch_size = args.batch_size
    #window_len = args.window_len

    logger_tb = TensorBoardLogger("tb_logs", name="test_4c")

    args = {'n_features':8, 'dataset_name': 'ausgrid', 'epochs': 1, 'batch_size': 32, 
            'lr': 0.001, 'hidden_size': 32, 'n_layers': (2, 2), 'use_bias': (True, True), 
            'dropout': (0, 0),
            'random_seed': 42, 'window_len': 30}
    
    device_obj = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Model and data mnist
    # model = mnist.LitAutoEncoder()
    # train_loader, val_loader, test_loader = mnist.load_data(node_id)

    # Load data
    train_loader, val_loader, test_loader = ausgrid.load_data(node_id, args['batch_size'], args['window_len'])

    model = ausgrid.LSTMAE_Lightning(n_features=7, hidden_size=args['hidden_size'],
                   n_layers=args['n_layers'], use_bias=args['use_bias'], 
                   dropout=args['dropout'])
    
    # hyperparameter epochs
    trainer = pl.Trainer(max_epochs=10, accelerator='cpu', default_root_dir='/home/labnet/Documents/JulianaPiaz/quickstart-pytorch-lightning/checkpoints/model/')

    # Train
    trainer.fit(model, train_loader)

    # Validation
    trainer.validate(model, val_loader)

    # Test
    trainer.test(model, test_loader)

    val_loader = None

    # Metrics
    model_save_path = '/home/labnet/Documents/JulianaPiaz/quickstart-pytorch-lightning/logs_metrics/models/' + args['dataset_name'] + '_model_epoch_' + '.pth'
    score_save_path = '/home/labnet/Documents/JulianaPiaz/quickstart-pytorch-lightning/logs_metrics/scores/' + args['dataset_name'] + '_score_epoch_' + '.npy'

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

    # Flower client
    client = FlowerClient(model, train_loader, val_loader, test_loader, logger_tb).to_client()
    fl.client.start_client(server_address="127.0.0.1:8080", client=client)


if __name__ == "__main__":
    main()
