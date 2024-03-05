import argparse
from collections import OrderedDict

import pytorch_lightning as pl
import torch
from datasets.utils.logging import disable_progress_bar

import flwr as fl
import mnist
import ausgrid

disable_progress_bar()


class FlowerClient(fl.client.NumPyClient):
    def __init__(self, model, train_loader, val_loader, test_loader):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader

    def get_parameters(self, config):
        encoder_params = _get_parameters(self.model.encoder)
        decoder_params = _get_parameters(self.model.decoder)
        return encoder_params + decoder_params

    def set_parameters(self, parameters):
        _set_parameters(self.model.encoder, parameters[:4])
        _set_parameters(self.model.decoder, parameters[4:])

    def fit(self, parameters, config):
        self.set_parameters(parameters)

        trainer = pl.Trainer(max_epochs=1)
        trainer.fit(self.model, self.train_loader,)

        return self.get_parameters(config={}), 55000, {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)

        trainer = pl.Trainer()
        results = trainer.test(self.model, self.test_loader)
        loss = results[0]["test_loss"]

        return loss, 10000, {"loss": loss}


def _get_parameters(model):
    return [val.cpu().numpy() for _, val in model.state_dict().items()]


def _set_parameters(model, parameters):
    params_dict = zip(model.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    model.load_state_dict(state_dict, strict=True)


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


    args = {'n_features':8, 'dataset_name': 'ausgrid', 'epochs': 1, 'batch_size': 32, 
            'lr': 0.001, 'hidden_size': 32, 'n_layers': (2, 2), 'use_bias': (True, True), 
            'dropout': (0, 0),
            'random_seed': 42, 'window_len': 30}
    
    # Model and data mnist
    # model = mnist.LitAutoEncoder()
    # train_loader, val_loader, test_loader = mnist.load_data(node_id)

    # Load data
    train_loader, val_loader, test_loader = ausgrid.load_data(node_id, args['batch_size'], args['window_len'])

    model = ausgrid.LSTMAE_Lightning(n_features=7, hidden_size=args['hidden_size'],
                   n_layers=args['n_layers'], use_bias=args['use_bias'], 
                   dropout=args['dropout'])
    
    trainer = pl.Trainer(max_epochs=1, accelerator='cpu')

    # Train
    trainer.fit(model, train_loader)

    # Test
    trainer.test(model, test_loader)

    val_loader = None

    # Flower client
    client = FlowerClient(model, train_loader, val_loader, test_loader).to_client()
    fl.client.start_client(server_address="127.0.0.1:8080", client=client)


if __name__ == "__main__":
    main()
