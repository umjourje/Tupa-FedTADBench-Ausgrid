import flwr as fl
from flwr.common import Metrics, FitRes, Parameters, Scalar
from typing import Dict, List, Tuple, Union, Optional, OrderedDict
from flwr.server.client_proxy import ClientProxy
import torch
import numpy as np

def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    # Multiply accuracy of each client by number of examples used
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    # Aggregate and return custom metric (weighted average)
    return {"accuracy": sum(accuracies) / sum(examples)}


class SaveModelStrategy(fl.server.strategy.FedAvg):
    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:

        # Call aggregate_fit from base class (FedAvg) to aggregate parameters and metrics
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(server_round, results, failures)

        if aggregated_parameters is not None:
            # Convert `Parameters` to `List[np.ndarray]`
            aggregated_ndarrays: List[np.ndarray] = fl.common.parameters_to_ndarrays(aggregated_parameters)

            # Save aggregated_ndarrays
            run_id = 1
            print(f"Saving round {server_round} aggregated_ndarrays...")
            np.savez(f"/home/labnet/Documents/JulianaPiaz/quickstart-pytorch-lightning/checkpoints/server_model_10ep/run_{run_id}-round_{server_round}-weights.npz", *aggregated_ndarrays)

        return aggregated_parameters, aggregated_metrics
    

def main() -> None:
    # Define strategy
    strategy = SaveModelStrategy(
        fraction_fit=0.8,
        fraction_evaluate=0.5,
        #evaluate_metrics_aggregation_fn=weighted_average,  # <-- pass the metric aggregation function
    )

    # Start Flower server for three rounds of federated learning
    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=5),
        strategy=strategy,
    )


if __name__ == "__main__":
    main()
