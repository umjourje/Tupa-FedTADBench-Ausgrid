import flwr as fl
from flwr.common import Metrics
from typing import Dict, List, Tuple


def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    # Multiply accuracy of each client by number of examples used
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    # Aggregate and return custom metric (weighted average)
    return {"accuracy": sum(accuracies) / sum(examples)}


def main() -> None:
    # Define strategy
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=0.8,
        fraction_evaluate=0.5,
        #evaluate_metrics_aggregation_fn=weighted_average,  # <-- pass the metric aggregation function
    )

    # Start Flower server for three rounds of federated learning
    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=1),
        strategy=strategy,
    )


if __name__ == "__main__":
    main()
