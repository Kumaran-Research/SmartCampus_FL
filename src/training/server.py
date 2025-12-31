import flwr as fl
import torch
import pickle
from model import GhostFaceNetV2
import os

def get_initial_parameters(model):
    """Initialize model parameters."""
    return [val.cpu().numpy() for val in model.state_dict().values()]

def main():
    with open('configs/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    # Initialize model
    data_dir = config['data']['client1_dir']
    num_classes = len(os.listdir(data_dir))
    model = GhostFaceNetV2(num_classes=num_classes)
    initial_parameters = get_initial_parameters(model)

    # Define FedProx strategy
    strategy = fl.server.strategy.FedProx(
        fraction_fit=config['server']['fraction_fit'],
        fraction_evaluate=config['server']['fraction_evaluate'],
        min_fit_clients=config['server']['min_fit_clients'],
        min_evaluate_clients=config['server']['min_evaluate_clients'],
        min_available_clients=config['server']['min_available_clients'],
        initial_parameters=fl.common.ndarrays_to_parameters(initial_parameters),
        proximal_mu=config['server']['proximal_mu'],
    )

    # Start server
    history = fl.server.start_server(
        server_address=config['server']['address'],
        config=fl.server.ServerConfig(num_rounds=config['server']['num_rounds']),
        strategy=strategy,
    )

    # Save final parameters
    final_params = history.parameters
    with open(config['experiment']['checkpoints_dir'] + "/final_model.pth", "wb") as f:
        pickle.dump(final_params, f)

if __name__ == "__main__":
    main()