import flwr as fl

if __name__ == "__main__":
    fl.server.start_server("0.0.0.0:8080")