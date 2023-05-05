# Human Activity Recognition with Flower

This project demonstrates how to use the Flower library for distributed deep learning to train a Human Activity Recognition (HAR) model. The model is trained on the [UCI HAR Dataset](https://archive.ics.uci.edu/ml/datasets/human+activity+recognition+using+smartphones), which contains data from smartphone sensors to identify six different activities: walking, walking upstairs, walking downstairs, sitting, standing, and laying.

## Requirements

To run this project, you need the following dependencies:

- Python 3.7 or higher
- TensorFlow 2.x
- Flower

Install the required packages using pip:

`pip install tensorflow flwr`


## Dataset

Download the UCI HAR dataset from [here](https://archive.ics.uci.edu/ml/datasets/human+activity+recognition+using+smartphones) and extract it into a folder named 'UCI_HAR_Dataset' in the project directory.

## Running the project

First, start the Flower server by running the `server.py` file:

python server.py

Then, start one or more Flower clients by running the `client.py` file:

python client.py

You can run multiple instances of the `client.py` file to simulate a distributed deep learning setup.

## Project structure

This project contains two main files:

- `server.py`: Starts the Flower server, which coordinates the distributed training process among multiple clients.
- `client.py`: Defines the HAR model, loads the dataset, and starts the Flower client, which connects to the server and participates in the distributed training process.
