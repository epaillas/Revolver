import numpy as np
from cosmopower import cosmopower_NN
import tensorflow as tf
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("--params", type=str, required=True)
parser.add_argument("--features", type=str, required=True)
parser.add_argument("--output", type=str, required=True)
parser.add_argument("--n_nodes", type=int, default=512)
parser.add_argument("--n_layers", type=int, default=4)
parser.add_argument("--standardize", action="store_true")
args = parser.parse_args()

# checking that we are using a GPU
device = 'gpu:0' if tf.config.list_physical_devices('GPU') else 'cpu'
print('Using', device, 'device \n')


params = np.load(args.params, allow_pickle=True).item()
train_params = {}
for name in params.keys():
    train_params[name]=list(np.array(params[name])[600:])
features = np.load(args.features)[600:]

if args.standardize:
    print("Standardizing features")
    features_mean = np.mean(features, axis=0)
    features_std = np.std(features, axis=0)
    features = (features - features_mean) / features_std
else:
    features_mean = np.zeros(features.shape[1])
    features_std = np.ones(features.shape[1])

# set up the neural network
cp_nn = cosmopower_NN(parameters=list(params.keys()), 
                    modes=np.linspace(-1,1,features.shape[1]), 
                    n_hidden=[args.n_nodes for i in range(args.n_layers)],
                    features_mean=features_mean,
                    features_std=features_std,
                    verbose=True,
                    )
with tf.device(device):
    # train
    cp_nn.train(training_parameters=train_params,
                training_features=features,
                filename_saved_model=args.output,
                # cooling schedule
                validation_split=0.1,
                learning_rates=[1e-2, 1e-3, 1e-4, 1e-5, 1e-6],
                batch_sizes=[100, 100, 100, 100, 100],
                gradient_accumulation_steps = [1, 1, 1, 1, 1],
                # early stopping set up
                patience_values = [100,100,100,100,100],
                max_epochs = [1000,1000,1000,1000,1000],
                )