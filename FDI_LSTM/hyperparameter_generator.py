import argparse
import json

# Create the parser
parser = argparse.ArgumentParser(description='Process some integers.')

# Add the arguments
parser.add_argument('--BATCH_SIZE', type=int,default=16)
parser.add_argument('--FILL_NAN', type=int,default=10)
parser.add_argument('--TIME_STEP', type=int, default=10)
parser.add_argument('--COLUMN', type=str, default='flow')
parser.add_argument('--EPOCHS', type=int, default=100)
parser.add_argument('--lr', type=float,default=0.0001)
parser.add_argument('--HIDDEN_LAYER_SIZE', type=int, default=100)
parser.add_argument('--NUM_LAYERS', type=int,default=5)
parser.add_argument('--TRAIN_TEST_SPLIT', type=float,default=0.875)
parser.add_argument('--ROOT_FOLDER', type=str,default='.')

# Parse the arguments
hyperparameters = parser.parse_args()

# Save the arguments to a dictionary

hyperparameters_dict = vars(hyperparameters)


json_object = json.dumps(hyperparameters_dict,indent=4) ### this saves the array in .json format)


with open('hyperparameters.json', 'w') as outfile:
    outfile.write(json_object)