import argparse
import json

# Create the parser
parser = argparse.ArgumentParser(description='Process some integers.')

# Add the arguments
parser.add_argument('--BATCH_SIZE', type=int,default=64)
parser.add_argument('--FILL_NAN', type=int,default=10)
parser.add_argument('--TIME_STEP', type=int, default=10)
parser.add_argument('--COLUMN', type=str, default='cpu_consumption')
parser.add_argument('--EPOCHS', type=int, default=2000)
parser.add_argument('--lr', type=float,default=0.001)
parser.add_argument('--HIDDEN_LAYER_SIZE', type=int, default=10)
parser.add_argument('--NUM_LAYERS', type=int,default=3)
parser.add_argument('--TRAIN_TEST_SPLIT', type=float,default=0.9)
parser.add_argument('--ROOT_FOLDER', type=str,default='.')

# Parse the arguments
hyperparameters = parser.parse_args()

# Save the arguments to a dictionary

hyperparameters_dict = vars(hyperparameters)


json_object = json.dumps(hyperparameters_dict,indent=4) ### this saves the array in .json format)


with open('hyperparameters.json', 'w') as outfile:
    outfile.write(json_object)