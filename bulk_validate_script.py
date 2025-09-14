import argparse
import json

from src.validation import Validation

parser = argparse.ArgumentParser("Run a bulk validation on a set of models")

# Model list
parser.add_argument("models", type=list, help="Include the name of the model, it should match the model dir within results")
parser.add_argument("checkpoints", type=list, help="Include of a list of checkpoints that should be validated")
parser.add_argument("n_vals", type=int, help="Amount of different seeds the test data or validation data, should be ran on")
parser.add_argument("debug", type=bool, help="Executes prints to be able to debug")
parser.add_argument("mode", type=str, help="Use validation or test data")
args = parser.parse_args()



for model in args.models:
    model_dir =  "results/" + model
    val = Validation(model_dir=model_dir)
    stats = val.validate_policy(
        n_eval_episodes=args.n_vals,
        checkpoints=args.checkpoints,
        mode=args.mode,
        debug=args.debug
        )
    
    log_path = model_dir + "/validation_results.txt"
    with open(log_path, 'w') as f: 

        f.write(f"Validation results for model: {model}\n\n")
        f.write(f"n_eval: {args.n_vals} \n")
        f.write(f"checkpoints: {args.checkpoints} \n")
        f.write(f"mode: {args.mode} \n\n\n")
        json.dump(stats,f,indent=4)
""" 
import argparse

# 1. Create a parser object
parser = argparse.ArgumentParser(description='A simple script to greet a user.')

# 2. Add an argument
parser.add_argument('name', type=str, help='The name of the user to greet.')

# 3. Parse the arguments from the command line
args = parser.parse_args()

# 4. Use the parsed arguments
print(f"Hello, {args.name}!") """