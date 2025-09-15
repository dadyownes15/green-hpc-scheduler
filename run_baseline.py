from src.validation import Validation
from src.utils import convert_numpy_types
import json 

import argparse

parser = argparse.ArgumentParser("Run baseline validatioon use median baseline")
parser.add_argument("model_dir", type=str)
parser.add_argument("n_vals", type=int, help="Amount of different seeds the test data or validation data, should be ran on")
parser.add_argument("debug", type=bool, help="Executes prints to be able to debug")
parser.add_argument("mode", type=str, help="Use validation or test data")
args = parser.parse_args()


val = Validation()
val.load_dir(args.model_dir)

## For now we will just use a models config, to ensure consistency
stats = val.run_baselines(n_eval_episodes=args.n_vals,debug=args.debug, mode="test")
log_path = "results/median_baseline_results.txt"

with open(log_path, 'w') as f: 
    f.write(f"n_eval: {args.n_vals} \n")
    f.write(f"mode: {args.mode} \n\n\n")
    print(stats)
    cleaned_stats = convert_numpy_types(stats)
    json.dump(cleaned_stats,f,indent=4);