from src.baseline import Baseline 
from sb3_contrib import MaskablePPO
from stable_baselines3.common.evaluation import evaluate_policy
from src.hpc_env import HPCenv
from src.utils import get_config_as_dict, mask_fn
from sb3_contrib.common.wrappers import ActionMasker
from sb3_contrib.ppo_mask import MaskablePPO
from stable_baselines3.common.monitor import Monitor
import configparser

class Validation():

    """
    Validation suite takes a trained model, for now we will simply hardcode the baseline.py and evaluates the model and produces rendering, and overview statistics for n different episodes.
    """

    def __init__(self, model_path, config : configparser.ConfigParser, workload_path) -> None:
        self.model_path = model_path 
        self.config_dict = get_config_as_dict(config=config) 
        self.config = config
        self.workload_path = workload_path
        self.val_env = Monitor(ActionMasker(HPCenv(workload_path=workload_path,config=config), mask_fn)) 



        self.model = MaskablePPO("MlpPolicy", self.val_env, verbose=1,
                        seed=42,
                        )
    def compare(self,n_eval_episodes : int, baseline : Baseline, generate_plots = False, generate_renderings = False, seed_for_rendering = None, ):
        """
        Evalutes the model on a job trace with episode lengths, on different seeds (thus different episodes).

        1, We should calculate the cummlative award for the model and all baselines 
        """
        # Evaluate model
        mean_reward = evaluate_policy(model=self.model,env=self.val_env,n_eval_episodes=n_eval_episodes)

        print(mean_reward)

        
        
        