from src.baseline import Baseline 
from sb3_contrib import MaskablePPO
from stable_baselines3.common.evaluation import evaluate_policy
from src.hpc_env import HPCenv
from src.utils import get_config_as_dict, mask_fn
from sb3_contrib.common.wrappers import ActionMasker
from sb3_contrib.ppo_mask import MaskablePPO
from stable_baselines3.common.monitor import Monitor
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from sb3_contrib.common.maskable.utils import get_action_masks
import configparser
from typing import List, Type
class Validation():

    """
    Validation suite takes a trained model, for now we will simply hardcode the baseline.py and evaluates the model and produces rendering, and overview statistics for n different episodes.
    """

    def __init__(self, model_path, config : configparser.ConfigParser, workload_path, baselines = []) -> None:
        self.model_path = model_path 
        self.config_dict = get_config_as_dict(config=config) 
        self.config = config
        self.workload_path = workload_path
        self.env = ActionMasker(
            HPCenv(workload_path=workload_path,config=config), action_mask_fn=
            mask_fn)
        
        self.baselines = [baseline(self.config,HPCenv(workload_path=workload_path,config=config)) for baseline in baselines]

        self.model = MaskablePPO("MlpPolicy", self.env, verbose=2,
                        seed=42,
                        )
        
        self.model.load(self.model_path)
    def compare(self,n_eval_episodes : int, generate_plots = False, generate_renderings = False, seed_for_rendering = None, ):
        """
        Evalutes the model on a job trace with episode lengths, on different seeds (thus different episodes).

        1, We should calculate the cummlative award for the model and all baselines 
        """
        rewards_dict = {
            "model": [],
        }
        for baseline in self.baselines:
            rewards_dict[baseline.name] = []


        # Simulate for n episodes across model and baselines
        for i in range(n_eval_episodes):
            model_reward = self.evaluate_policy(seed=i)
            rewards_dict['model'].append(model_reward)

            for baseline in self.baselines:
                baseline_reward = baseline.run(seed=i)
                rewards_dict[baseline.name].append(baseline_reward) 

        return rewards_dict

    def evaluate_policy(self,seed):
        obs, _ = self.env.reset(seed=seed)
        terminated = False
        total_reward = 0
        step_count = 0  # Add a counter
        while not terminated:
            # Retrieve current action mask
            action_masks = get_action_masks(self.env)
            # --- DEBUGGING STEP ---
            # Print the mask and the number of valid actions
            num_valid_actions = sum(action_masks)
            if num_valid_actions <= 1 and step_count < 10: # Print for the first 10 steps
                print(f"Step {step_count}: Valid Actions = {num_valid_actions}, Mask = {action_masks}")
            # --------------------

            action, _states = self.model.predict(obs, action_masks=action_masks)
            obs, reward, terminated, truncated, info = self.env.step(action)
            total_reward += float(reward)
        
        return total_reward