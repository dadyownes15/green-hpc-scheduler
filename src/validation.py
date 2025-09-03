from src.baseline import Baseline, MedianBaseline
from sb3_contrib import MaskablePPO
from stable_baselines3.common.evaluation import evaluate_policy
from src.hpc_env import HPCenv
from src.utils import VideoGenerator, get_config_as_dict, mask_fn
from sb3_contrib.common.wrappers import ActionMasker
from sb3_contrib.ppo_mask import MaskablePPO
from stable_baselines3.common.monitor import Monitor
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from sb3_contrib.common.maskable.utils import get_action_masks
import configparser
import json
import os
from typing import List, Type
class Validation():

    """
    Validation suite takes a trained model, for now we will simply hardcode the baseline.py and evaluates the model and produces rendering, and overview statistics for n different episodes.
    """

    def __init__(self, model_dir,  workload_path, baselines = []) -> None:
        self.model_dir = model_dir 
        config = configparser.ConfigParser()
        config_path = os.path.join(os.getcwd(), self.model_dir, 'config.json')

        try:
            with open(config_path, 'r') as f:
                self.config_dict = json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Configuration file not found at '{config_path}'. Please ensure it exists.")
        except json.JSONDecodeError:
            raise ValueError(f"Configuration file at '{config_path}' is not a valid JSON file. Please check its syntax.")

        self.workload_path = workload_path
        self.baselines_types = baselines
        # self.env = ActionMasker(HPCenv(workload_path=workload_path,config_dict=self.config_dict), action_mask_fn= mask_fn)

        # self.model = MaskablePPO("MlpPolicy", self.env)

    def compare(self,n_eval_episodes : int, checkpoints : List[str], generate_plots = False, generate_renderings = False, seed_for_rendering = None, reward_type = "CO2_direct" ):
        """
        Evalutes the model on a job trace with episode lengths, on different seeds (thus different episodes).

        """
        self.config_dict["reward_type"] = reward_type

        self.baselines = [baseline(self.config_dict,HPCenv(workload_path=self.workload_path,config_dict=self.config_dict)) for baseline in self.baselines_types]

        models_dict = {} 
        rewards_dict = {
            "model": [],
        }
        self.env =   ActionMasker(HPCenv(workload_path=self.workload_path,config_dict=self.config_dict), action_mask_fn= mask_fn)
        for checkpoint in checkpoints:
            base_model =   MaskablePPO("MlpPolicy", self.env)
            base_model.load(self.model_dir + "/logs/"+checkpoint)
            models_dict[checkpoint] = base_model
            rewards_dict[checkpoint] = [] 

        

        for baseline in self.baselines:
            rewards_dict[baseline.name] = []


        # Simulate for n episodes across model and baselines
        for i in range(n_eval_episodes):
            for model_name, model in models_dict.items():
                model_reward = self.evaluate_policy(seed=i, model=model)
                rewards_dict[model_name].append(model_reward)

            for baseline in self.baselines:
                baseline_reward = baseline.run(seed=i)
                rewards_dict[baseline.name].append(baseline_reward) 

        return rewards_dict
    
    def deep_dive(self, seed, model):
       pass 

    def evaluate_policy(self,seed, model):
        obs, _ = self.env.reset(seed=seed, options={})
        
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

            action, _states = model.predict(obs, action_masks=action_masks)
            obs, reward, terminated, truncated, info = self.env.step(action)
            total_reward += float(reward)
        
        return total_reward


    def render_input_model(self, model_path: str, seed: int, step_interval=1, name="model_rendering"):
        """
        Renders an episode of a trained model by generating plots for each step and compiling them into a video.
        """
        # 1. Instantiate the environment with rendering enabled
        self.config_dict['generate_rendering'] = True
        self.config_dict['name'] = name
        render_env = ActionMasker(HPCenv(workload_path=self.workload_path, config_dict=self.config_dict), action_mask_fn=mask_fn)

        # 2. Load the trained model
        model = MaskablePPO.load(model_path, env=render_env)

        # 3. Run the episode and render each step
        obs, _ = render_env.reset(seed=seed, options={})
        terminated = False
        step_count = 0

        while not terminated:
            if step_count % step_interval == 0:
                render_env.render(step_count=step_count)

            action_masks = get_action_masks(render_env)
            action, _states = model.predict(obs, action_masks=action_masks, deterministic=True)
            obs, reward, terminated, truncated, info = render_env.step(action)
            step_count += 1

        # 4. Generate the video from the saved images
        video_gen = VideoGenerator(path=render_env.dir_path)
        video_gen.generate_video()
        print(f"Rendering complete. Video saved at {render_env.dir_path}/rendering.mp4")