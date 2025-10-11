import configparser
import os
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from sb3_contrib.common.wrappers import ActionMasker
from sb3_contrib.ppo_mask import MaskablePPO
from src.hpc_env import HPCenv
from src.utils import get_config_as_dict, mask_fn


def main() -> None:
    config = configparser.ConfigParser()
    config_path = os.path.join(os.getcwd(), "config_file", "config.ini")
    config.read(config_path)
    config_dict = get_config_as_dict(config)

    n_envs = 16
    rollout_steps = max(1, config_dict.get("n_steps", 2048) // n_envs)
    env = make_vec_env(
        HPCenv,
        n_envs=n_envs,
        env_kwargs=dict(config_dict=config_dict, mode="training"),
        wrapper_class=ActionMasker,
        wrapper_kwargs=dict(action_mask_fn=mask_fn),
        vec_env_cls=SubprocVecEnv,
        seed=config_dict.get("seed", 42),
    )

    model = MaskablePPO(
        "MlpPolicy",
        env,
        n_steps=rollout_steps,
        batch_size=config_dict.get("batch_size", 64),
        n_epochs=config_dict.get("n_epochs", 10),
        gamma=config_dict.get("gamma", 0.99),
        gae_lambda=config_dict.get("gae_lambda", 0.95),
        ent_coef=config_dict.get("ent_coef", 0.0),
        vf_coef=config_dict.get("vf_coef", 0.5),
        clip_range=config_dict.get("clip_range", 0.2),
        clip_range_vf=config_dict.get("clip_range_vf", 1.0),
        learning_rate=config_dict.get("learning_rate", 3e-4),

        verbose=0,
        seed=config_dict.get("seed", 42),
    )
    model.learn(total_timesteps=config_dict.get("total_timesteps", 1_000_000))


if __name__ == "__main__":
    main()
