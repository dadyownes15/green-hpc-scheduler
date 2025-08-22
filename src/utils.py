from pathlib import Path
import configparser
import ast
from moviepy import ImageSequenceClip
import glob
import numpy as np
import gymnasium as gym 

class VideoGenerator():
    def __init__(self, path):
        self.path = path


    def generate_video(self):
        # Get a list of all PNG files and sort them
        path_name = self.path + "/*.png"
        image_files = sorted(glob.glob(path_name))

        assert len(image_files) > 0, "Could not find imagefiles"
        # Set the frames per second
        fps = 2

        # Create a clip from the image sequence
        clip = ImageSequenceClip(image_files, fps=fps)

        # Write the video file to disk
        clip.write_videofile(path_name + "rendering.mp4", codec='libx264', fps=fps)
        print("Video saved as ", path_name + "rendering.my")


def get_config_as_dict(config: configparser.ConfigParser) -> dict:
    """
    Reads a configparser object and returns a dictionary containing all
    specified configuration values with their correct data types.

    Args:
        config: A configparser.ConfigParser object that has already read
                the configuration file.

    Returns:
        A dictionary where keys are the configuration option names and
        values are the parsed configuration values.
    """
    config_dict = {}

    # Helper function to get a value with a default
    def get_value(section, key, dtype, default=None):
        try:
            if dtype == 'int':
                return config.getint(section, key)
            elif dtype == 'float':
                return config.getfloat(section, key)
            elif dtype == 'bool':
                return config.getboolean(section, key)
            elif dtype == 'list':
                return ast.literal_eval(config.get(section, key))
            else:
                return config.get(section, key)
        except (configparser.NoSectionError, configparser.NoOptionError):
            return default

    # Power settings
    config_dict['use_constant_power'] = get_value('power settings', 'use_constant_power', 'bool')
    config_dict['constant_power_per_processor'] = get_value('power settings', 'constant_power_per_processor', 'int')
    config_dict['procs_per_node'] = get_value('power settings', 'procs_per_node', 'int')
    config_dict['idle_power'] = get_value('power settings', 'idle_power', 'float')
    config_dict['carbon_year'] = get_value('power settings', 'carbon_year', 'int')
    config_dict['variable_carbon_intensities'] = get_value('power settings', 'variable_carbon_intensities', 'bool')

    # Architecture
    config_dict['green_forecast_length'] = get_value('architecture', 'green_forecast_length', 'int')
    config_dict['eta'] = get_value('architecture', 'eta', 'float')
    config_dict['max_queue_size'] = get_value('architecture', 'max_queue_size', 'int')
    config_dict['run_win_length'] = get_value('architecture', 'run_win_length', 'int')
    config_dict['delay_time_list'] = get_value('architecture', 'delay_time_list', 'list')
    # delay_time_list_length can be calculated from the list itself
    config_dict['delay_time_list_length'] = len(get_value('architecture', 'delay_time_list', 'list'))
    config_dict['max_wait_n_jobs'] = get_value('architecture', 'max_wait_n_jobs', 'int')
    config_dict['job_feature'] = get_value('architecture', 'job_feature', 'int')
    config_dict['run_feature'] = get_value('architecture', 'run_feature', 'int')
    config_dict['green_feature_pr_timeslot'] = get_value('architecture', 'green_feature_pr_timeslot', 'int')
    config_dict['green_feature_constant'] = get_value('architecture', 'green_feature_constant', 'int')

    # Training
    config_dict['episode_length'] = get_value('training', 'episode_length', 'int')
    config_dict['gamma'] = get_value('training', 'gamma', 'float')
    config_dict['gae_lambda'] = get_value('training', 'gae_lambda', 'float')

    # Reward
    config_dict['base_line_wait_carbon_penality'] = get_value('reward', 'base_line_wait_carbon_penality', 'float')
    config_dict['bounded_slowdown_threshhold'] = get_value('reward', 'bounded_slowdown_threshhold', 'int')
    config_dict['eta'] = get_value('reward', 'eta', 'float')
    config_dict['reward_type'] = get_value('reward', 'reward_type', 'str')

    # Normalization constants
    config_dict['max_power'] = get_value('normalization constants', 'max_power', 'int')
    config_dict['max_green'] = get_value('normalization constants', 'max_green', 'int')
    config_dict['max_wait_time'] = get_value('normalization constants', 'max_wait_time', 'int')
    config_dict['max_run_time'] = get_value('normalization constants', 'max_run_time', 'int')
    config_dict['max_requested_processors'] = get_value('normalization constants', 'max_requested_processors', 'int')

    return config_dict

def create_directory_if_not_exists(directory_path: str):
    """
    Creates a directory if it does not already exist.

    The function will raise an AssertionError if the directory already exists.

    Args:
        directory_path (str): The path to the directory to create.
    """
    # Create a Path object for the directory
    p = Path(directory_path)

    # Use an assert to fail if the directory already exists.
    # The 'not' operator inverts the check: the assertion passes if the path does NOT exist.
    assert not p.exists(), f"Assertion failed: Directory already exists at '{directory_path}'"

    # Create the directory.
    # The 'parents=True' argument ensures that any missing parent directories are also created.
    try:
        p.mkdir(parents=True)
        print(f"Directory created successfully at '{directory_path}'.")
    except OSError as e:
        # This catch block is a failsafe in case of permission issues or other
        # unexpected errors during creation.
        print(f"Error creating directory at '{directory_path}': {e}")

def mask_fn(env: gym.Env) -> np.ndarray:
    # Do whatever you'd like in this function to return the action mask
    # for the current env. In this example, we assume the env has a
    # helpful method we can rely on.
    return env.valid_action_mask()

