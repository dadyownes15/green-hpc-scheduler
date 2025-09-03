from pathlib import Path
import configparser
import ast
from moviepy import ImageSequenceClip
import glob
import numpy as np
import gymnasium as gym 
import os

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
    configuration values with their correct data types.

    Args:
        config: A configparser.ConfigParser object that has already read
                the configuration file.

    Returns:
        A dictionary where keys are the configuration option names and
        values are the parsed configuration values.
    """
    config_dict = {}

    for section in config.sections():
        for key, value in config.items(section):
            try:
                # Attempt to convert to a specific type
                if value.lower() in ('true', 'false'):
                    parsed_value = config.getboolean(section, key)
                elif value.isdigit() or (value.startswith('-') and value[1:].isdigit()):
                    parsed_value = config.getint(section, key)
                elif '.' in value and all(part.isdigit() for part in value.split('.', 1)):
                    parsed_value = config.getfloat(section, key)
                elif value.startswith('[') and value.endswith(']'):
                    parsed_value = ast.literal_eval(value)
                else:
                    parsed_value = value
            except (ValueError, SyntaxError):
                # Fall back to string if conversion fails
                parsed_value = value
            
            # The key 'eta' appears in two different sections, so we need a
            # way to handle this, for example by including the section in the key
            # to prevent overwriting. You can adjust this as needed.
            # Example: 'reward_eta' instead of just 'eta'
            
            # For this example, we simply use the key and assume no duplicates.
            # If you want to differentiate, a good practice is to create a nested dictionary:
            # if section not in config_dict:
            #     config_dict[section] = {}
            # config_dict[section][key] = parsed_value

            config_dict[key] = parsed_value

    # Additional logic for derived values, like the length of a list.
    if 'delay_time_list' in config_dict and isinstance(config_dict['delay_time_list'], list):
        config_dict['delay_time_list_length'] = len(config_dict['delay_time_list'])

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


def create_experiment_name(config: dict, workload_file: None) -> str:
    """
    Creates an experiment name string based on a configuration dictionary and an optional workload file.

    Args:
        config (dict): A dictionary containing experiment configuration parameters.
        workload_file (str, optional): The path to the workload file. Defaults to None.

    Returns:
        str: A string representing the experiment name.
    """
    name_parts = []

    # Handle variable_carbon_intensities
    if config.get("variable_carbon_intensities"):
        name_parts.append("VI")
    else:
        name_parts.append("CI")

    # Handle batch_size
    batch_size = config.get("batch_size")
    if batch_size:
        name_parts.append(f"B{batch_size}")

    # Handle reward_type
    reward_type = config.get("reward_type")
    if "CO2_direct" == reward_type:
        name_parts.append("DC")
    if "CO2_direct_c" == reward_type:
        name_parts.append("DC-A") 
    elif "delay_vs_now_reward" == reward_type:
        name_parts.append("RC")
    else:
        # A more generic way to handle other reward types if needed
        name_parts.append("RC")

    # Handle learning_rate
    learning_rate = config.get("learning_rate")
    if learning_rate:
        # Format the learning rate string
        lr_str = f"LR-{learning_rate:.4f}"
        name_parts.append(lr_str.replace('.', '')) # Remove the dot

    # Handle eta
    eta = config.get("eta")
    if eta is not None:
        name_parts.append(f"ETA{int(eta)}")

    # Handle carbon_granularity
    carbon_granularity = config.get("carbon_granularity")
    if carbon_granularity == "minutely":
        name_parts.append("C-min")
    elif carbon_granularity == "hourly":
        name_parts.append("C-hour")
    else:
        name_parts.append(f"C-{carbon_granularity}")

    custom_itensity = config.get("custom_intensity")
    if custom_itensity == True:
        name_parts.append("EASY")
    # Handle Trace based on workload_file
    if workload_file:
        file_name = os.path.basename(workload_file).lower()
        if "synthetic" in file_name:
            name_parts.append("Lu-s")
        else:
            name_parts.append("Lu")
    else:
        name_parts.append("Lu")  # Default to "Lu" if no workload file is provided

    # Join all parts with underscores to create the final name
    return "_".join(name_parts)
