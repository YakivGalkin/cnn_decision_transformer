import numpy as np
import torch
import torch.nn as nn
import gymnasium as gym
from stable_baselines3.common.torch_layers import NatureCNN
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage

from utils.storage import get_downloaded_model_file



def get_cnn_feature_extractor_model(features_dim=512, pretrained_model_name=None):
    env =  gym.make('CarRacing-v2', continuous=False) 
    wrapped_env = VecTransposeImage(DummyVecEnv([lambda: env]))
    obs_space = wrapped_env.observation_space
    #obs_space = Box(0, 255, (3, 96, 96), uint8)
    model = NatureCNN(obs_space, features_dim=features_dim)
    if pretrained_model_name is not None:
        model_path = get_downloaded_model_file(pretrained_model_name)
        model.load_state_dict(torch.load(model_path))
    return model



def reshape_observation( observation): # move channel to the end, add batch dimension if needed
    observation_space_shape = (3,96,96)
    if not (observation.shape == observation_space_shape or observation.shape[1:] == observation_space_shape):
        # Try to re-order the channels
        transpose_obs = VecTransposeImage.transpose_image(observation)
        if transpose_obs.shape == observation_space_shape or transpose_obs.shape[1:] == observation_space_shape:
            observation = transpose_obs
    # Add batch dimension if needed
    observation = observation.reshape((-1, *observation_space_shape))

def prepare_env_observation_tensor( observation):
    observation = reshape_observation( observation)
    observation = torch.as_tensor(observation) 
    observation = observation.float() / 255.0
    return observation


def prepare_env_observation_array( observation):
    observation = reshape_observation( observation)
    observation = observation.astype(np.float32) / 255.0
    return observation

