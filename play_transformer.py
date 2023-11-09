import os
import random
from dataclasses import dataclass

import numpy as np
import torch
from datasets import load_dataset
from transformers import Trainer, TrainingArguments
from visual_decision_transformer.configuration import DecisionTransformerConfig
from visual_decision_transformer.visual_decision_transformer_model import VisualDecisionTransformerModel


import wandb
import random

import torch
import torch.nn as nn
from stable_baselines3 import DQN
import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.torch_layers import NatureCNN
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage

import os
import random
from dataclasses import dataclass

import numpy as np
import torch
from datasets import load_dataset
from datasets import load_from_disk
from datasets import Dataset
from transformers import Trainer, TrainingArguments
from visual_decision_transformer.configuration import DecisionTransformerConfig
from visual_decision_transformer.visual_decision_transformer_model import VisualDecisionTransformerModel

from visual_decision_transformer.visual_decision_transformer_trainable import VisualDecisionTransformerGymDataCollator,  TrainableVisualDecisionTransformer
import toml



config = DecisionTransformerConfig(state_dim=96*96*3, act_dim=1)

import utils.storage as storage
#loaded_model = storage.get_pretrained_model(TrainableVisualDecisionTransformer,  'visual_transformer_01.pt')

dt_config = DecisionTransformerConfig(state_dim=96*96*3, act_dim=1,
                                      max_length = 10,
                                      max_ep_len = 1000,  
                                      )

model = TrainableVisualDecisionTransformer(dt_config)
model.load_state_dict(torch.load('/Users/jacob/Documents/T/hdt/models/transformer_pretrained/visual_transformer_01.pt'))

exit(0)

model.load_state_dict(torch.load('./models/transformer_pretrained/pytorch_model.bin'))

print(model)

env = gym.make('CarRacing-v2', continuous=False, render_mode='human')



# Function that gets an action from the model using autoregressive prediction with a window of the previous 20 timesteps.
def get_action(model, states, actions, rewards, returns_to_go, timesteps):
    # This implementation does not condition on past rewards

    states = states.reshape(1, -1, model.config.state_dim)
    actions = actions.reshape(1, -1, model.config.act_dim)
    returns_to_go = returns_to_go.reshape(1, -1, 1)
    timesteps = timesteps.reshape(1, -1)

    states = states[:, -model.config.max_length :]
    actions = actions[:, -model.config.max_length :]
    returns_to_go = returns_to_go[:, -model.config.max_length :]
    timesteps = timesteps[:, -model.config.max_length :]
    padding = model.config.max_length - states.shape[1]
    # pad all tokens to sequence length
    attention_mask = torch.cat([torch.zeros(padding), torch.ones(states.shape[1])])
    attention_mask = attention_mask.to(dtype=torch.long).reshape(1, -1)
    states = torch.cat([torch.zeros((1, padding, model.config.state_dim)), states], dim=1).float()
    actions = torch.cat([torch.zeros((1, padding, model.config.act_dim)), actions], dim=1).long()
    returns_to_go = torch.cat([torch.zeros((1, padding, 1)), returns_to_go], dim=1).float()
    timesteps = torch.cat([torch.zeros((1, padding), dtype=torch.long), timesteps], dim=1)

    state_preds, action_preds, return_preds = model.original_forward(
        states=states,
        actions=actions,
        rewards=rewards,
        returns_to_go=returns_to_go,
        timesteps=timesteps,
        attention_mask=attention_mask,
        return_dict=False,
    )

    return action_preds[0, -1]

#import gymnasium as gym
# build the environment
max_ep_len = 1000
device = "cpu"
scale = 1000.0  # normalization for rewards/returns
TARGET_RETURN = 1000 / scale  # evaluation is conditioned on a return of 12000, scaled accordingly


state_dim = 96*96*3
act_dim = 1
# Create the decision transformer model

# Interact with the environment and create a video
episode_return, episode_length = 0, 0
[state, _] = env.reset()
state = prepare_observation_array(state)
target_return = torch.tensor(TARGET_RETURN, device=device, dtype=torch.float32).reshape(1, 1)
states = torch.from_numpy(state).reshape(1, state_dim).to(device=device, dtype=torch.float32)
actions = torch.zeros((0, act_dim),  device=device, dtype=torch.long)
rewards = torch.zeros(0, device=device, dtype=torch.float32)

timesteps = torch.tensor(0, device=device, dtype=torch.long).reshape(1, 1)
for t in range(max_ep_len):
    actions = torch.cat([actions, torch.zeros((1, act_dim), dtype=torch.long,  device=device)], dim=0)
    rewards = torch.cat([rewards, torch.zeros(1, device=device)])

    action = get_action(
        model,
        states,
        actions,
        rewards,
        target_return,
        timesteps,
    )
    
    action =   torch.argmax(action).item() # action.detach().cpu().numpy()
    print(f"{action} ", end="")
    actions[-1] = torch.tensor(action, dtype=torch.long) 

    state, reward, done, _, _ = env.step(action)

    state = prepare_observation_array(state)
    cur_state = torch.from_numpy(state).to(device=device).reshape(1, state_dim)
    states = torch.cat([states, cur_state], dim=0)
    rewards[-1] = reward

    pred_return = target_return[0, -1] - (reward / scale)
    target_return = torch.cat([target_return, pred_return.reshape(1, 1)], dim=1)
    timesteps = torch.cat([timesteps, torch.ones((1, 1), device=device, dtype=torch.long) * (t + 1)], dim=1)

    episode_return += reward
    episode_length += 1

    if done:
        break

#env.play()