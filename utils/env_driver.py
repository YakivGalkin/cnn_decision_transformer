import torch
import numpy as np

from visual_decision_transformer.cnn_feature_extractor import prepare_env_observation_array, prepare_env_observation_tensor

class EnvDriverBase:
    def drive(self, state, previous_reward=0):
        pass

class SimplePolicyEnvDriver(EnvDriverBase):
    def __init__(self, model):
        self.model = model

    def drive(self, state, previous_reward=0):
        with torch.no_grad():
            action, _ = self.model.predict(state,deterministic=True)
        return action
    
class TransformerEnvDriver(EnvDriverBase):
    def __init__(self, transformer, target_return_value):
        self.model = transformer
        self.scale = 1000.0  # normalization for rewards/returns
        self.timestep = 0
        self.max_length = self.model.config.max_length
        self.act_dim = self.model.config.act_dim
        self.state_dim = self.model.config.state_dim
        self.max_ep_len = self.model.config.max_ep_len

        self.target_return_value = target_return_value# = torch.tensor(target_return, dtype=torch.float32).reshape(1, 1)

        self.states = torch.zeros((0, self.model.config.state_dim), dtype=torch.float32)
        self.actions = torch.zeros((0, self.model.config.act_dim),  dtype=torch.long)



    def drive(self, state, previous_reward=0):
        if self.timestep >= self.max_ep_len-1:
            print("Max episode length reached")
            return 0
        #action = self.transformer(state, reward)
        if self.states.nelement() == 0:
            self.target_return = torch.tensor(self.target_return_value/self.scale, dtype=torch.float32).reshape(1, 1)
            self.timesteps = torch.tensor(0, dtype=torch.long).reshape(1, 1)
        else:
            pred_return = self.target_return[0, -1] - (previous_reward / self.scale)
            self.target_return = torch.cat([self.target_return, pred_return.reshape(1, 1)], dim=1)
            self.timesteps = torch.cat([self.timesteps, torch.ones((1, 1),  dtype=torch.long) * (self.timestep + 1)], dim=1)
        
        state = prepare_env_observation_tensor(state)
        self.states = torch.cat([self.states, state.flatten().reshape(1, -1)], dim=0)
        self.actions = torch.cat([self.actions, torch.zeros((1, self.act_dim), dtype=torch.long)], dim=0)



        action = self.predict_action()

        self.actions[-1] = torch.tensor(action, dtype=torch.long) 

        self.timestep += 1
        return action



    # Function that gets an action from the model using autoregressive prediction with a window of the previous 20 timesteps.
    def predict_action(self):
        #adding batch dimention
        ctx_states = self.states.reshape(1, -1, self.state_dim)
        ctx_actions = self.actions.reshape(1, -1, self.act_dim)
        ctx_returns_to_go = self.target_return.reshape(1, -1, 1)
        ctx_timesteps = self.timesteps.reshape(1, -1)

        # cut to context length
        ctx_states = ctx_states[:, -self.max_length :]
        ctx_actions = ctx_actions[:, -self.max_length :]
        ctx_returns_to_go = ctx_returns_to_go[:, -self.max_length :]
        ctx_timesteps = ctx_timesteps[:, -self.max_length :]

        # pad all tokens to sequence length
        padding = self.max_length - ctx_states.shape[1]
        attention_mask = torch.cat([torch.zeros(padding), torch.ones(ctx_states.shape[1])])
        attention_mask = attention_mask.to(dtype=torch.long).reshape(1, -1)
        ctx_states = torch.cat([torch.zeros((1, padding, self.state_dim)), ctx_states], dim=1).float()
        ctx_actions = torch.cat([torch.zeros((1, padding, self.act_dim)), ctx_actions], dim=1).long()
        ctx_returns_to_go = torch.cat([torch.zeros((1, padding, 1)), ctx_returns_to_go], dim=1).float()
        ctx_timesteps = torch.cat([torch.zeros((1, padding), dtype=torch.long), ctx_timesteps], dim=1)

        with torch.no_grad():
            _, action_preds, _ = self.model.original_forward(
                states=ctx_states,
                actions=ctx_actions,
                returns_to_go=ctx_returns_to_go,
                timesteps=ctx_timesteps,
                attention_mask=attention_mask,
                return_dict=False,
            )

        action = torch.argmax(action_preds[0, -1]).item() # action.detach().cpu().numpy()
        return action