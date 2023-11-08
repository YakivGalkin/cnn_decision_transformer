import os
import random
from dataclasses import dataclass

import numpy as np
import torch
from datasets import load_dataset
from transformers import Trainer, TrainingArguments
from cnn_decision_transformer.configuration import DecisionTransformerConfig
from cnn_decision_transformer.cnn_decision_transformer_model import CnnDecisionTransformerModel
import torch.nn.functional as F

from utils.config import ACTION_PAD_TOKEN_ID
from utils.timing import execution_timing


@dataclass
class CnnDecisionTransformerGymDataCollator:
    max_len: int #subsets of the episode we use for training
    max_ep_len: int # max episode length in the dataset
    return_tensors: str = "pt"
    state_dim: int = 96*96*3  # size of state space
    act_dim: int = 1  # size of action space - one action at a time
    scale: float = 1000.0  # normalization of rewards/returns
    state_mean: np.array = None  # to store state means
    state_std: np.array = None  # to store state stds
    p_sample: np.array = None  # a distribution to take account trajectory lengths
    n_traj: int = 0 # to store the number of trajectories in the dataset


    def __init__(self, dataset, max_ep_len = 1000, max_len = 20  ) -> None:
        self.max_ep_len = max_ep_len
        self.max_len = max_len
        self.act_dim = 1# len(dataset[0]["actions"][0])
        self.state_dim = len(dataset[0]["observations"][0])
        self.dataset = dataset
        # calculate dataset stats for normalization of states
        states = []
        traj_lens = []
        for obs in dataset["observations"]:
            states.extend(obs)
            traj_lens.append(len(obs))
        self.n_traj = len(traj_lens)
        states = np.vstack(states)
        #yakiv. no normalisation here 
        # self.state_mean, self.state_std = np.mean(states, axis=0), np.std(states, axis=0) + 1e-6

        traj_lens = np.array(traj_lens)
        self.p_sample = traj_lens / sum(traj_lens)

    def _discount_cumsum(self, x, gamma):
        discount_cumsum = np.zeros_like(x)
        discount_cumsum[-1] = x[-1]
        for t in reversed(range(x.shape[0] - 1)):
            discount_cumsum[t] = x[t] + gamma * discount_cumsum[t + 1]
        return discount_cumsum

    def __call__(self, features):
        with execution_timing(f"Collator sz:{len(features)}"):
            batch_size = len(features)
            # this is a bit of a hack to be able to sample of a non-uniform distribution
            batch_inds = np.random.choice(
                np.arange(self.n_traj),
                size=batch_size,
                replace=True,
                p=self.p_sample,  # reweights so we sample according to timesteps
            )
            # a batch of dataset features
            s, a, r, d, rtg, timesteps, mask = [], [], [], [], [], [], []

            for ind in batch_inds:
                # for feature in features:
                feature = self.dataset[int(ind)]
                if(len(feature["rewards"]) < self.max_len):
                    print("skipping batch item")
                    continue
                si = random.randint(0, len(feature["rewards"]) - self.max_len)

                # get sequences from dataset
                s.append(np.array(feature["observations"][si : si + self.max_len]).reshape(1, -1, self.state_dim))
                a.append(np.array(feature["actions"][si : si + self.max_len]).reshape(1, -1, self.act_dim))
                r.append(np.array(feature["rewards"][si : si + self.max_len]).reshape(1, -1, 1))
                rtg.append(np.array(feature["rtg"][si : si + self.max_len]).reshape(1, -1, 1))

                d.append(np.array(feature["dones"][si : si + self.max_len]).reshape(1, -1))
                timesteps.append(np.arange(si, si + s[-1].shape[1]).reshape(1, -1))
                mask.append(np.ones((1, self.max_len)))
                #timesteps[-1][timesteps[-1] >= self.max_ep_len] = self.max_ep_len - 1  # padding cutoff
                # rtg.append(
                #     self._discount_cumsum(np.array(feature["rewards"][si:]), gamma=1.0)[
                #         : s[-1].shape[1]   # TODO check the +1 removed here
                #     ].reshape(1, -1, 1)
                # )
                # if rtg[-1].shape[1] < s[-1].shape[1]:
                #     print("if true")
                #     rtg[-1] = np.concatenate([rtg[-1], np.zeros((1, 1, 1))], axis=1)

                # padding and state + reward normalization
                ### REMOVED PADDING
            #     tlen = s[-1].shape[1]
            #     s[-1] = np.concatenate([np.zeros((1, self.max_len - tlen, self.state_dim)), s[-1]], axis=1)
            #     #yakiv. no normalisation here
            # #  s[-1] = (s[-1] - self.state_mean) / self.state_std
            #     a[-1] = np.concatenate(
            #         [np.ones((1, self.max_len - tlen, self.act_dim)) * ACTION_PAD_TOKEN_ID, a[-1]],
            #         axis=1,
            #     )
            #     r[-1] = np.concatenate([np.zeros((1, self.max_len - tlen, 1)), r[-1]], axis=1)
            #     d[-1] = np.concatenate([np.ones((1, self.max_len - tlen)) * 2, d[-1]], axis=1)
            #     rtg[-1] = np.concatenate([np.zeros((1, self.max_len - tlen, 1)), rtg[-1]], axis=1) / self.scale
            #     timesteps[-1] = np.concatenate([np.zeros((1, self.max_len - tlen)), timesteps[-1]], axis=1)
            #     mask.append(np.concatenate([np.zeros((1, self.max_len - tlen)), np.ones((1, tlen))], axis=1))
            

            s = torch.from_numpy(np.concatenate(s, axis=0)).float()
            a = torch.from_numpy(np.concatenate(a, axis=0)).long()
            r = torch.from_numpy(np.concatenate(r, axis=0)).float()
            d = torch.from_numpy(np.concatenate(d, axis=0))
            rtg = torch.from_numpy(np.concatenate(rtg, axis=0)).float()
            timesteps = torch.from_numpy(np.concatenate(timesteps, axis=0)).long()
            mask = torch.from_numpy(np.concatenate(mask, axis=0)).float()

            return {
                "states": s,
                "actions": a,
                "rewards": r,
                "returns_to_go": rtg,
                "timesteps": timesteps,
                "attention_mask": mask,
            }
    
class TrainableCnnDecisionTransformer(CnnDecisionTransformerModel):
    def __init__(self, config,  pretrained_cnn_feature_extractor: bool = True):
        super().__init__(config, pretrained_cnn_feature_extractor=pretrained_cnn_feature_extractor)

    def forward(self, **kwargs):
        output = super().forward(**kwargs)
        with execution_timing("Trainable Forward pass"):
            #  output[1] contains the raw logits for the actions
            action_preds = output[1]
            action_targets = kwargs["actions"]
            attention_mask = kwargs["attention_mask"]

            attention_mask = (attention_mask.reshape(-1) > 0)

            # We expect action_preds to be a 3D tensor: [batch_size, sequence_length, num_actions]
            # After masking, we need action_preds to be 2D: [num_valid_entries, num_actions]
            action_preds = action_preds.reshape(-1, action_preds.size(-1))
            masked_action_preds = torch.masked_select(action_preds, attention_mask.unsqueeze(-1)).view(-1, action_preds.size(-1))

            # After masking, we need action_targets to be 1D: [num_valid_entries]
            action_targets = action_targets.reshape(-1)
            masked_action_targets = torch.masked_select(action_targets, attention_mask)

            loss = F.cross_entropy(masked_action_preds, masked_action_targets)


            # # add the DT loss
            # action_preds = output[1]
            # action_targets = kwargs["actions"]
            # attention_mask = kwargs["attention_mask"]
            # act_dim = action_preds.shape[2]
            # action_preds = action_preds.reshape(-1, act_dim)[attention_mask.reshape(-1) > 0]
            # action_targets = action_targets.reshape(-1, act_dim)[attention_mask.reshape(-1) > 0]

            # loss = torch.mean((action_preds - action_targets) ** 2)
            print(f"loss: {loss}")
            return {"loss": loss}

    def original_forward(self, **kwargs):
        return super().forward(**kwargs)



# collator = DecisionTransformerGymDataCollator(dataset_half["train"])

# config = DecisionTransformerConfig(state_dim=collator.state_dim, act_dim=collator.act_dim)
# model = TrainableDT(config)


# training_args = TrainingArguments(
#     output_dir="output/",
#     remove_unused_columns=False,
#     num_train_epochs=120,
#     per_device_train_batch_size=64,
#     learning_rate=1e-4,
#     weight_decay=1e-4,
#     warmup_ratio=0.1,
#     optim="adamw_torch",
#     max_grad_norm=0.25,
# )

# trainer = Trainer(
#     model=model,
#     args=training_args,
#     train_dataset=dataset_half["train"],
#     data_collator=collator,
# )

# trainer.train()

# import gymnasyium as gym

# # Function that gets an action from the model using autoregressive prediction with a window of the previous 20 timesteps.
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
    actions = torch.cat([torch.zeros((1, padding, model.config.act_dim)), actions], dim=1).float()
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



# #import gymnasium as gym
# # build the environment
# directory = './video'
# model = model.to("cpu")
# env = gym.make("HalfCheetah-v3")
# env = Recorder(env, directory, fps=30)
# max_ep_len = 1000
# device = "cpu"
# scale = 1000.0  # normalization for rewards/returns
# TARGET_RETURN = 12000 / scale  # evaluation is conditioned on a return of 12000, scaled accordingly

# state_mean = collator.state_mean.astype(np.float32)
# state_std = collator.state_std.astype(np.float32)
# print(state_mean)

# state_dim = env.observation_space.shape[0]
# act_dim = env.action_space.shape[0]
# # Create the decision transformer model

# state_mean = torch.from_numpy(state_mean).to(device=device)
# state_std = torch.from_numpy(state_std).to(device=device)


# # Interact with the environment and create a video
# episode_return, episode_length = 0, 0
# state = env.reset()
# target_return = torch.tensor(TARGET_RETURN, device=device, dtype=torch.float32).reshape(1, 1)
# states = torch.from_numpy(state).reshape(1, state_dim).to(device=device, dtype=torch.float32)
# actions = torch.zeros((0, act_dim), device=device, dtype=torch.float32)
# rewards = torch.zeros(0, device=device, dtype=torch.float32)

# timesteps = torch.tensor(0, device=device, dtype=torch.long).reshape(1, 1)
# for t in range(max_ep_len):
#     actions = torch.cat([actions, torch.zeros((1, act_dim), device=device)], dim=0)
#     rewards = torch.cat([rewards, torch.zeros(1, device=device)])

#     action = get_action(
#         model,
#         (states - state_mean) / state_std,
#         actions,
#         rewards,
#         target_return,
#         timesteps,
#     )
#     actions[-1] = action
#     action = action.detach().cpu().numpy()

#     state, reward, done, _ = env.step(action)

#     cur_state = torch.from_numpy(state).to(device=device).reshape(1, state_dim)
#     states = torch.cat([states, cur_state], dim=0)
#     rewards[-1] = reward

#     pred_return = target_return[0, -1] - (reward / scale)
#     target_return = torch.cat([target_return, pred_return.reshape(1, 1)], dim=1)
#     timesteps = torch.cat([timesteps, torch.ones((1, 1), device=device, dtype=torch.long) * (t + 1)], dim=1)

#     episode_return += reward
#     episode_length += 1

#     if done:
#         break
