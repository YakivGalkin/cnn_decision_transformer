import math
import os
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn
from torch.cuda.amp import autocast

from transformers.activations import ACT2FN
from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions
from transformers.modeling_utils import PreTrainedModel
from transformers.pytorch_utils import Conv1D, find_pruneable_heads_and_indices, prune_conv1d_layer
from transformers.utils import (
    ModelOutput,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
)
from cnn_decision_transformer.configuration import DecisionTransformerConfig
from cnn_decision_transformer.cnn_feature_extractor import get_cnn_feature_extractor_model
from utils.storage import get_pretrained_model
from utils.timing import execution_timing
import cnn_decision_transformer.gpt2_base_transformer as gpt2_base_transformer
from stable_baselines3.common.torch_layers import NatureCNN

_CONFIG_FOR_DOC = "DecisionTransformerConfig"

@dataclass
class DecisionTransformerOutput(ModelOutput):
    """
    Base class for model's outputs that also contains a pooling of the last hidden states.

    Args:
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
        state_preds (`torch.FloatTensor` of shape `(batch_size, sequence_length, state_dim)`):
            Environment state predictions
        action_preds (`torch.FloatTensor` of shape `(batch_size, sequence_length, action_dim)`):
            Model action predictions
        return_preds (`torch.FloatTensor` of shape `(batch_size, sequence_length, 1)`):
            Predicted returns for each state
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer) of
            shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    state_preds: torch.FloatTensor = None
    action_preds: torch.FloatTensor = None
    return_preds: torch.FloatTensor = None
    hidden_states: torch.FloatTensor = None
    attentions: torch.FloatTensor = None
    last_hidden_state: torch.FloatTensor = None


class DecisionTransformerPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = DecisionTransformerConfig
    base_model_prefix = "decision_transformer"
    main_input_name = "states"
    supports_gradient_checkpointing = False

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, nn.Linear):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)


DECISION_TRANSFORMER_START_DOCSTRING = r"""
    This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) sub-class. Use
    it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and
    behavior.

    Parameters:
        config ([`~DecisionTransformerConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

DECISION_TRANSFORMER_INPUTS_DOCSTRING = r"""
    Args:
        states (`torch.FloatTensor` of shape `(batch_size, episode_length, state_dim)`):
            The states for each step in the trajectory
        actions (`torch.FloatTensor` of shape `(batch_size, episode_length, act_dim)`):
            The actions taken by the "expert" policy for the current state, these are masked for auto regressive
            prediction
        rewards (`torch.FloatTensor` of shape `(batch_size, episode_length, 1)`):
            The rewards for each state, action
        returns_to_go (`torch.FloatTensor` of shape `(batch_size, episode_length, 1)`):
            The returns for each state in the trajectory
        timesteps (`torch.LongTensor` of shape `(batch_size, episode_length)`):
            The timestep for each step in the trajectory
        attention_mask (`torch.FloatTensor` of shape `(batch_size, episode_length)`):
            Masking, used to mask the actions when performing autoregressive prediction
"""


@add_start_docstrings("The Decision Transformer Model", DECISION_TRANSFORMER_START_DOCSTRING)
class CnnDecisionTransformerModel(DecisionTransformerPreTrainedModel):
    """

    The model builds upon the GPT2 architecture to perform autoregressive prediction of actions in an offline RL
    setting. Refer to the paper for more details: https://arxiv.org/abs/2106.01345

    """

    def __init__(self, config, pretrained_cnn_feature_extractor=True):
        super().__init__(config)
        self.config = config
        self.hidden_size = config.hidden_size
        # note: the only difference between this GPT2Model and the default Huggingface version
        # is that the positional embeddings are removed (since we'll add those ourselves)
        self.encoder = gpt2_base_transformer.DecisionTransformerGPT2Model(config)

        self.embed_timestep = nn.Embedding(config.max_ep_len, config.hidden_size)
        self.embed_return = torch.nn.Linear(1, config.hidden_size)
        
        if pretrained_cnn_feature_extractor:
            self.cnn_state_feature_extractor = get_cnn_feature_extractor_model(pretrained_model_name = 'nature_cnn_dql_pretrained.pt')
        else:
            self.cnn_state_feature_extractor = get_cnn_feature_extractor_model()
        self.embed_state = torch.nn.Linear(512, config.hidden_size)
        self.embed_action = nn.Embedding(config.action_num, config.hidden_size) 

        self.embed_ln = nn.LayerNorm(config.hidden_size)

        # note: we don't predict states or returns for the paper
        self.predict_state = torch.nn.Linear(config.hidden_size, config.state_dim)
        self.predict_action = torch.nn.Linear(config.hidden_size, config.action_num) #= nn.Sequential(            *([ linear was here!!!] + ([nn.Tanh()] if config.action_tanh else []))        )
        self.predict_return = torch.nn.Linear(config.hidden_size, 1)

        # Initialize weights and apply final processing
        self.post_init()

    @add_start_docstrings_to_model_forward(DECISION_TRANSFORMER_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @replace_return_docstrings(output_type=DecisionTransformerOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        states: Optional[torch.FloatTensor] = None,
        actions: Optional[torch.LongTensor] = None,
        rewards: Optional[torch.FloatTensor] = None,
        returns_to_go: Optional[torch.FloatTensor] = None,
        timesteps: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        output_hidden_states: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.FloatTensor], DecisionTransformerOutput]:
        r"""
        Returns:

        Examples:

        ```python
        >>> from transformers import DecisionTransformerModel
        >>> import torch

        >>> model = DecisionTransformerModel.from_pretrained("edbeeching/decision-transformer-gym-hopper-medium")
        >>> # evaluation
        >>> model = model.to(device)
        >>> model.eval()

        >>> env = gym.make("Hopper-v3")
        >>> state_dim = env.observation_space.shape[0]
        >>> act_dim = env.action_space.shape[0]

        >>> state = env.reset()
        >>> states = torch.from_numpy(state).reshape(1, 1, state_dim).to(device=device, dtype=torch.float32)
        >>> actions = torch.zeros((1, 1, act_dim), device=device, dtype=torch.float32)
        >>> rewards = torch.zeros(1, 1, device=device, dtype=torch.float32)
        >>> target_return = torch.tensor(TARGET_RETURN, dtype=torch.float32).reshape(1, 1)
        >>> timesteps = torch.tensor(0, device=device, dtype=torch.long).reshape(1, 1)
        >>> attention_mask = torch.zeros(1, 1, device=device, dtype=torch.float32)

        >>> # forward pass
        >>> with torch.no_grad():
        ...     state_preds, action_preds, return_preds = model(
        ...         states=states,
        ...         actions=actions,
        ...         rewards=rewards,
        ...         returns_to_go=target_return,
        ...         timesteps=timesteps,
        ...         attention_mask=attention_mask,
        ...         return_dict=False,
        ...     )
        ```"""
        with execution_timing(f"DecisionTransformerModel.forward batch on device: { str(states.device)} sz:{states.shape[0]} seq len: {states.shape[1]} "):
            output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
            output_hidden_states = (
                output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
            )
            return_dict = return_dict if return_dict is not None else self.config.use_return_dict

            batch_size, seq_length = states.shape[0], states.shape[1]

            if attention_mask is None:
                # attention mask for GPT: 1 if can be attended to, 0 if not
                attention_mask = torch.ones((batch_size, seq_length), dtype=torch.long)

            # embed each modality with a different head
            states = states.view(-1, 3, 96, 96)
            state_features = self.cnn_state_feature_extractor( states )
            state_features = state_features.view(batch_size, seq_length, -1)
            state_embeddings = self.embed_state(  state_features   )
            action_embeddings = self.embed_action(actions.view(batch_size, seq_length))
            returns_embeddings = self.embed_return(returns_to_go)
            time_embeddings = self.embed_timestep(timesteps)

            # time embeddings are treated similar to positional embeddings
            state_embeddings = state_embeddings + time_embeddings
            action_embeddings = action_embeddings + time_embeddings
            returns_embeddings = returns_embeddings + time_embeddings

            # this makes the sequence look like (R_1, s_1, a_1, R_2, s_2, a_2, ...)
            # which works nice in an autoregressive sense since states predict actions
            stacked_inputs = (
                torch.stack((returns_embeddings, state_embeddings, action_embeddings), dim=1)
                .permute(0, 2, 1, 3)
                .reshape(batch_size, 3 * seq_length, self.hidden_size)
            )
            stacked_inputs = self.embed_ln(stacked_inputs)

            # to make the attention mask fit the stacked inputs, have to stack it as well
            stacked_attention_mask = (
                torch.stack((attention_mask, attention_mask, attention_mask), dim=1)
                .permute(0, 2, 1)
                .reshape(batch_size, 3 * seq_length)
            )
            device = stacked_inputs.device
            # we feed in the input embeddings (not word indices as in NLP) to the model
            encoder_outputs = self.encoder(
                inputs_embeds=stacked_inputs,
                attention_mask=stacked_attention_mask,
                position_ids=torch.zeros(stacked_attention_mask.shape, device=device, dtype=torch.long),
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
            x = encoder_outputs[0]

            # reshape x so that the second dimension corresponds to the original
            # returns (0), states (1), or actions (2); i.e. x[:,1,t] is the token for s_t
            x = x.reshape(batch_size, seq_length, 3, self.hidden_size).permute(0, 2, 1, 3)

            # get predictions
            return_preds = self.predict_return(x[:, 2])  # predict next return given state and action
            state_preds = self.predict_state(x[:, 2])  # predict next state given state and action
            action_preds = self.predict_action(x[:, 1])  # predict next action given state
            action_preds[..., -1] = float('-inf') # mask out the PAD token
            if not return_dict:
                return (state_preds, action_preds, return_preds)

            return DecisionTransformerOutput(
                last_hidden_state=encoder_outputs.last_hidden_state,
                state_preds=state_preds,
                action_preds=action_preds,
                return_preds=return_preds,
                hidden_states=encoder_outputs.hidden_states,
                attentions=encoder_outputs.attentions,
            )
