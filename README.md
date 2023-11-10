# Applying Transformer Architecture to classical car-driving RL tasks

This project investigates the application of transformer architectures in classical car-driving reinforcement learning tasks.

## Demonstration

The project has led to improvements in the Hugging Face Decision Transformer, allowing it to process visual inputs and discrete actions. Major steps included training a basic Deep Q-Network (DQN) to explore Gym CarRacing V2 the environment, creating an offline dataset, and further training of the Decision Transformer with this data.


| DQN (Deep Q Learning) | Visual Decision Transformer |
| --------------------- | --------------------------- |
| ![DQN GIF](./media/dqn_ride.gif) | ![VDT GIF](./media/vdt_ride.gif) |
| DQN Training | VDT Training |
| Generating Offline Datasewt |  |


## Transformers in RL 

A good overview of Transformer Architecture applications in RL can be found in the paper:
[A Survey on Transformers in Reinforcement Learning](https://arxiv.org/abs/2301.03044).

Two primary uses of transformers are:
- Representation Learning
- Sequential Decision-making

In the scope of this project, I decided to focus on Sequential Decision-making.
One of the first papers in this regard was [Offline Reinforcement Learning as One Big Sequence Modeling Problem](https://arxiv.org/abs/2106.02039).
As part of the experiments, the authors implemented a Trajectory Transformer that discretizes all continuous states, actions, and rewards (a quantile discretizer was used by default) and then trained it on a sequence of states, actions, and rewards. 

![Trajectory Transformer Tokens](./media/trajectory_transformer_tokens.jpg)

As a result, the transformer learns common behavior (imitation learning) from offline training data. Later, by using beam search as a limited horizon reward-maximising procedure, the Trajectory Transformer can predict the best trajectory based on past experience.
For image-based observations, this model doesn't seem appropriate as it heavily relies on discretization.

Next, I paid attention to the paper:
[Decision Transformer: Reinforcement Learning via Sequence Modeling](https://arxiv.org/abs/2106.01345).
The key idea of the Decision Transformer was to replace the reward in the trajectory transformer with the future cumulative reward (often called return). By conditioning a pre-trained transformer with a desired future return, it can predict the best trajectories that should achieve it. 

![Decision Transformer Tokens](./media/decision_transformer_training.jpg)

The authors also discovered that knowledge of the possible future return is important when using their approach.

The HuggingFace community has already included the Decision Transformer as part of their transformers library:
[Transformers Model Doc: Decision Transformer](https://huggingface.co/docs/transformers/model_doc/decision_transformer).
Unfortunately, their implementation is strictly dedicated to continuous state and action spaces, with a comment "image-based states will come soon". Implementing this for image-based states sounds like a good challenge, which becomes **the purpose of this project, which I have named Visual Decision Transformer**. 

### Implementing Visual Decision Transformer

Exploration from the paper "A Survey on Transformers" states, "Interestingly, to the best of our knowledge, we do not find related works about learning DT in the pure online setting without any pre-training. We suspect that this is because pure online learning will exacerbate the non-stationary nature of training data, which will severely harm the performance of the current DT policy. Instead, offline pre-training can help to warm up a well-behaved policy that is stable for further online training." So, as a first step, we need to build an offline dataset by exploring the environment with one of the classic RL algorithms.

As a result, the major steps become:
- Reviewing/choosing a classic RL algorithm/model for exploration of the CarRacingV2 environment
- Training the chosen classic RL model on the environment
- Building a dataset, storing it for future use
- Modifying HuggingFace's decision transformer to support image data and discrete actions
- Training the Visual Decision Transformer
- Evaluation, comparison, and experiments.
