import torch

# Define the global device variable
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


RTG_GAMMA = 1.0 # fixed - should make configurable
ACTION_PAD_TOKEN_ID = 5 #yakiv.tbd tmp - move to config!!!
ACTION_VOCAB_SIZE = 6 # 5 actions + 1 PAD token
