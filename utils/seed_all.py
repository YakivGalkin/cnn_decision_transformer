import random
import numpy as np
import torch
# import tensorflow as tf  # Uncomment if you're using TensorFlow

def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # tf.random.set_seed(seed)  # Uncomment if you're using TensorFlow
    # Add other libraries' seeding mechanisms here as needed.
