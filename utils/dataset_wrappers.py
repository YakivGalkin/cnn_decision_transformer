from datasets import Dataset

class CarRacingFeatureDataset(Dataset):
    def __init__(self, src):
        self.size = len(src["observations"])  # Assuming all lists are the same length
        self.src = src

    def __len__(self):
        return self.size
    
    def __getitems__(self, index):
        return [self._item(i) for i in index]

    def _item(self, idx):
        # It is better to ensure this is an internal method used within the class only.
        if isinstance(idx, str):
            return self.src[idx]
        
        return {
            "observations": self.src["observations"][idx],
            "actions": self.src["actions"][idx], 
            "rewards": self.src["rewards"][idx],
            "dones": self.src["dones"][idx],
            "rtg": self.src["rtg"][idx]
        }
    
    def __getitem__(self, index):
        # Here, we use 'index' instead of 'i'
        return self._item(index)


class DummyDataset(Dataset):
    def __init__(self, size):
        self.size = size
        self.arr = [0] * size
    def __len__(self):
        return self.size
    def __getitems__(self, index):
        return index
    def __getitem__(self, index):
        return index
