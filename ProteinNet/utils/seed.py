import random
import numpy as np
import torch

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.set_default_tensor_type('torch.FloatTensor')
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.set_default_tensor_type('torch.cuda.FloatTensor')

        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        use_cuda = True