import torch
import numpy as np
import utils
from arguments import parse_args
try:
    import wandb
except:
	print('Wandb is not installed in your env. Skip `import wandb`.')
 
def predict(args):
    pass

if __name__ == '__main__':
    args = parse_args()
    print(args)
    predict(args)