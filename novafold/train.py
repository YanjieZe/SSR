from random import randint
import torch
import torch.utils.data as data
import torch.optim as optim
import torch.nn as nn
import numpy as np
import utils
import dataset
import time
from tqdm import tqdm
from arguments import parse_args
from loss import Loss
from tensorboardX import SummaryWriter
import time

try:
    import wandb
except:
	print('Wandb is not installed in your env. Skip `import wandb`.')

### activate tensorboard
writer = SummaryWriter('./logs')
now = time.asctime().replace(" ", "_")
now = now.replace(":","_")

def train(args):
    if args.wandb:
		# wandb.login(key=args.wandb_key)
        wandb.init(project=args.wandb_project, name=args.wandb_name, \
            group=args.wandb_group, job_type=args.wandb_job)
    
    if (args.seed is not None):
        utils.set_seed(args.seed)
    else:
        seed = int(time.time())
        utils.set_seed(seed)
        print("seed: %u "%seed)
    device = torch.device('cuda:0' if args.device == 'gpu' and torch.cuda.is_available() else 'cpu')
    print(f'device: {device}, CUDA Available: {torch.cuda.is_available()}')
    
    train_dataset = dataset.NovaDataset(args)
    train_loader = data.DataLoader(dataset=train_dataset,
                                   batch_size=args.batch_size,
                                   shuffle=True,
                                   num_workers=args.num_workers,
                                   pin_memory=True,
                                   drop_last=True,
                                   collate_fn=dataset.collate_fn_map[args.model]
                                   )
    
    model = utils.build_model(args).to(device).float()
    
    compute_loss = Loss(args, model).to(device)
    
    try:
        optimizer = optim.RAdam(params=model.parameters(), lr=args.lr)
    except:
        optimizer = optim.Adam(params=model.parameters(), lr=args.lr)
        
    
    
    model.train()
    print("Start training...")


    for e in range(args.epoch):
        loop = tqdm(train_loader)
        for idx, (x, y_gt) in enumerate(loop):
        # for idx, (x, y_gt) in enumerate(train_loader):
            # x = torch.from_numpy(x).to(device)
            # y_gt = torch.from_numpy(y_gt).to(device)
            
            loss = compute_loss(x, y_gt)
            writer.add_scalars('Train_loss_' + now, {"Train_loss": float(loss)}, e)
            
            optimizer.zero_grad()
            # loss.register_hook(lambda grad: print(grad))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5, norm_type=2)
            optimizer.step()
            
            if args.wandb:
                wandb.log({'train/loss' : loss.item()})
            
            loop.set_description(f'Epoch [{e}/{args.epoch}], Iter [{idx}/{len(loop)}]')
            loop.set_postfix(loss = loss.item())
            
        utils.save_model(model, e, args)


if __name__ == '__main__':
    args = parse_args()
    print(args)
    train(args)