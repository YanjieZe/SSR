import torch
import torch.nn as nn
import numpy as np
from models.mxfold2.loss import StructuredLoss, StructuredLossWithTurner
### e2e specific
from utils import f1_loss

class Loss(nn.Module):
    def __init__(self, args, model):
        super(Loss, self).__init__()
        self.args = args
        self.verbose = args.verbose
        self.device = torch.device('cuda:0' if args.device == 'gpu' and torch.cuda.is_available() else 'cpu')
        
        if args.model == 'mxfold2':
            if args.loss_func == 'hinge':
                self.loss_fn = StructuredLoss(model, verbose=self.verbose,
                                loss_pos_paired=args.loss_pos_paired, loss_neg_paired=args.loss_neg_paired, 
                                loss_pos_unpaired=args.loss_pos_unpaired, loss_neg_unpaired=args.loss_neg_unpaired, 
                                l1_weight=args.l1_weight, l2_weight=args.l2_weight)
            elif args.loss_func == 'hinge_mix':
                self.loss_fn = StructuredLossWithTurner(model, verbose=self.verbose,
                                loss_pos_paired=args.loss_pos_paired, loss_neg_paired=args.loss_neg_paired, 
                                loss_pos_unpaired=args.loss_pos_unpaired, loss_neg_unpaired=args.loss_neg_unpaired, 
                                l1_weight=args.l1_weight, l2_weight=args.l2_weight, sl_weight=args.score_loss_weight)
            else:
                raise RuntimeError(f'{args.loss_func} loss for mxfold2 is not implemented yet.')
        elif args.model == 'e2efold':
            pos_weight = torch.Tensor([300])
            self.pp_steps = args.pp_steps
            self.pp_loss = args.pp_loss
            self.model = model
            self.step_gamma = args.step_gamma
            self.loss_fn = torch.nn.BCEWithLogitsLoss(
                pos_weight = pos_weight)
            self.criterion_bce_weighted = torch.nn.BCEWithLogitsLoss(
                pos_weight = pos_weight)
            self.criterion_mse = torch.nn.MSELoss(reduction='sum')
        else:
            raise NotImplementedError(f'Loss for {args.model} is not implemented yet.')
    
    def forward(self, x, y):
        if self.args.model == 'mxfold2':
            # pred = self.model(x)
            return self.loss_fn(x, y)
        elif self.args.model == 'e2efold':
            (PE_batch, seq_embedding_batch, state_pad, contact_masks) = x
            PE_batch, seq_embedding_batch, state_pad, contact_masks =\
                 PE_batch.to(self.device), seq_embedding_batch.to(self.device), state_pad.to(self.device), contact_masks.to(self.device)
            contacts_batch = y.to(self.device)
            # print(PE_batch.shape, seq_embedding_batch.shape, state_pad.shape, contact_masks.shape, contacts_batch.shape)
            # pred_contacts = self.model(PE_batch, seq_embedding_batch, state_pad)
            pred_contacts, a_pred_list = self.model(PE_batch, 
                seq_embedding_batch, state_pad)
            # print(pred_contacts.shape)
            # loss = self.loss_fn(pred_contacts*contact_masks, contacts_batch)
            loss_u = self.criterion_bce_weighted(pred_contacts*contact_masks, contacts_batch)

            if self.pp_loss == "l2":
                loss_a = self.criterion_mse(
                    a_pred_list[-1]*contact_masks, contacts_batch)
                for i in range(self.pp_steps-1):
                    loss_a += np.power(self.step_gamma, self.pp_steps-1-i)*self.criterion_mse(
                        a_pred_list[i]*contact_masks, contacts_batch)
                # 600 is a fixed number, no need to change
                mse_coeff = 1.0/(600*self.pp_steps)

            if self.pp_loss == 'f1':
                loss_a = f1_loss(a_pred_list[-1]*contact_masks, contacts_batch)
                for i in range(self.pp_steps-1):
                    loss_a += np.power(self.step_gamma, self.pp_steps-1-i)*f1_loss(
                        a_pred_list[i]*contact_masks, contacts_batch)            
                mse_coeff = 1.0/self.pp_steps
            
            loss_a = mse_coeff*loss_a

            loss = loss_u + loss_a

            # print(loss)
            return loss
        else:
            raise NotImplementedError(f'Loss for {self.args.model} is not implemented yet.')
