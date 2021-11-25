import torch
import torch.nn as nn
from models.mxfold2.loss import StructuredLoss, StructuredLossWithTurner

class Loss(nn.Module):
    def __init__(self, args, model):
        super(Loss, self).__init__()
        self.args = args
        self.verbose = args.verbose
        
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
            self.model = model
            self.loss_fn = torch.nn.BCEWithLogitsLoss(
                pos_weight = pos_weight)
        else:
            raise NotImplementedError(f'Loss for {args.model} is not implemented yet.')
    
    def forward(self, x, y):
        if self.args.model == 'mxfold2':
            # pred = self.model(x)
            return self.loss_fn(x, y)
        elif self.args.model == 'e2efold':
            (PE_batch, seq_embedding_batch, state_pad, contact_masks) = x
            contacts_batch = y
            # print(PE_batch.shape, seq_embedding_batch.shape, state_pad.shape, contact_masks.shape, contacts_batch.shape)
            pred_contacts = self.model(PE_batch, seq_embedding_batch, state_pad)
            # print(pred_contacts.shape)
            loss = self.loss_fn(pred_contacts*contact_masks, contacts_batch)
            # print(loss)
            return loss
        else:
            raise NotImplementedError(f'Loss for {self.args.model} is not implemented yet.')
