import torch
import math
import numpy as np
import os
import random

from torch._C import device
import utils
from torch.utils.data import Dataset
from itertools import groupby

class NovaDataset(Dataset):
    def __init__(self, args):
        if args.model == 'e2efold':
            self.seq_max_len = args.seq_max_len
        self.path_list = []
        self.model = args.model
        # self.data = []
        dataset = args.train_set.split(',')
        
        for lst in dataset:
            path = os.path.join(args.data_root, lst)
            print(f'Reading data from {path}')
            with open(os.path.join(path)) as f:
                for l in f:
                    l = l.rstrip('\n').split()
                    if len(l)==1:
                        # self.data.append(self.read(l[0], args))
                        count = 0
                        with open(os.path.join(l[0])) as seqfile:
                            for index, line in enumerate(seqfile):
                                count += 1
                        # print(count)
                        if count <= args.seq_max_len:
                            self.path_list.append(l[0])
                        else:
                            print(f"length of file {l[0]} is {count}, exceeding {args.seq_max_len}")
                    elif len(l)==2:
                        raise NotImplementedError("Original: read_pdb() in mxfold2. Not implemented here.")
        # print(f'Dataset size: {len(self.data)}')
        print(f'Dataset size: {len(self.path_list)}')
    def __len__(self):
        return len(self.path_list)
        # return len(self.data)

    def __getitem__(self, idx):
        return self.read(self.path_list[idx])
        # return self.data[idx]
        
    def read(self, filename):
        if self.model == 'mxfold2' or self.model == 'linearfold':
            with open(filename) as f:
                pairs = [0]
                s = ['']
                for l in f:
                    if not l.startswith('#'):
                        l = l.rstrip('\n').split()
                        if len(l) == 3:
                            idx, c, pair = l
                            pos = 'x.<>|'.find(pair)
                            if pos >= 0:
                                # idx, pair = int(idx), -pos
                                raise RuntimeError(f'NovaDataset::read(): pos >= 0 caught')
                            idx, pair = int(idx), int(pair)
                            s.append(c)
                            pairs.append(pair)
                        else:
                            raise RuntimeError('invalid format: {}'.format(filename))
            seq = ''.join(s)
            # print(len(seq))
            # print(torch.tensor(pairs).shape)
            return seq, torch.tensor(pairs)
            # return (filename, seq, torch.tensor(pairs))
        elif self.model == 'e2efold':
            with open(filename) as f:
                pairs = []
                s = ['']
                for l in f:
                    if not l.startswith('#'):
                        l = l.rstrip('\n').split()
                        if len(l) == 3:
                            idx, c, pair = l
                            pos = 'x.<>|'.find(pair)
                            if pos >= 0:
                                # idx, pair = int(idx), -pos
                                raise RuntimeError(f'NovaDataset::read(): pos >= 0 caught')
                            idx, pair = int(idx), int(pair)
                            s.append(c)
                            pairs.append(pair)
                        else:
                            raise RuntimeError('invalid format: {}'.format(filename))
            seq = ''.join(s)

            contact = utils.p2contact(self.seq_max_len, pairs)
            data_seq = torch.tensor(utils.seq2encoding(self.seq_max_len, seq))
            matrix_rep = torch.zeros(contact.shape)
            data_len = torch.tensor(len(seq))

            # contact = torch.tensor(utils.p2contact(self.seq_max_len, pairs)).unsqueeze(0)
            # data_seq = torch.tensor(utils.seq2encoding(self.seq_max_len, seq)).unsqueeze(0)
            # matrix_rep = torch.zeros(contact.shape).unsqueeze(0)
            # data_len = torch.tensor(len(seq)).unsqueeze(0)
            
            # contacts_batch = torch.Tensor(contact).float()
            # seq_embedding_batch = data_seq.float()
            # matrix_reps_batch = torch.unsqueeze(
            #     torch.Tensor(matrix_rep).float(), -1)

            # # padding the states for supervised training with all 0s
            # state_pad = torch.zeros([matrix_reps_batch.shape[0], 
            #     data_len, data_len])

            # PE_batch = utils.get_pe(data_len, data_len).float()
            # contact_masks = torch.Tensor(utils.contact_map_masks(data_len, data_len))
            # print(PE_batch.shape, seq_embedding_batch.shape, state_pad.shape, contact_masks.shape, contacts_batch.shape)
            # return ((PE_batch.squeeze(0), seq_embedding_batch.squeeze(0), state_pad.squeeze(0), contact_masks.squeeze(0)), contacts_batch)
            return (contact, data_seq, matrix_rep, data_len)
        else:
            raise NotImplementedError(f'NovaDataset::read(): Unimplemented Model {self.model}')

class NovaDatasetV2(Dataset):
    def __init__(self, args):
        if args.model == 'e2efold':
            self.seq_max_len = args.seq_max_len
        self.path_list = []
        self.model = args.model
        # self.data = []
        dataset = args.train_set.split(',')
        
        for lst in dataset:
            path = os.path.join(args.data_root, lst)
            print(f'Reading data from {path}')
            with open(os.path.join(path)) as f:
                for l in f:
                    l = l.rstrip('\n').split()
                    if len(l)==1:
                        # self.data.append(self.read(l[0], args))
                        count = 0
                        with open(os.path.join(l[0])) as seqfile:
                            for index, line in enumerate(seqfile):
                                count += 1
                        # print(count)
                        if count <= args.seq_max_len:
                            self.path_list.append(l[0])
                        else:
                            print(f"length of file {l[0]} is {count}, exceeding {args.seq_max_len}")
                    elif len(l)==2:
                        raise NotImplementedError("Original: read_pdb() in mxfold2. Not implemented here.")
        # print(f'Dataset size: {len(self.data)}')
        print(f'Dataset size: {len(self.path_list)}')
    def __len__(self):
        return len(self.path_list)
        # return len(self.data)

    def __getitem__(self, idx):
        return self.read(self.path_list[idx])
        # return self.data[idx]
        
    def read(self, filename):
        if self.model == 'mxfold2' or self.model == 'linearfold':
            with open(filename) as f:
                pairs = [0]
                s = ['']
                for l in f:
                    if not l.startswith('#'):
                        l = l.rstrip('\n').split()
                        if len(l) == 3:
                            idx, c, pair = l
                            pos = 'x.<>|'.find(pair)
                            if pos >= 0:
                                # idx, pair = int(idx), -pos
                                raise RuntimeError(f'NovaDataset::read(): pos >= 0 caught')
                            idx, pair = int(idx), int(pair)
                            s.append(c)
                            pairs.append(pair)
                        else:
                            raise RuntimeError('invalid format: {}'.format(filename))
            seq = ''.join(s)
            # print(len(seq))
            # print(torch.tensor(pairs).shape)
            return seq, torch.tensor(pairs), filename
            # return (filename, seq, torch.tensor(pairs))
        elif self.model == 'e2efold':
            with open(filename) as f:
                pairs = []
                s = ['']
                for l in f:
                    if not l.startswith('#'):
                        l = l.rstrip('\n').split()
                        if len(l) == 3:
                            idx, c, pair = l
                            pos = 'x.<>|'.find(pair)
                            if pos >= 0:
                                # idx, pair = int(idx), -pos
                                raise RuntimeError(f'NovaDataset::read(): pos >= 0 caught')
                            idx, pair = int(idx), int(pair)
                            s.append(c)
                            pairs.append(pair)
                        else:
                            raise RuntimeError('invalid format: {}'.format(filename))
            seq = ''.join(s)

            contact = utils.p2contact(self.seq_max_len, pairs)
            data_seq = torch.tensor(utils.seq2encoding(self.seq_max_len, seq))
            matrix_rep = torch.zeros(contact.shape)
            data_len = torch.tensor(len(seq))

            # contact = torch.tensor(utils.p2contact(self.seq_max_len, pairs)).unsqueeze(0)
            # data_seq = torch.tensor(utils.seq2encoding(self.seq_max_len, seq)).unsqueeze(0)
            # matrix_rep = torch.zeros(contact.shape).unsqueeze(0)
            # data_len = torch.tensor(len(seq)).unsqueeze(0)
            
            # contacts_batch = torch.Tensor(contact).float()
            # seq_embedding_batch = data_seq.float()
            # matrix_reps_batch = torch.unsqueeze(
            #     torch.Tensor(matrix_rep).float(), -1)

            # # padding the states for supervised training with all 0s
            # state_pad = torch.zeros([matrix_reps_batch.shape[0], 
            #     data_len, data_len])

            # PE_batch = utils.get_pe(data_len, data_len).float()
            # contact_masks = torch.Tensor(utils.contact_map_masks(data_len, data_len))
            # print(PE_batch.shape, seq_embedding_batch.shape, state_pad.shape, contact_masks.shape, contacts_batch.shape)
            # return ((PE_batch.squeeze(0), seq_embedding_batch.squeeze(0), state_pad.squeeze(0), contact_masks.squeeze(0)), contacts_batch)
            return (contact, data_seq, matrix_rep, data_len)
        else:
            raise NotImplementedError(f'NovaDataset::read(): Unimplemented Model {self.model}')

def collate_fn_e2e(data):
    contacts = []
    seq_embeddings = []
    matrix_reps = []
    seq_lens = []
    for unit in data:
        contacts.append(unit[0])
        seq_embeddings.append(unit[1])
        matrix_reps.append(unit[2])
        seq_lens.append(unit[3])
    
    contacts = torch.stack(contacts, axis=0)
    seq_embeddings = torch.stack(seq_embeddings, axis=0)
    matrix_reps = torch.stack(matrix_reps, axis=0)
    seq_lens = torch.stack(seq_lens, axis=0)

    seq_max_len = seq_embeddings.shape[-2]
    # print(seq_max_len)

    contacts_batch = torch.Tensor(contacts.float())
    seq_embedding_batch = torch.Tensor(seq_embeddings.float())
    matrix_reps_batch = torch.unsqueeze(
        torch.Tensor(matrix_reps.float()), -1)

    # padding the states for supervised training with all 0s
    state_pad = torch.zeros([matrix_reps_batch.shape[0], 
        seq_max_len, seq_max_len])


    PE_batch = utils.get_pe(seq_lens, seq_max_len).float()
    contact_masks = torch.Tensor(utils.contact_map_masks(seq_lens, seq_max_len))
    
    return ((PE_batch, seq_embedding_batch, state_pad, contact_masks), contacts_batch)

collate_fn_map = {'e2efold' : collate_fn_e2e, 'mxfold2' : None, 'linearfold' : None}
