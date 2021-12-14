import argparse


def parse_args(args=None):
    parser = argparse.ArgumentParser()
    
    # basic
    parser.add_argument('--model', default='e2efold', choices=['mxfold2', 'linearfold', 'e2efold'])
    parser.add_argument('--model-type', default=None, type=str)
    parser.add_argument('--seed', default=None, type=int)
    parser.add_argument('--device', default='gpu', type=str)
    parser.add_argument('--verbose', action='store_true',
                         help='enable verbose outputs for debugging')
    
    # train
    parser.add_argument('--lr', default=1e-3, type=float)
    parser.add_argument('--epoch', default=20, type=int)
    parser.add_argument('--batch-size', default=4, type=int)
    parser.add_argument('--num-workers', default=0, type=int)
    parser.add_argument('--exp', default='', type=str)
    parser.add_argument('--log-dir', default='./exp/', type=str)
    parser.add_argument('--data-root', default='./data/', type=str)
    parser.add_argument('--train-set', default='valid_train.lst', type=str)
    
    # Currently, regularization only works on mxfold2.
    parser.add_argument('--l1-weight', type=float, default=0.,
                        help='the weight for L1 regularization (default: 0)')
    parser.add_argument('--l2-weight', type=float, default=0.,
                        help='the weight for L2 regularization (default: 0)')
    # predict
    parser.add_argument('--predict_epoch', default=0, type=int)
    
    # wandb
    parser.add_argument('--wandb', default=False, action='store_true')
    parser.add_argument('--wandb_project', default='AI_Project_2', type=str)
    parser.add_argument('--wandb_name', default=None, type=str)
    parser.add_argument('--wandb_group', default=None, type=str)
    parser.add_argument('--wandb_job', default=None, type=str)
    parser.add_argument('--wandb_key', default=None, type=str)
    
    # model specific
    # e2efold    
    parser.add_argument('--seq-max-len', default=600, type=int)
    parser.add_argument('--u-net-d', default=10, type=int)
    
    # mxfold2
    parser.add_argument('--loss-func', choices=('hinge', 'hinge_mix'), default='hinge',
                        help="loss fuction ('hinge', 'hinge_mix') ")
    parser.add_argument('--loss-pos-paired', type=float, default=0.5,
                        help='the penalty for positive base-pairs for loss augmentation (default: 0.5)')
    parser.add_argument('--loss-neg-paired', type=float, default=0.005,
                        help='the penalty for negative base-pairs for loss augmentation (default: 0.005)')
    parser.add_argument('--loss-pos-unpaired', type=float, default=0,
                        help='the penalty for positive unpaired bases for loss augmentation (default: 0)')
    parser.add_argument('--loss-neg-unpaired', type=float, default=0,
                        help='the penalty for negative unpaired bases for loss augmentation (default: 0)')

    parser.add_argument('--max-helix-length', type=int, default=30, 
                    help='the maximum length of helices (default: 30)')
    parser.add_argument('--embed-size', type=int, default=0,
                    help='the dimention of embedding (default: 0 == onehot)')
    parser.add_argument('--num-filters', type=int, action='append',
                    help='the number of CNN filters (default: 96)')
    parser.add_argument('--filter-size', type=int, action='append',
                    help='the length of each filter of CNN (default: 5)')
    parser.add_argument('--pool-size', type=int, action='append',
                    help='the width of the max-pooling layer of CNN (default: 1)')
    parser.add_argument('--dilation', type=int, default=0, 
                    help='Use the dilated convolution (default: 0)')
    parser.add_argument('--num-lstm-layers', type=int, default=0,
                    help='the number of the LSTM hidden layers (default: 0)')
    parser.add_argument('--num-lstm-units', type=int, default=0,
                    help='the number of the LSTM hidden units (default: 0)')
    parser.add_argument('--num-transformer-layers', type=int, default=0,
                    help='the number of the transformer layers (default: 0)')
    parser.add_argument('--num-transformer-hidden-units', type=int, default=2048,
                    help='the number of the hidden units of each transformer layer (default: 2048)')
    parser.add_argument('--num-transformer-att', type=int, default=8,
                    help='the number of the attention heads of each transformer layer (default: 8)')
    parser.add_argument('--num-paired-filters', type=int, action='append', default=[],
                    help='the number of CNN filters (default: 96)')
    parser.add_argument('--paired-filter-size', type=int, action='append', default=[],
                    help='the length of each filter of CNN (default: 5)')
    parser.add_argument('--num-hidden-units', type=int, action='append',
                    help='the number of the hidden units of full connected layers (default: 32)')
    parser.add_argument('--dropout-rate', type=float, default=0.0,
                    help='dropout rate of the CNN and LSTM units (default: 0.0)')
    parser.add_argument('--fc-dropout-rate', type=float, default=0.0,
                    help='dropout rate of the hidden units (default: 0.0)')
    parser.add_argument('--num-att', type=int, default=0,
                    help='the number of the heads of attention (default: 0)')
    parser.add_argument('--pair-join', choices=('cat', 'add', 'mul', 'bilinear'), default='cat', 
                        help="how pairs of vectors are joined ('cat', 'add', 'mul', 'bilinear') (default: 'cat')")
    parser.add_argument('--no-split-lr', default=False, action='store_true')
    

   
    
    args = parser.parse_args(args)
    
    return args