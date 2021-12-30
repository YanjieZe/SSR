# > 
# UUUAUCCCUCGAAUAUUGAUCGAAUAUGUAUGAAAUAUGAUGGUUAAAUGUUUCAUUCACAUUCGGUCAACGUUCGAGUGAUAAA
# .(((((.((((((..((((((((((.((.(((((((.............))))))).)).))))))))))..)))))).))))).

from random import randint
import torch
import torch.utils.data as data
import torch.optim as optim
import numpy as np
import utils
import dataset
import time
from tqdm import tqdm
from arguments import parse_args
from loss import Loss
import os

def benchmark(args):
    device = torch.device('cuda:0' if args.device == 'gpu' and torch.cuda.is_available() else 'cpu')
    print(f'device: {device}, CUDA Available: {torch.cuda.is_available()}')
    args.train_set = "inference_data.lst"
    print("testing on", args.train_set)
    valid_test_dataset = dataset.NovaDataset(args)
    valid_test_loader = data.DataLoader(dataset=valid_test_dataset,
                                   batch_size=args.batch_size,
                                   shuffle=True,
                                   num_workers=args.num_workers,
                                   pin_memory=True,
                                   drop_last=True,
                                   collate_fn=dataset.collate_fn_map[args.model]
                                   )
    
    
    if (args.model != 'linearfold'):
        #! Please manually set this path!
        save_path = os.path.join(args.log_dir, args.model+'_'+args.exp, str(args.epoch)+'.pth')
        model = utils.build_model(args)
        if (not torch.cuda.is_available()) or args.device == 'cpu':
            model.load_state_dict(torch.load(save_path, map_location=torch.device('cpu')))
        else:
            try:
                model.load_state_dict(torch.load(save_path, map_location=torch.device('gpu')))
            except:
                print("Model loading fails. Use the initial model.")
        model.eval()
    else:
        import models.linearfold as model
    
    print("Benchmarking on archii data...")
    p_sum = 0
    r_sum = 0
    f1_sum = 0

    loop = tqdm(valid_test_loader)
    start_time = time.time()
    for idx, (x, y_gt) in enumerate(loop):
        ps, rs, f1s = evaluate(model, args, x, y_gt)
        p_sum += np.sum(ps)
        r_sum += np.sum(rs)
        f1_sum += np.sum(f1s)
        
        loop.set_description(f'{args.model}, archii \
                             Avg. PPV {p_sum/(idx+1)} | Avg. SEN {r_sum/(idx+1)} | Avg. F1 {f1_sum/(idx+1)}')
    end_time = time.time()
    print("time cost:", end_time-start_time)
        
            #! todo: e2efold rewrite
def evaluate(model, args, x, y):
    if (args.model == 'e2efold'):
        (PE_batch, seq_embedding_batch, state_pad, contact_masks) = x
        contacts_batch = y
        pred_contacts = model(PE_batch, seq_embedding_batch, state_pad)
        
        result_tuple_list = list(map(lambda i: evaluate_e2e(pred_contacts.cpu()[i], 
        contacts_batch.cpu()[i]), range(contacts_batch.shape[0])))
        
        ps, rs, f1s = zip(*result_tuple_list)
        return ps, rs, f1s
    
    elif (args.model == 'mxfold2'):
        seqs = x
        pairs = y
        scs, preds, bps = model(seqs)
        result_tuple_list = list(map(lambda i: evaluate_mx2(pairs[i], 
        bps[i]), range(len(bps))))
        ps, rs, f1s = zip(*result_tuple_list)
        print('--------------------------------')
        print(seqs[0], utils.pairs2dot_bracket(bps[0]))
        print('--------------------------------')
        return ps, rs, f1s
    
    elif (args.model == 'linearfold'):
        seqs = x
        pairs = y
        bps = []
        for seq in seqs:
            bp = model.predict(seqs[0])
            bps.append(bp)
        assert (bps[0].shape == pairs[0].shape), f"{bps[0].shape}, {pairs[0].shape}"
        # print("seqs[0]", seqs[0])
        # print("bps[0]", bps[0])
        # print("pairs[0]", pairs[0].numpy())
        result_tuple_list = list(map(lambda i: evaluate_mx2(pairs[i], 
        bps[i]), range(len(bps))))
        ps, rs, f1s = zip(*result_tuple_list)
        print('--------------------------------')
        print(seqs[0], utils.pairs2dot_bracket(bps[0]))
        print('--------------------------------')
        return ps, rs, f1s

def evaluate_e2e(pred_a, true_a):
    tp_map = torch.sign(torch.Tensor(pred_a)*torch.Tensor(true_a))
    tp = tp_map.sum()   #* true positive
    pred_p = torch.sign(torch.Tensor(pred_a)).sum() #* predicted positive (TP + FP)
    true_p = true_a.sum()   #* actual positive (TP + FN)
    fp = pred_p - tp
    fn = true_p - tp
    #? True Positive, True Negative, False Positive, False Negative
    
    p = tp / (tp + fp + 1e-9)          #* Precision / PPV
    r = tp / (tp + fn + 1e-9)          #* Recall / TPR / Sensitivity
    f1 = 2 * p * r / (p + r + 1e-9)    #* F1
    
    return p, r, f1


def evaluate_mx2(ref, pred):
    L = len(ref) - 1
    tp = fp = fn = 0
    if ((len(ref)>0 and isinstance(ref[0], list)) or (isinstance(ref, torch.Tensor) and ref.ndim==2)):
        if isinstance(ref, torch.Tensor):
            ref = ref.tolist()
        ref = {(min(i, j), max(i, j)) for i, j in ref}
        pred = {(i, j) for i, j in enumerate(pred) if i < j}
        tp = len(ref & pred)
        fp = len(pred - ref)
        fn = len(ref - pred)
    else:
        assert(len(ref) == len(pred))
        for i, (j1, j2) in enumerate(zip(ref, pred)):
            if j1 > 0 and i < j1: # pos
                if j1 == j2:
                    tp += 1
                elif j2 > 0 and i < j2:
                    fp += 1
                    fn += 1
                else:
                    fn += 1
            elif j2 > 0 and i < j2:
                fp += 1
    tn = L * (L - 1) // 2 - tp - fp - fn
    #? True Positive, True Negative, False Positive, False Negative
    
    p = tp / (tp + fp + 1e-9)          #* Precision / PPV
    r = tp / (tp + fn + 1e-9)          #* Recall / TPR / Sensitivity
    f1 = 2 * p * r / (p + r + 1e-9)    #* F1
    
    return p, r, f1
        
if __name__ == '__main__':
    args = parse_args()
    print(args)
    benchmark(args)
    

