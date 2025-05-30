
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm


def estimate_gamma(X, Y=None, quant=0.5):
    if Y is None:
        Y = X
    
    dists = torch.cdist(X, Y, p=2)
    i, j = torch.triu_indices(*dists.shape, offset=1)
    return 1 / np.quantile(dists[i,j].flatten().numpy(force=True), q=quant)

def rbf_kernel(X, Y=None, gamma=None):
    if Y is None:
        Y = X  
    
    if gamma is None:
        gamma = 1.0 / X.shape[1] 

    X_norm = (X ** 2).sum(dim=1, keepdim=True)
    Y_norm = (Y ** 2).sum(dim=1, keepdim=True)
    distances = X_norm - 2 * torch.mm(X, Y.T) + Y_norm.T

    kernel = torch.exp(-gamma * distances)
    return kernel



class Profiler(object):
    def __init__(self, args):
        super(Profiler, self).__init__()
        self.args = args

    def get_embeddings(self, args, model, dataset):
        
        dataloader = DataLoader(dataset, batch_size=args.inference_batch_size, shuffle=False)

        print('INFO: Extracting Embeddings!')
        emb_dict = {}
        for batch in tqdm(dataloader):
            inp = model.tokenizer(batch['input'], padding=True, return_tensors='pt')
            inp['length'] = torch.Tensor(inp['attention_mask'].sum(1))
            
            hidden_states = model(inp, output_hidden_states=True, hidden_states_layers_to_output=[-1])[0]
            
            for lidx in range(len(hidden_states)-1, 0, -1) :
                embs = hidden_states[lidx][:,-1,:].float()
                
                if torch.isnan(embs).any().item(): print(f"WARNING: NaN on layer {lidx}")
                if torch.isinf(embs).any().item(): print(f"WARNING: inf on layer {lidx}")
                
                embs = embs / (embs.shape[1]**0.5) # to prevent overflow
                    
                if lidx not in emb_dict.keys():
                    emb_dict[lidx] = [embs]
                else :
                    emb_dict[lidx] = emb_dict[lidx] + [embs]

        for lidx, embs in emb_dict.items():
            emb_dict[lidx] = torch.cat(embs)

        return emb_dict


    def profile(self, args, model, dataset, pre_embs, post_embs):
        
        # for lidx in tqdm(pre_embs.keys()):
        #     print(lidx)
        # Prepare Embeddings and Kernels 
        pre_emb = pre_embs[32]
        if args.cpu_profiler:
            pre_emb = pre_emb.cpu()
        pre_emb_normed = pre_emb / torch.norm(pre_emb, dim=1, keepdim=True)

        post_emb = post_embs[32]
        if args.cpu_profiler:
            post_emb = post_emb.cpu()
        post_emb_normed = post_emb / torch.norm(post_emb, dim=1, keepdim=True)
        
        if args.gamma is None :
            gamma1, gamma2 = estimate_gamma(pre_emb_normed), estimate_gamma(post_emb_normed)
        else :
            gamma1 = gamma2 = args.gamma
        
        pre_K = rbf_kernel(pre_emb_normed, gamma=gamma1)
        post_K = rbf_kernel(post_emb_normed, gamma=gamma2)

        # Compute Kernel Divergence Score
        score = -F.kl_div(post_K.log(), pre_K, reduction='none').abs().sum() / pre_K.sum()**0.5
        kernel_divergence_score = score.item()


        ###### Save
        if args.answer_level_shuffling :
            prefix = f"{args.timestamp}\t{args.model}_{args.data}_{args.sub_data}{args.target_num}_{args.split}_{args.contamination}_ALS{args.perturbation}"
        else :
            prefix = f"{args.timestamp}\t{args.model}_{args.data}_{args.sub_data}{args.target_num}_{args.split}_{args.contamination}"
        
        if args.gamma is not None :
            prefix = prefix + f"_gamma={args.gamma}"
        if args.epochs != 1 :
            prefix = prefix + f"_epoch={args.epochs}"
        if args.sgd :
            prefix = prefix + "_sgd"

        with open('out/results.tsv', 'a') as f:
            line = f"{prefix}_seed{args.seed}_KDS\t{kernel_divergence_score}\t{str(args)}\n"
            f.writelines(line)

