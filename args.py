import argparse
from  ECCT.Main import set_seed

import os
import torch
import logging
from datetime import datetime

def set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def pass_args():
    parser = argparse.ArgumentParser(description='PyTorch ECCT')
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--gpus', type=str, default='-1', help='gpus ids')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--test_batch_size', type=int, default=2048)
    parser.add_argument('--seed', type=int, default=42)
    
    # Code args
    parser.add_argument('--code_type', type=str, default='POLAR',
                        choices=['BCH', 'POLAR', 'LDPC', 'CCSDS', 'MACKAY'])
    parser.add_argument('--code_k', type=int, default=32)
    parser.add_argument('--code_n', type=int, default=64)
    parser.add_argument('--standardize', action='store_true')
    
    # Model args
    parser.add_argument('--N_dec', type=int, default=6)
    parser.add_argument('--d_model', type=int, default=32)
    parser.add_argument('--h', type=int, default=8)
    
    # This is a workaround to avoid taking arguments from Jupyter
    args = parser.parse_args(args=[])
    
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    set_seed(args.seed)

    class Code:
        pass

    code = Code()
    code.k = args.code_k
    code.n = args.code_n
    code.code_type = args.code_type
    # G, H = Get_Generator_and_Parity(code, standard_form=args.standardize)
    code.generator_matrix = None #torch.from_numpy(G).transpose(0, 1).long()
    code.pc_matrix = None #torch.from_numpy(H).long()
    args.code = code

    model_dir = os.path.join('Results_ECCT', f"{args.code_type}__Code_n_{args.code_n}_k_{args.code_k}__{datetime.now().strftime('%d_%m_%Y_%H_%M_%S')}")
    os.makedirs(model_dir, exist_ok=True)
    args.path = model_dir
    handlers = [logging.FileHandler(os.path.join(model_dir, 'logging.txt'))]
    handlers += [logging.StreamHandler()]
    logging.basicConfig(level=print, format='%(message)s', handlers=handlers)
    print(f"Path to model/logs: {model_dir}")
    print(args)

    return args
