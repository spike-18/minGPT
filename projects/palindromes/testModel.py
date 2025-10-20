import sys
import json

import torch
from torch.nn import functional as F
from utils import str_to_tns, tns_to_str
from mingpt.model import GPT
from mingpt.utils import CfgNode as CN
from mingpt.trainer import Trainer
from checkPal import PalindromeDataset

# -----------------------------------------------------------------------------

def get_config():

    C = CN()

    # system
    C.system = CN()
    C.system.seed = 3407
    C.system.work_dir = './out/palindromes'

    # data
    C.data = PalindromeDataset.get_default_config()

    # model
    C.model = GPT.get_default_config()
    C.model.model_type = 'gpt-micro'
    C.model.num_classes = 2
    C.model.block_size = 50
    C.model.vocab_size = 10

    # trainer
    C.trainer = Trainer.get_default_config()
    C.trainer.learning_rate = 3e-4
    C.trainer.max_iters = 50000
    C.trainer.batch_size = 64
    C.val_max_batches = 50

    return C


# -----------------------------------------------------------------------------



if __name__ == '__main__':
    
    assert len(sys.argv) > 1
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_path = sys.argv[1]
    config = get_config()
    print("\n", config, "\n")

    print("Loading model from %s" % model_path)
    model = GPT(config.model).to(device)
    state_dict = torch.load(model_path, weights_only=True)
    model.load_state_dict(state_dict)
    
    model.eval()        
    inp = input("Type 'q' to exit.\nInput sequence of max %d digits: " % (config.model.block_size))
    
    while(inp != 'q'):
        inp = str_to_tns(inp, config.data.num_digits, config.model.block_size).to(device)
        
        inp = inp.view(1, -1)
        logits, _ = model(inp)
        
        logits = logits[:, -1, :]
        probs = F.softmax(logits, dim=-1)
        
        print('Model output:')
        prediction = torch.argmax(probs)
        print("%d (%spalindrome)" % (prediction, 'not ' if prediction == 0 else ''))
         
        inp = input("Input sequence of %d digits (q to exit): " % (config.model.block_size))