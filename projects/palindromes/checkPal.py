import os
import sys

import torch
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader

from mingpt.model import GPT
from mingpt.trainer import Trainer
from mingpt.utils import set_seed, setup_logging, CfgNode as CN

from utils import tns_to_str

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
    C.model.model_type = 'gpt-mini'
    C.model.num_classes = 2

    # trainer
    C.trainer = Trainer.get_default_config()
    C.trainer.learning_rate = 5e-4 # the model we're using is so small that we can go a bit faster
    C.train_max_batches = 10
    C.trainer.max_iters = 3000

    return C

# -----------------------------------------------------------------------------

class PalindromeDataset(Dataset):
    # TODO: implement variable input length
    """ 
    Dataset for palindrome detection. E.g. for problem length 6:
    Input: 1 2 3 2 1 -> Output: 1
    Input: 1 2 3 3 3 -> Output: 0
    Which will feed into the transformer concatenated as:
    input:  1 2 3 2 1 1
    output: I I I I I 1
    where I is "ignore", as the transformer is reading the input sequence
    """
    
    @staticmethod
    def get_default_config():
        C = CN()
        C.num_digits = 10
        C.length = 6
        return C

    def __init__(self, split, length=6, num_digits=3):
        assert split in {'train', 'test'}
        self.split = split
        self.length = length
        self.num_digits = num_digits
        self.num_classes = 2
    
    def __len__(self):
        return 10000 # leave for now
    
    def get_vocab_size(self):
        return self.num_digits
    
    def get_block_size(self):
        # the length of the sequence that will feed into transformer, 
        # containing concatenated input and the output, but -1 because
        # the transformer starts making predictions at the last input element
        return self.length + 1

    def __getitem__(self, idx):
        
        # use rejection sampling to generate an input example from the desired split
        while True:
            # generate some random integers
            inp = torch.randint(self.num_digits, size=(self.length,), dtype=torch.long)
            # half of the time generate palindrome
            if torch.rand(1).item() < 0.5:
                inp[:self.length//2] = inp[ self.length//2 + self.length%2 :].flipud()
            # figure out if this generated example is train or test based on its hash
            h = hash(inp.__str__())
            inp_split = 'test' if h % 4 == 0 else 'train' # designate 25% of examples as test
            if inp_split == self.split:
                break # ok
        
        # solve the task:
        sol = torch.all(inp[:self.length//2] == inp[ self.length//2 + self.length%2 :].flipud(), keepdim=True)

        # concatenate the problem specification and the solution
        cat = torch.cat((inp, sol), dim=0)

        # the inputs to the transformer will be the offset sequence
        x = cat[:-1].clone()
        y = cat[1:].clone()
        # we only want to predict at output locations, mask out the loss at the input locations
        y[:self.length-1] = -1
        return x, y
    
    
if __name__ == '__main__':
    
    # get default config and overrides from the command line, if any
    config = get_config()
    config.merge_from_args(sys.argv[1:])
    set_seed(config.system.seed)
    
    # construct train and test datasets
    train_dataset = PalindromeDataset(split='train',  length=config.data.length, num_digits=config.data.num_digits)
    test_dataset  = PalindromeDataset(split='test', length=config.data.length, num_digits=config.data.num_digits)
    
    # print(train_dataset[0])
    
    # construct the model
    config.model.vocab_size = train_dataset.get_vocab_size()
    config.model.block_size = train_dataset.get_block_size()
    print(config)
    setup_logging(config)

    model = GPT(config.model)

    # construct the trainer object
    trainer = Trainer(config.trainer, model, train_dataset)

    # helper function for the evaluation of a model
    def eval_split(trainer, split, max_batches=None):
        dataset = {'train':train_dataset, 'test':test_dataset}[split]
        results = []
        mistakes_printed_already = 0
        loader = DataLoader(dataset, batch_size=100, num_workers=0, drop_last=False)
        for b, (x, y) in enumerate(loader):
            x = x.to(trainer.device)
            # let the model sample the rest of the sequence
            out = model.generate(x, 1, do_sample=False) # using greedy argmax, not sampling
            # isolate the last digit of the sampled sequence
            pred = out[:, -1:].to('cpu')
            # evaluate the correctness of the results in this batch
            gt = y[:, -1:]
            correct = (pred == gt).cpu() # Software 1.0 vs. Software 2.0 fight RIGHT on this line haha
            for i in range(x.size(0)):
                results.append(int(correct[i]))
                if not correct[i] and mistakes_printed_already < 5: # only print up to 5 mistakes to get a sense
                    mistakes_printed_already += 1
                    print("GPT claims that %s is %spalindrome" % (tns_to_str(x[i]), 'not ' if pred[i] == 0 else ''))
            if max_batches is not None and b+1 >= max_batches:
                break
        rt = torch.tensor(results, dtype=torch.float)
        print("%s final score: %d/%d = %.2f%% correct" % (split, rt.sum(), len(results), 100*rt.mean()))
        return rt.sum()

    # iteration callback
    top_score = 0
    def batch_end_callback(trainer):
        global top_score

        if trainer.iter_num % 10 == 0:
            print(f"iter_dt {trainer.iter_dt * 1000:.2f}ms; iter {trainer.iter_num}: train loss {trainer.loss.item():.5f}")

        if trainer.iter_num % 500 == 0:
            # evaluate both the train and test score
            train_max_batches = config.train_max_batches
            model.eval()
            with torch.no_grad():
                train_score = eval_split(trainer, 'train', max_batches=train_max_batches)
                test_score  = eval_split(trainer, 'test',  max_batches=None)
            
            score = train_score + test_score
            # save the model if this is the best score we've seen so far
            if score > top_score:
                top_score = score
                print(f"saving model with new top score of {score}")
                ckpt_path = os.path.join(config.system.work_dir, "model.pt")
                torch.save(model.state_dict(), ckpt_path)
            
            model.train()

    trainer.set_callback('on_batch_end', batch_end_callback)

    # run the optimization
    trainer.run()
