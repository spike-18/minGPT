import torch

def tns_to_str(lst: torch.tensor):
    assert len(lst.shape) == 1
    lst = lst.tolist()
    out = ''.join(map(str, lst))
    return out    

def str_to_tns(inp: str, num_digits, block_size):
    assert inp.isnumeric
    assert len(inp) <= block_size
    
    out = torch.tensor(list(map(int, inp)))
    for i in out:
        assert 0 <= i < num_digits
    
    return out

def chech_for_pal(inp: torch.tensor):
    sz = len(inp)
    return torch.all(inp[:sz//2] == inp[ sz//2 + sz%2 : sz].flipud())

