import torch

def tns_to_str(lst: torch.tensor):
    assert len(lst.shape) == 1
    lst = lst.tolist()
    out = ''.join(map(str, lst))
    return out    

def str_to_tns(inp: str):
    assert inp.isnumeric
    out = torch.tensor(list(map(int, inp)), dtype=torch.long)
    return out