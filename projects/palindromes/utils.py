import torch

def tns_to_str(lst: torch.tensor):
    assert len(lst.shape) == 1
    lst = lst.tolist()
    while(lst[-1] == 0):
        lst = lst[:-1]
    lst[-1] = ''
    out = ''.join(map(str, lst))
    return out    

def str_to_tns(inp: str, num_digits, block_size):
    assert inp.isnumeric
    assert len(inp) < block_size
    
    head = torch.tensor(list(map(int, inp)))
    for i in head:
        assert 0 <= i < num_digits
    
    tail = torch.zeros(size=(block_size-len(head),), dtype=torch.long)
    out = torch.cat((head, tail), dim=-1)
    out[len(head)] = num_digits
    return out