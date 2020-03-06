import torch
from torch import Tensor

def srcsTensor(names: list, max_name_len: int, allowed_letters: list):
    """
    Turn a list of name strings into a tensor of one-hot letter vectors
    of shape: <max_name_len x len(names) x n_letters>

    All names are padded with '<pad_character>' such that they have the length: desired_len
    names: List of names to converted to a one-hot-encded vector
    max_name_len: The max name length allowed
    """
    tensor = torch.zeros(max_name_len, len(names), len(allowed_letters))
    for i_name, name in enumerate(names):
        for i_char, letter in enumerate(name):
            tensor[i_char][i_name][allowed_letters.index(letter)] = 1
    return tensor


def trgsTensor(formats: list):
    batch_sz = len(formats)
    ret = torch.zeros(1, batch_sz).type(torch.LongTensor)
    
    for i in range(batch_sz):
        ret[0][i] = int(formats[i])
    
    return ret