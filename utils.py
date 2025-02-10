
import numpy as np
import pandas as pd
import torch 
import selfies as sf
from rdkit.Chem.rdmolfiles import MolFromSmiles, MolToSmiles
from rdkit import Chem
from rdkit.Chem import QED
import importlib
from tqdm.notebook import tqdm
from tqdm.auto import trange
import sys
from torch.utils.data import Dataset, DataLoader
import re

def tensor2smiles_sampling(x, word2idx, idx2word):
    out_smiles = []
    test_selfies = []
    for src in x:
        probs = torch.softmax(src, dim=-1)
        sampled_indices = torch.multinomial(probs, 1).reshape(-1).cpu().detach().numpy()

        first_eos = np.where(sampled_indices == word2idx['<eos>'])[0]
        if len(first_eos) > 0:
            sampled_indices = sampled_indices[:first_eos[0]]

        test_selfies.append([idx2word[i] for i in sampled_indices])

        sampled_indices = np.delete(sampled_indices, np.where(sampled_indices == word2idx['<pad>']))
        sampled_indices = np.delete(sampled_indices, np.where(sampled_indices == word2idx['<sos>']))
        sampled_indices = np.delete(sampled_indices, np.where(sampled_indices == word2idx['<unk>']))

        out_sentence = [idx2word[i] for i in sampled_indices]
        out_sf = ''.join(out_sentence)
        out_smi = sf.decoder(out_sf)
        out_smiles.append(out_smi)

    return out_smiles, test_selfies

def tensor2smiles_v2(x, word2idx, idx2word):
    out_smiles = []
    out_selfies = []
    
    for src in x:
        _, sentence = torch.max(src, dim=-1)
        sentence_np = sentence.reshape(-1).cpu().detach().numpy()
        first_eos = np.where(sentence_np == word2idx['<eos>'])[0] 
        if len(first_eos) > 0:
            sentence_np = sentence_np[:first_eos[0]]
        sentence_np = np.delete(sentence_np, np.where((sentence_np == word2idx['<pad>'])))
        sentence_np = np.delete(sentence_np, np.where((sentence_np == word2idx['<sos>'])))
        sentence_np = np.delete(sentence_np, np.where((sentence_np == word2idx['<unk>'])))
        out_sentence = [idx2word[i] for i in sentence_np]
        out_sf = ''.join(out_sentence)
        out_smi = sf.decoder(out_sf)
        out_smiles.append(out_smi)
        out_selfies.append(out_sf)
    return out_smiles, out_selfies


def print_token2sf(target, out_x,
                    word2idx, idx2word,
                    batch_size, num, print_only = False):
    output_smiles = []
    if batch_size >= num:
        idx = np.random.randint(0, batch_size, num, dtype=int)
        in_x = target[idx]
        out_x = out_x[idx]
        
        label = []
        predict = []
    else:
        print("The number to display must not be greater than the batch size.")
        return
    for i in range(num):
        with torch.no_grad():
            _, sentence = torch.max(out_x[i], dim=-1)
            sentence_np = sentence.reshape(-1).cpu().detach().numpy()

            first_eos = np.where(sentence_np == word2idx['<eos>'])[0] 
            if len(first_eos) > 0:
                sentence_np = sentence_np[:first_eos[0]]

        src = in_x[i].cpu().numpy()
        src = np.delete(src, np.where((src == word2idx['<pad>'])))
        src = np.delete(src, np.where((src == word2idx['<eos>'])))
        in_sentence = [idx2word[i] for i in src]
        in_sf = ''.join(in_sentence)
        in_smi = sf.decoder(in_sf)
        in_n_smi = normalize_SMILES(in_smi)
        
        
        sentence_np = np.delete(sentence_np, np.where((sentence_np == word2idx['<pad>'])))
        sentence_np = np.delete(sentence_np, np.where((sentence_np == word2idx['<sos>'])))
        sentence_np = np.delete(sentence_np, np.where((sentence_np == word2idx['<unk>'])))
        out_sentence = [idx2word[i] for i in sentence_np]
        out_sf = ''.join(out_sentence)
        try:
            out_smi = sf.decoder(out_sf)
        except Exception as e:
            print("=== == == = = error = = == == ===")
            print(out_sf)
            print(e)
        out_n_smi = normalize_SMILES(out_smi)

        if print_only:
            print("x before : ", in_n_smi)
            print("x after : ", out_n_smi)
        else:
            print("x before : ", in_n_smi)
            print("x after : ", out_smi)
            if out_n_smi is None:
                print("Output X is unvalid SMILEs")
            else:
                print("Output X is Valid SMILEs")
                print("n_smi : ", out_n_smi)
        output_smiles.append(out_n_smi)
    return idx, output_smiles


def convert_time(seconds):
    if seconds < 1:
        return f'{round(seconds, 4)}s'
    else:
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        seconds = round(seconds % 60, 2) 

        return f"{hours}h {minutes}m {seconds}s"

def normalize_SMILES(smi):
    try:
        mol = MolFromSmiles(smi)
        smi_rdkit = MolToSmiles(
                        mol,
                        isomericSmiles=False,   # modified because this option allows special tokens (e.g. [125I])
                        kekuleSmiles=False,     # default
                        rootedAtAtom=-1,        # default
                        canonical=True,         # default
                        allBondsExplicit=False, # default
                        allHsExplicit=False     # default
                    )
    except:
        smi_rdkit = None
    return smi_rdkit



class selfiesDataset(Dataset):
    def __init__(self, data, prop_data, word2index, device, num_samples = None):
        self.data = data
        self.prop_data = prop_data
        self.is_pre = True if self.prop_data is None else False
        self.word2index = word2index
        self.device = device
        self.num_samples = num_samples if num_samples is not None else len(data)
        self.pattern = self.get_pattern()

    def get_pattern(self):
        excluded_keys = {'<pad>', '<unk>', '<sos>', '<eos>'}
        tokens = [re.escape(key) for key in self.word2index if key not in excluded_keys]
        return '|'.join(tokens)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        prop_list = []
        if self.is_pre:
            return self.data[idx], idx
        else:
            for prop in self.prop_data:
                prop_list.append(prop[idx])
            return self.data[idx], prop_list, idx

def collate_fn_pre(batch, word2index, pattern, device):
    data, origin_indexs = zip(*batch)
    tokenized_selfies = sf2token(data, pattern)
    sorted_source, sorted_target, sorted_lengths, max_len, sorted_idx = token2tensor_sf(tokenized_selfies, word2index, device)
    sorted_origin_indexs = [origin_indexs[i] for i in sorted_idx]
    return sorted_source, sorted_target, sorted_lengths, max_len, sorted_origin_indexs


def collate_fn(batch, word2index, pattern, device):
    data, props, origin_indexs = zip(*batch)
    tokenized_selfies = sf2token(data, pattern)
    sorted_source, sorted_target, sorted_lengths, max_len, sorted_idx = token2tensor_sf(tokenized_selfies, word2index, device)
    sorted_props = [[props[i][j] for i in sorted_idx] for j in range(len(props[0]))]
    sorted_origin_indexs = [origin_indexs[i] for i in sorted_idx]

    return sorted_source, sorted_target, sorted_lengths, max_len, sorted_props, sorted_origin_indexs


def sf2token(selfies, pattern):
    tokenized_data = []
    for smi in selfies:
        tokens = re.findall(pattern, smi)
        if smi == ''.join(tokens):
            tokenized_data.append(tokens)
        else:
            print(smi)
            print(tokens)
            print(''.join(tokens))
            print("Invalid Split Pattern, Check Vocab")
    return tokenized_data


def token2tensor_sf(tokenized_selfies, word2index, device):
    batch_size = len(tokenized_selfies)
    lines = [["<sos>"] + tokens + ["<eos>"] for tokens in tokenized_selfies]
    smi_length = np.array([len(line) for line in lines])
    max_len = np.max(smi_length)

    source = torch.LongTensor([
        [word2index.get(w, word2index["<unk>"]) for w in line[:-1]]
        + [word2index["<pad>"]] * (max_len - len(line))
        for line in lines]).to(device)

    target = torch.LongTensor([
        [word2index.get(w, word2index["<unk>"]) for w in line[1:]]
        + [word2index["<pad>"]] * (max_len - len(line))
        for line in lines]).to(device)

    sorted_lengths = torch.LongTensor([torch.max(source[i, :].data.nonzero()) + 1
                                       for i in range(batch_size)])

    sorted_lengths, sorted_idx = sorted_lengths.sort(0, descending=True)
    sorted_lengths = sorted_lengths.tolist()

    sorted_source = source[sorted_idx]
    sorted_target = target[sorted_idx]

    
    return sorted_source, sorted_target, sorted_lengths, max(sorted_lengths), sorted_idx




def normalize_values(values):
    return (values - np.min(values)) / (np.max(values) - np.min(values))