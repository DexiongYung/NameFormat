import torch
import argparse
import string
import json
import matplotlib.pyplot as plt
import pandas as pd
import torch.nn as nn
from collections import OrderedDict
from Utilities.Convert import *
from Model.NameFormatModel import NameFormatModel
from Datasets.FormatDataset import FormatDataset
from torch.utils.data import DataLoader

parser = argparse.ArgumentParser()
parser.add_argument('--name', help='Name of the Session', nargs='?', default='name_format', type=str)
parser.add_argument('--hidden_size', help='Size of the hidden state tensors', nargs='?', default=256, type=int)
parser.add_argument('--num_layers', help='Number of layers', nargs='?', default=5, type=int)
parser.add_argument('--batch', help='Batch size', nargs='?', default=256, type=int)
parser.add_argument('--lr', help='Learning rate', nargs='?', default=0.005, type=float)
parser.add_argument('--epoch', help='Number of epochs', nargs='?', default=5000, type=int)
parser.add_argument('--train_file', help='File to train on', nargs='?', default='Data/FullNames.csv', type=str)
parser.add_argument('--name_col', help='Column header of data name', nargs='?', default='name', type=str)
parser.add_argument('--format_col', help='Column header of data format', nargs='?', default='format', type=str)
parser.add_argument('--print', help='Print every', nargs='?', default=100, type=int)
parser.add_argument('--continue_training', help='Boolean whether to continue training an existing model', nargs='?',
                    default=False, type=bool)

# Parse optional args from command line and save the configurations into a JSON file
args = parser.parse_args()
NAME = args.name
EPOCH = args.epoch
LR = args.lr
HIDDEN_SZ = args.hidden_size
NUM_LAYERS = args.num_layers
BATCH_SZ = args.batch
TRAIN_FILE = args.train_file
NAME_COL = args.name_col
FORMAT_COL = args.format_col
PRINTS = args.print
CLIP = 1

# Global Variables
SOS = '<SOS>'
PAD = '<PAD>'
EOS = '<EOS>'
INPUT = [char for char in string.printable]
INPUT.extend([SOS, PAD, EOS])
# Formats:
# 0 = first last
# 1 = first middle last
# 2 = last, first
# 3 = last, first middle
# 4 = first middle_initial. last
# 5 = last, first middle_initial.
FORMAT_REGEX = ["^(\w[A-Za-z'-]+\s\w[A-Za-z'-]+)$", "^(\w[A-Za-z'-]+\s\w[A-Za-z'-]+\s\w[A-Za-z'-]+)$",
                "^(\w[a-zA-Z-']+,\s\w[a-zA-Z-']+)$", "^(\w[a-zA-Z-']+,\s\w[a-zA-Z-']+\s\w[a-zA-Z-']+)$",
                "^(\w[A-Za-z'-]+\s[A-Z]+.\s\w[A-Za-z'-]+)$", "^(\w[A-Za-z'-]+,\s\w[A-Za-z'-]+\s[A-Z]+\.)$"]
OUTPUT = [num for num in range(len(FORMAT_REGEX))]
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train(x: list, trg: torch.Tensor):
    batch_sz = len(x)
    max_len = len(max(x, key=len))

    src_x = list(map(lambda s: [char for char in s] + ([PAD] * (max_len - len(s))), x))

    src = srcsTensor(src_x, max_len, INPUT).to(DEVICE)

    output = model.forward(src)
    loss = criterion(output, trg.to(DEVICE))
    loss.backward()
    
    for p in model.parameters():
        p.data.add_(-LR, p.grad.data)

    return loss.item()

def iter_train(dl: DataLoader, epochs: int = EPOCH, path: str = "Checkpoints/", print_every: int = PRINTS):
    all_losses = []
    total_loss = 0

    for e in range(1, EPOCH + 1):
        for x in dl:
            total_loss = train(x[0], x[1])

            if e % print_every == 0:
                all_losses.append(total_loss / print_every)
                total_loss = 0
                plot_losses(all_losses, x_label="Epochs", y_label="NLLosss", filename=NAME)
                torch.save(model.state_dict(), os.path.join(f"{path}{NAME}.path.tar"))

def load_json(jsonpath: str) -> dict:
    with open(jsonpath) as jsonfile:
        return json.load(jsonfile, object_pairs_hook=OrderedDict)

def save_json(jsonpath: str, content):
    with open(jsonpath, 'w') as jsonfile:
        json.dump(content, jsonfile)

def plot_losses(loss: list, x_label: str, y_label: str, folder: str = "Result", filename: str = None):
    x = list(range(len(loss)))
    plt.plot(x, loss, 'r--', label="Loss")
    plt.title("Losses")
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend(loc='upper left')
    plt.savefig(f"{folder}/{filename}")
    plt.close()

to_save = {
    'session_name': NAME,
    'hidden_size': HIDDEN_SZ,
    'num_layers': NUM_LAYERS,
    'input_size': len(INPUT),
    'ouput_size': len(OUTPUT), 
    'input': INPUT,
    'output': OUTPUT
}

save_json(f'Config/{NAME}.json', to_save)

file_obj = open(TRAIN_FILE, encoding='utf-8')
df = pd.read_csv(file_obj)
ds = FormatDataset(df, NAME_COL, FORMAT_COL)
dl = DataLoader(ds, batch_size=BATCH_SZ, shuffle=True)

model = NameFormatModel(DEVICE, len(INPUT), HIDDEN_SZ, len(OUTPUT), NUM_LAYERS)
model.to(DEVICE)
criterion = nn.NLLLoss()

iter_train(dl)