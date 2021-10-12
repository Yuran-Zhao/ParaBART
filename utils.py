import torch
import torch.nn as nn
from torch.utils.data import Dataset
import os, errno
import numpy as np
from datetime import datetime
import pdb
from numpy.lib.index_tricks import IndexExpression
import json

import glob


def make_path(path):
    try:
        os.makedirs(path)
    except OSError as exc:
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


class Timer:
    def __init__(self):
        self.start_time = datetime.now()
        self.last_time = self.start_time

    def get_time_from_last(self, update=True):
        now_time = datetime.now()
        diff_time = now_time - self.last_time
        if update:
            self.last_time = now_time
        return diff_time.total_seconds()

    def get_time_from_start(self, update=True):
        now_time = datetime.now()
        diff_time = now_time - self.start_time
        if update:
            self.last_time = now_time
        return diff_time.total_seconds()


def is_paren(tok):
    return tok == ")" or tok == "("


def deleaf(tree):
    # pdb.set_trace()
    tree = tree.decode('utf-8')
    nonleaves = ''
    # for w in tree.replace('\n', '').split():
    for w in tree.split():
        w = w.replace('(', '( ').replace(')', ' )')
        nonleaves += w + ' '

    arr = nonleaves.split()
    for n, i in enumerate(arr):
        if n + 1 < len(arr):
            tok1 = arr[n]
            tok2 = arr[n + 1]
            if not is_paren(tok1) and not is_paren(tok2):
                arr[n + 1] = ""

    nonleaves = " ".join(arr)
    return nonleaves.split()


def last_checkpoint(path):
    names = glob.glob(os.path.join(path, "*.pt"))
    if len(names) == 0:
        return None, None
    oldest_counter = 0
    checkpoint_name = names[0]
    for name in names:
        counter = name.rstrip(".pt").split("epoch")[-1]
        if not counter.isdigit():
            continue
        else:
            counter = int(counter)

        if counter > oldest_counter:
            checkpoint_name = name
            oldest_counter = counter

    return checkpoint_name, oldest_counter
