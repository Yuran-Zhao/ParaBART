from __future__ import absolute_import, division, unicode_literals
import json
import os
import sys
import numpy as np
import logging
import pickle
import torch
import argparse
from transformers import BartTokenizer, BartConfig, BartModel
import pdb

parser = argparse.ArgumentParser()
parser.add_argument('--model_dir', type=str, default="./model/")
parser.add_argument('--model_name', type=str, default="model_epoch10.pt")
parser.add_argument('--cache_dir', type=str, default="./bart-base/")
parser.add_argument('--senteval_dir', type=str, default="../SentEval/")
# parser.add_argument('--structural_parabart', type=bool, default=False)
# parser.add_argument('--new_decoder', type=bool, default=False)
parser.add_argument('--model_type', type=str, default='baseline')
parser.add_argument('--syntax_encoder_layer_num', type=int, default=1)
parser.add_argument('--adv_rank', type=int, default=64)
parser.add_argument(
    '--beta',
    type=float,
    default=0.8,
    help=
    "Beta used to balance the mixed attention in decoder. beta * sem + ( 1 - beta) * syn"
)
parser.add_argument(
    '--use_GAT',
    action='store_true',
    help="Use Graph Attention Networks as syntax encoder or not.")
args = parser.parse_args()

# import SentEval
sys.path.insert(0, args.senteval_dir)
import senteval

sys.path.insert(0, args.model_dir)
if args.model_type == 'structural':
    from structural_parabart import StructuralParaBart as ParaBart
elif args.model_type == 'new_decoder':
    from new_decoder_parabart import NewDecoderParaBart as ParaBart
elif args.model_type == 'mixed_attention':
    from sem_extractor import SemExtractor as ParaBart
else:
    from parabart import ParaBart

# if args.structural_parabart:
# elif args.new_decoder:
# else:


# SentEval prepare and batcher
def prepare(params, samples):
    pass


def batcher(params, batch):
    batch = [' '.join(sent) if sent != [] else '.' for sent in batch]
    embeddings = build_embeddings(embed_model, tokenizer, batch)
    return embeddings


def build_embeddings(model, tokenizer, sents):
    model.eval()
    embeddings = torch.ones((len(sents), model.config.d_model))
    with torch.no_grad():
        for i, sent in enumerate(sents):
            sent_inputs = tokenizer(sent, return_tensors="pt")
            sent_token_ids = sent_inputs['input_ids']

            sent_embed = model.encoder.embed(sent_token_ids.cuda())
            embeddings[i] = sent_embed.detach().cpu().clone()
    return embeddings


print("==== loading model ====")
config = BartConfig.from_pretrained('facebook/bart-base',
                                    cache_dir=args.cache_dir)
config.syntax_encoder_layer_num = args.syntax_encoder_layer_num
config.rank = args.adv_rank
config.use_GAT = args.use_GAT
config.beta = args.beta
# if args.structural_parabart:
#     embed_model = StructuralParaBart(config)
# else:
#     embed_model = ParaBart(config)

embed_model = ParaBart(config)

# pdb.set_trace()

tokenizer = BartTokenizer.from_pretrained('facebook/bart-base',
                                          cache_dir=args.cache_dir)

embed_model.load_state_dict(
    torch.load(os.path.join(args.model_dir, args.model_name),
               map_location='cpu'))

embed_model = embed_model.cuda()

# Set params for SentEval
params = {
    'task_path': os.path.join(args.senteval_dir, 'data'),
    'usepytorch': True,
    'kfold': 10
}
params['classifier'] = {
    'nhid': 50,
    'optim': 'adam',
    'batch_size': 64,
    'tenacity': 5,
    'epoch_size': 4
}

# Set up logger
logging.basicConfig(format='%(asctime)s : %(message)s', level=logging.DEBUG)

if __name__ == "__main__":
    se = senteval.engine.SE(params, batcher, prepare)

    transfer_tasks = [
        'STS12', 'STS13', 'STS14', 'STS15', 'STS16', 'STSBenchmark',
        'BigramShift', 'Depth', 'TopConstituents'
    ]

    results = se.eval(transfer_tasks)
    print(results)
