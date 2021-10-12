import os, argparse, pickle, h5py
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import pdb

import gc

from utils import Timer, last_checkpoint, make_path, deleaf
from pprint import pprint
from tqdm import tqdm
from transformers import BartTokenizer, BartConfig, BartModel
from parabart import ParaBart
# from new_decoder_parabart import NewDecoderParaBart as ParaBart
from structural_parabart import StructuralParaBart
# from tokenizers import Tokenizer
# from tokenizers.models import WordLevel

import pickle
from loss import L1DistanceLoss, L1DepthLoss
from structural_dataset import StructuralDataset
from prepare_data import prepare_dataset, collate

import torch.utils.tensorboard as tensorboard

parser = argparse.ArgumentParser()
parser.add_argument('--model_dir', type=str, default="./model/")
parser.add_argument('--cache_dir', type=str, default="./bart-base/")
parser.add_argument('--data_dir', type=str, default="./data/")
parser.add_argument('--max_sent_len', type=int, default=40)
parser.add_argument('--max_synt_len', type=int, default=160)
parser.add_argument('--word_dropout', type=float, default=0.2)
parser.add_argument('--n_epoch', type=int, default=10)
parser.add_argument('--train_batch_size', type=int, default=64)
parser.add_argument('--accumulation_steps', type=int, default=1)
parser.add_argument('--valid_batch_size', type=int, default=16)
parser.add_argument('--lr', type=float, default=2e-5)
parser.add_argument('--adv_lr', type=float, default=1e-3)
parser.add_argument('--fast_lr', type=float, default=1e-4)
parser.add_argument('--weight_decay', type=float, default=1e-2)
parser.add_argument('--log_interval', type=int, default=1000)
parser.add_argument('--temp', type=float, default=0.5)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--mode',
                    type=str,
                    default="baseline",
                    help="Use which structure of model in the training stage.")
parser.add_argument(
    '--adv_pred_distance',
    action='store_true',
    help="Demand the discriminator to execute prediction on distance.")
parser.add_argument(
    '--adv_pred_depth',
    action='store_true',
    help="Demand the discriminator to execute prediction on depth.")
parser.add_argument(
    '--adv_pred_bow',
    action='store_true',
    help="Demand the discriminator to execute prediction on bag-of-word vector."
)
parser.add_argument(
    '--scheduler',
    action='store_true',
    help="Use some new scheduler for learning rate during training stage.")
parser.add_argument(
    '--adv_coefficient',
    type=float,
    default=0.001,
    help="The coefficient to balance adv_loss and generating_loss.")
parser.add_argument('--syntax_encoder_layer_num',
                    type=int,
                    default=1,
                    help="The number of layers of syntax encoder")
parser.add_argument('--adv_rank',
                    type=int,
                    default=64,
                    help="The internal dim of syntax discriminator.")
parser.add_argument(
    '--predict_distance_square',
    action='store_true',
    help=
    "When calculating the L1Distance Loss, the ground truth is the square of distance or not."
)
parser.add_argument(
    '--use_GAT',
    action='store_true',
    help="Use Graph Attention Networks as syntax encoder or not.")
parser.add_argument(
    '--use_linear_tree',
    action='store_true',
    help=
    "When `use_linear_tree` is set as True, `use_GAT` should be set as False, and vise versa."
)
parser.add_argument(
    '--use_freeze',
    action='store_true',
    help=
    "Freeze the parameters in the discriminator and other parameters in the model in turn."
)
parser.add_argument(
    '--similarity_as_weights',
    action='store_true',
    help="Use the similarity between sentence pairs as the weights of loss.")

# parser.add_argument('--adv_graph', type=bool, default=False)
args = parser.parse_args()
pprint(vars(args))
print()

writer = tensorboard.SummaryWriter(os.path.join(args.model_dir, 'run'))

# fix random seed
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
torch.backends.cudnn.deterministic = True


def structural_train(epoch, model, tokenizer, optimizer, args):
    timer = Timer()
    n_it = len(train_loader)
    optimizer.zero_grad()
    # pdb.set_trace()
    # for it, (sent1_token_ids, synt1_token_ids, sent1_target_role_ids,
    #          synt1_distances, synt1_depth, synt1_bow, graph1, sent2_token_ids,
    #          synt2_token_ids, sent2_target_role_ids, synt2_distances,
    #          synt2_depth, synt2_bow, graph2) in enumerate(train_loader):
    for it, (sent1_token_ids, synt1_token_ids, synt1_distances, synt1_depth,
             synt1_bow, graph1, sent2_token_ids, synt2_token_ids,
             synt2_distances, synt2_depth, synt2_bow, graph2,
             similarity) in enumerate(train_loader):
        total_loss = 0.0
        adv_total_loss = 0.0
        model.train()

        sent1_token_ids = sent1_token_ids.cuda()
        synt1_token_ids = synt1_token_ids.cuda()
        # sent1_target_role_ids = sent1_target_role_ids.cuda()
        synt1_distances = synt1_distances.cuda()
        synt1_depth = synt1_depth.cuda()
        synt1_bow = synt1_bow.cuda()

        sent2_token_ids = sent2_token_ids.cuda()
        synt2_token_ids = synt2_token_ids.cuda()
        # sent2_target_role_ids = sent2_target_role_ids.cuda()
        synt2_distances = synt2_distances.cuda()
        synt2_depth = synt2_depth.cuda()
        synt2_bow = synt2_bow.cuda()

        similarity = torch.tensor(similarity).cuda()
        similarity = nn.functional.softmax(similarity, dim=0)

        similarity = model.compute_similarity(sent1_token_ids, sent2_token_ids,
                                              similarity).cuda()

        if args.use_freeze:
            for p in model.parameters():
                p.requires_grad = False
            for p in model.adversary.parameters():
                p.requires_grad = True

        pred_dist, pred_depth, pred_bow = model.forward_adv(sent1_token_ids)
        true_dist, true_depth, true_bow = synt1_distances, synt1_depth, synt1_bow

        if args.similarity_as_weights and args.adv_pred_bow and (
                not args.adv_pred_distance) and (not args.adv_pred_depth):
            all_adv_loss = adv_criterion(pred_bow, true_bow)
            all_adv_loss = torch.mean(all_adv_loss, dim=-1)
            loss1 = torch.sum(all_adv_loss * similarity)
        else:
            if args.predict_distance_square:
                loss1 = adv_criterion(pred_dist, true_dist.pow(2), pred_depth,
                                      true_depth, pred_bow, true_bow)
            else:
                loss1 = adv_criterion(pred_dist, true_dist, pred_depth,
                                      true_depth, pred_bow, true_bow)

        loss1.backward()
        adv_total_loss += loss1.item()

        # pdb.set_trace()

        pred_dist, pred_depth, pred_bow = model.forward_adv(sent2_token_ids)
        true_dist, true_depth, true_bow = synt2_distances, synt2_depth, synt2_bow
        if args.similarity_as_weights and args.adv_pred_bow and (
                not args.adv_pred_distance) and (not args.adv_pred_depth):
            all_adv_loss = adv_criterion(pred_bow, true_bow)
            all_adv_loss = torch.mean(all_adv_loss, dim=-1)
            loss2 = torch.sum(all_adv_loss * similarity)
        else:
            if args.predict_distance_square:
                loss2 = adv_criterion(pred_dist, true_dist.pow(2), pred_depth,
                                      true_depth, pred_bow, true_bow)
            else:
                loss2 = adv_criterion(pred_dist, true_dist, pred_depth,
                                      true_depth, pred_bow, true_bow)

        loss2.backward()
        adv_total_loss += loss2.item()

        if (it + 1) % args.accumulation_steps == 0:
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if epoch > 1:
                adv_optimizer.step()
            adv_optimizer.zero_grad()

        if args.use_freeze:
            for p in model.parameters():
                p.requires_grad = True
            for p in model.adversary.parameters():
                p.requires_grad = False
        # optimize model
        # sent1->sent2 para & sent1 adv
        outputs, pred_dist, pred_depth, pred_bow = model(
            sent1_token_ids, synt2_token_ids, graph2, sent2_token_ids)
        batch_size, _, _ = outputs.shape
        targs = sent2_token_ids[:, 1:].contiguous().view(-1)
        outputs = outputs.contiguous().view(-1, outputs.size(-1))

        if args.similarity_as_weights:
            all_loss = para_criterion(outputs, targs)
            # pdb.set_trace()
            all_loss = all_loss.view(batch_size, -1).contiguous()
            element_num = torch.sum(all_loss != 0 + 0).item()
            all_loss = torch.sum(all_loss, dim=-1)
            loss = torch.sum(all_loss * similarity * batch_size / element_num)
        else:
            loss = para_criterion(outputs, targs)

        true_dist, true_depth, true_bow = synt1_distances, synt1_depth, synt1_bow

        if epoch > 1:
            if args.similarity_as_weights and args.adv_pred_bow and (
                    not args.adv_pred_distance) and (not args.adv_pred_depth):
                all_adv_loss = adv_criterion(pred_bow, true_bow)
                all_adv_loss = torch.mean(all_adv_loss, dim=-1)
                adv_loss = torch.sum(all_adv_loss * similarity)
            else:
                if args.predict_distance_square:
                    adv_loss = adv_criterion(pred_dist, true_dist.pow(2),
                                             pred_depth, true_depth, pred_bow,
                                             true_bow)
                else:
                    adv_loss = adv_criterion(pred_dist, true_dist, pred_depth,
                                             true_depth, pred_bow, true_bow)

            loss -= 0.1 * adv_loss

        loss.backward()
        total_loss += loss.item()

        # sent2->sent1 para & sent2 adv
        outputs, pred_dist, pred_depth, pred_bow = model(
            sent2_token_ids, synt1_token_ids, graph1, sent1_token_ids)
        batch_size, _, _ = outputs.shape
        targs = sent1_token_ids[:, 1:].contiguous().view(-1)
        outputs = outputs.contiguous().view(-1, outputs.size(-1))

        if args.similarity_as_weights:
            all_loss = para_criterion(outputs, targs)
            # pdb.set_trace()
            all_loss = all_loss.view(batch_size, -1).contiguous()
            element_num = torch.sum(all_loss != 0 + 0).item()
            all_loss = torch.sum(all_loss, dim=-1)
            loss = torch.sum(all_loss * similarity * batch_size / element_num)
        else:
            loss = para_criterion(outputs, targs)

        true_dist, true_depth, true_bow = synt2_distances, synt2_depth, synt2_bow

        if epoch > 1:
            if args.similarity_as_weights and args.adv_pred_bow and (
                    not args.adv_pred_distance) and (not args.adv_pred_depth):
                all_adv_loss = adv_criterion(pred_bow, true_bow)
                all_adv_loss = torch.mean(all_adv_loss, dim=-1)
                adv_loss = torch.sum(all_adv_loss * similarity)
            else:
                if args.predict_distance_square:
                    adv_loss = adv_criterion(pred_dist, true_dist.pow(2),
                                             pred_depth, true_depth, pred_bow,
                                             true_bow)
                else:
                    adv_loss = adv_criterion(pred_dist, true_dist, pred_depth,
                                             true_depth, pred_bow, true_bow)

            loss -= 0.1 * adv_loss

        loss.backward()
        total_loss += loss.item()

        if (it + 1) % args.accumulation_steps == 0:
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()

        if (it + 1) % args.log_interval == 0 or it == 0:
            para_1_2_loss, para_2_1_loss, adv_1_loss, adv_2_loss = structural_evaluate(
                model, tokenizer, args)
            valid_loss = para_1_2_loss + para_2_1_loss - adv_1_loss - adv_2_loss
            if args.scheduler:
                adv_scheduler.step((adv_1_loss + adv_2_loss))
            print(
                "| ep {:2d}/{} | it {:3d}/{} | {:5.2f} s | adv loss {:.4f} | loss {:.4f} | para 1-2 loss {:.4f} | para 2-1 loss {:.4f} | adv 1 loss {:.4f} | adv 2 loss {:.4f} | valid loss {:.4f} |"
                .format(epoch, args.n_epoch, it + 1, n_it,
                        timer.get_time_from_last(), adv_total_loss, total_loss,
                        para_1_2_loss, para_2_1_loss, adv_1_loss, adv_2_loss,
                        valid_loss))
            writer.add_scalar('Loss/train', total_loss,
                              (epoch - 1) * 16 + (it + 1) // args.log_interval)
            writer.add_scalar('Loss/valid', valid_loss,
                              (epoch - 1) * 16 + (it + 1) // args.log_interval)
            writer.add_scalar('adv_loss/total_loss', adv_total_loss,
                              (epoch - 1) * 16 + (it + 1) // args.log_interval)
            writer.add_scalar('paraloss/para_1_2_loss', para_1_2_loss,
                              (epoch - 1) * 16 + (it + 1) // args.log_interval)
            writer.add_scalar('paraloss/para_2_1_loss', para_2_1_loss,
                              (epoch - 1) * 16 + (it + 1) // args.log_interval)
            writer.add_scalar('adv_loss/adv_1_loss', adv_1_loss,
                              (epoch - 1) * 16 + (it + 1) // args.log_interval)
            writer.add_scalar('adv_loss/adv_2_loss', adv_2_loss,
                              (epoch - 1) * 16 + (it + 1) // args.log_interval)

            del sent1_token_ids
            del synt1_token_ids
            # del sent1_target_role_ids
            del synt1_distances
            del synt1_depth
            del synt1_bow

            del sent2_token_ids
            del synt2_token_ids
            # del sent2_target_role_ids
            del synt2_distances
            del synt2_depth
            del synt2_bow

            gc.collect()


def train(epoch, model, tokenizer, optimizer, args):
    # NOTE: old version
    timer = Timer()
    n_it = len(train_loader)
    optimizer.zero_grad()
    for it, idxs in enumerate(train_loader):
        total_loss = 0.0
        adv_total_loss = 0.0
        model.train()

        sent1_token_ids = dataset['sent1'][idxs].cuda()
        synt1_token_ids = dataset['synt1'][idxs].cuda()
        synt1_bow = dataset['synt1bow'][idxs].cuda()
        sent2_token_ids = dataset['sent2'][idxs].cuda()
        synt2_token_ids = dataset['synt2'][idxs].cuda()
        synt2_bow = dataset['synt2bow'][idxs].cuda()

        # compute the similarity between `sent1` and `sent2`
        similarity = model.compute_similarity(sent1_token_ids,
                                              sent2_token_ids).cuda()

        # optimize adv
        # sent1 adv
        outputs = model.forward_adv(sent1_token_ids)
        targs = synt1_bow
        if args.similarity_as_weights:
            all_adv_loss = adv_criterion(outputs, targs)
            # batch_size, element_num = all_adv_loss.shape
            all_adv_loss = torch.mean(all_adv_loss, dim=-1)
            loss = torch.sum(all_adv_loss * similarity)
        else:
            loss = adv_criterion(outputs, targs)
        loss.backward()
        adv_total_loss += loss.item()

        # sent2 adv
        outputs = model.forward_adv(sent2_token_ids)
        targs = synt2_bow
        if args.similarity_as_weights:
            all_adv_loss = adv_criterion(outputs, targs)
            # batch_size, element_num = all_adv_loss.shape
            all_adv_loss = torch.mean(all_adv_loss, dim=-1)
            loss = torch.sum(all_adv_loss * similarity)
        else:
            loss = adv_criterion(outputs, targs)
        loss.backward()
        adv_total_loss += loss.item()

        if (it + 1) % args.accumulation_steps == 0:
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if epoch > 1:
                adv_optimizer.step()
            adv_optimizer.zero_grad()

        # optimize model
        # sent1->sent2 para & sent1 adv
        outputs, adv_outputs = model(
            torch.cat((sent1_token_ids, synt2_token_ids), 1), sent2_token_ids)
        batch_size, _, _ = outputs.shape
        targs = sent2_token_ids[:, 1:].contiguous().view(-1)
        outputs = outputs.contiguous().view(-1, outputs.size(-1))
        adv_targs = synt1_bow
        if args.similarity_as_weights:
            all_loss = para_criterion(outputs, targs)
            # pdb.set_trace()
            all_loss = all_loss.view(batch_size, -1).contiguous()
            element_num = torch.sum(all_loss != 0 + 0).item()
            all_loss = torch.sum(all_loss, dim=-1)
            loss = torch.sum(all_loss * similarity * batch_size / element_num)

            if epoch > 1:
                all_adv_loss = adv_criterion(adv_outputs, adv_targs)
                # batch_size, element_num = all_adv_loss.shape
                all_adv_loss = torch.mean(all_adv_loss, dim=-1)
                adv_loss = torch.sum(all_adv_loss * similarity)
                loss -= 0.1 * adv_loss
        else:
            loss = para_criterion(outputs, targs)
            if epoch > 1:
                loss -= 0.1 * adv_criterion(adv_outputs, adv_targs)
        loss.backward()
        total_loss += loss.item()

        # sent2->sent1 para & sent2 adv
        outputs, adv_outputs = model(
            torch.cat((sent2_token_ids, synt1_token_ids), 1), sent1_token_ids)
        batch_size, _, _ = outputs.shape
        targs = sent1_token_ids[:, 1:].contiguous().view(-1)
        outputs = outputs.contiguous().view(-1, outputs.size(-1))
        adv_targs = synt2_bow
        if args.similarity_as_weights:
            all_loss = para_criterion(outputs, targs)
            # pdb.set_trace()
            all_loss = all_loss.view(batch_size, -1).contiguous()
            element_num = torch.sum(all_loss != 0 + 0).item()
            all_loss = torch.sum(all_loss, dim=-1)
            loss = torch.sum(all_loss * similarity * batch_size / element_num)
            if epoch > 1:
                all_adv_loss = adv_criterion(adv_outputs, adv_targs)
                # batch_size, element_num = all_adv_loss.shape
                all_adv_loss = torch.mean(all_adv_loss, dim=-1)
                adv_loss = torch.sum(all_adv_loss * similarity)
                loss -= 0.1 * adv_loss
        else:
            loss = para_criterion(outputs, targs)
            if epoch > 1:
                loss -= 0.1 * adv_criterion(adv_outputs, adv_targs)
        loss.backward()
        total_loss += loss.item()

        if (it + 1) % args.accumulation_steps == 0:
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()

        if (it + 1) % args.log_interval == 0 or it == 0:
            para_1_2_loss, para_2_1_loss, adv_1_loss, adv_2_loss = evaluate(
                model, tokenizer, args)
            valid_loss = para_1_2_loss + para_2_1_loss - 0.1 * adv_1_loss - 0.1 * adv_2_loss
            print(
                "| ep {:2d}/{} | it {:3d}/{} | {:5.2f} s | adv loss {:.4f} | loss {:.4f} | para 1-2 loss {:.4f} | para 2-1 loss {:.4f} | adv 1 loss {:.4f} | adv 2 loss {:.4f} | valid loss {:.4f} |"
                .format(epoch, args.n_epoch, it + 1, n_it,
                        timer.get_time_from_last(), adv_total_loss, total_loss,
                        para_1_2_loss, para_2_1_loss, adv_1_loss, adv_2_loss,
                        valid_loss))
            writer.add_scalar('Loss/train', total_loss,
                              (epoch - 1) * 16 + (it + 1) // args.log_interval)
            writer.add_scalar('Loss/valid', valid_loss,
                              (epoch - 1) * 16 + (it + 1) // args.log_interval)
            writer.add_scalar('adv_loss/total_loss', adv_total_loss,
                              (epoch - 1) * 16 + (it + 1) // args.log_interval)
            writer.add_scalar('paraloss/para_1_2_loss', para_1_2_loss,
                              (epoch - 1) * 16 + (it + 1) // args.log_interval)
            writer.add_scalar('paraloss/para_2_1_loss', para_2_1_loss,
                              (epoch - 1) * 16 + (it + 1) // args.log_interval)
            writer.add_scalar('adv_loss/adv_1_loss', adv_1_loss,
                              (epoch - 1) * 16 + (it + 1) // args.log_interval)
            writer.add_scalar('adv_loss/adv_2_loss', adv_2_loss,
                              (epoch - 1) * 16 + (it + 1) // args.log_interval)

            del sent1_token_ids
            del synt1_token_ids
            del synt1_bow
            del sent2_token_ids
            del synt2_token_ids
            del synt2_bow

            gc.collect()


def structural_evaluate(model, tokenizer, args):
    model.eval()
    para_1_2_loss = 0.0
    para_2_1_loss = 0.0
    adv_1_loss = 0.0
    adv_2_loss = 0.0
    with torch.no_grad():
        # for i, (sent1_token_ids, synt1_token_ids, sent1_target_role_ids,
        #         synt1_distances, synt1_depth, synt1_bow, graph1,
        #         sent2_token_ids, synt2_token_ids, sent2_target_role_ids,
        #         synt2_distances, synt2_depth, synt2_bow,
        #         graph2) in enumerate(valid_loader):
        for i, (sent1_token_ids, synt1_token_ids, synt1_distances, synt1_depth,
                synt1_bow, graph1, sent2_token_ids, synt2_token_ids,
                synt2_distances, synt2_depth, synt2_bow, graph2,
                similarity) in enumerate(valid_loader):
            sent1_token_ids = sent1_token_ids.cuda()
            synt1_token_ids = synt1_token_ids.cuda()
            # sent1_target_role_ids = sent1_target_role_ids.cuda()
            synt1_distances = synt1_distances.cuda()
            synt1_depth = synt1_depth.cuda()
            synt1_bow = synt1_bow.cuda()

            sent2_token_ids = sent2_token_ids.cuda()
            synt2_token_ids = synt2_token_ids.cuda()
            # sent2_target_role_ids = sent2_target_role_ids.cuda()
            synt2_distances = synt2_distances.cuda()
            synt2_depth = synt2_depth.cuda()
            synt2_bow = synt2_bow.cuda()

            similarity = torch.tensor(similarity).cuda()
            similarity = nn.functional.softmax(similarity, dim=0)

            similarity = model.compute_similarity(sent1_token_ids,
                                                  sent2_token_ids,
                                                  similarity).cuda()

            # optimize model
            # sent1->sent2 para & sent1 adv
            outputs, pred_dist, pred_depth, pred_bow = model(
                sent1_token_ids, synt2_token_ids, graph2, sent2_token_ids)
            batch_size, _, _ = outputs.shape
            targs = sent2_token_ids[:, 1:].contiguous().view(-1)
            outputs = outputs.contiguous().view(-1, outputs.size(-1))

            if args.similarity_as_weights:
                all_para_1_2_loss = para_criterion(outputs, targs)
                # pdb.set_trace()
                all_para_1_2_loss = all_para_1_2_loss.view(batch_size,
                                                           -1).contiguous()
                element_num = torch.sum(all_para_1_2_loss != 0 + 0).item()
                all_para_1_2_loss = torch.sum(all_para_1_2_loss, dim=-1)
                para_1_2_loss += torch.sum(all_para_1_2_loss * similarity *
                                           batch_size / element_num)
            else:
                para_1_2_loss += para_criterion(outputs, targs)

            true_dist, true_depth, true_bow = synt1_distances, synt1_depth, synt1_bow

            if args.similarity_as_weights and args.adv_pred_bow and (
                    not args.adv_pred_distance) and (not args.adv_pred_depth):
                all_adv_loss = adv_criterion(pred_bow, true_bow)
                all_adv_loss = torch.mean(all_adv_loss, dim=-1)
                adv_1_loss += torch.sum(all_adv_loss * similarity)
            else:
                if args.predict_distance_square:
                    adv_1_loss += adv_criterion(pred_dist, true_dist.pow(2),
                                                pred_depth, true_depth,
                                                pred_bow, true_bow)
                else:
                    adv_1_loss += adv_criterion(pred_dist, true_dist,
                                                pred_depth, true_depth,
                                                pred_bow, true_bow)

            # sent2->sent1 para & sent2 adv
            outputs, pred_dist, pred_depth, pred_bow = model(
                sent2_token_ids, synt1_token_ids, graph1, sent1_token_ids)
            batch_size, _, _ = outputs.shape
            targs = sent1_token_ids[:, 1:].contiguous().view(-1)
            outputs = outputs.contiguous().view(-1, outputs.size(-1))

            if args.similarity_as_weights:
                all_para_2_1_loss = para_criterion(outputs, targs)
                # pdb.set_trace()
                all_para_2_1_loss = all_para_2_1_loss.view(batch_size,
                                                           -1).contiguous()
                element_num = torch.sum(all_para_2_1_loss != 0 + 0).item()
                all_para_2_1_loss = torch.sum(all_para_2_1_loss, dim=-1)
                para_2_1_loss += torch.sum(all_para_2_1_loss * similarity *
                                           batch_size / element_num)
            else:
                para_2_1_loss += para_criterion(outputs, targs)

            true_dist, true_depth, true_bow = synt2_distances, synt2_depth, synt2_bow

            if args.similarity_as_weights and args.adv_pred_bow and (
                    not args.adv_pred_distance) and (not args.adv_pred_depth):
                all_adv_loss = adv_criterion(pred_bow, true_bow)
                all_adv_loss = torch.mean(all_adv_loss, dim=-1)
                adv_2_loss += torch.sum(all_adv_loss * similarity)
            else:
                if args.predict_distance_square:
                    adv_2_loss += adv_criterion(pred_dist, true_dist.pow(2),
                                                pred_depth, true_depth,
                                                pred_bow, true_bow)
                else:
                    adv_2_loss += adv_criterion(pred_dist, true_dist,
                                                pred_depth, true_depth,
                                                pred_bow, true_bow)

    return para_1_2_loss / len(valid_loader), para_2_1_loss / len(
        valid_loader), adv_1_loss / len(valid_loader), adv_2_loss / len(
            valid_loader)


def evaluate(model, tokenizer, args):
    # NOTE: old version
    model.eval()
    para_1_2_loss = 0.0
    para_2_1_loss = 0.0
    adv_1_loss = 0.0
    adv_2_loss = 0.0
    with torch.no_grad():
        for idxs in valid_loader:

            sent1_token_ids = dataset['sent1'][idxs].cuda()
            synt1_token_ids = dataset['synt1'][idxs].cuda()
            sent2_token_ids = dataset['sent2'][idxs].cuda()
            synt2_token_ids = dataset['synt2'][idxs].cuda()
            synt1_bow = dataset['synt1bow'][idxs].cuda()
            synt2_bow = dataset['synt2bow'][idxs].cuda()

            # compute the similarity between `sent1` and `sent2`
            similarity = model.compute_similarity(sent1_token_ids,
                                                  sent2_token_ids).cuda()

            outputs, adv_outputs = model(
                torch.cat((sent1_token_ids, synt2_token_ids), 1),
                sent2_token_ids)
            batch_size, _, _ = outputs.shape
            targs = sent2_token_ids[:, 1:].contiguous().view(-1)
            outputs = outputs.contiguous().view(-1, outputs.size(-1))
            adv_targs = synt1_bow
            if args.similarity_as_weights:
                all_para_1_2_loss = para_criterion(outputs, targs)
                all_para_1_2_loss = all_para_1_2_loss.view(batch_size,
                                                           -1).contiguous()
                element_num = torch.sum(all_para_1_2_loss != 0 + 0).item()
                all_para_1_2_loss = torch.sum(all_para_1_2_loss, dim=-1)
                para_1_2_loss += torch.sum(all_para_1_2_loss * similarity *
                                           batch_size / element_num)

                all_adv_loss = adv_criterion(adv_outputs, adv_targs)
                # batch_size, element_num = all_adv_loss.shape
                all_adv_loss = torch.mean(all_adv_loss, dim=-1)
                adv_1_loss += torch.sum(all_adv_loss * similarity)
            else:
                para_1_2_loss += para_criterion(outputs, targs)
                adv_1_loss += adv_criterion(adv_outputs, adv_targs)

            outputs, adv_outputs = model(
                torch.cat((sent2_token_ids, synt1_token_ids), 1),
                sent1_token_ids)
            batch_size, _, _ = outputs.shape
            targs = sent1_token_ids[:, 1:].contiguous().view(-1)
            outputs = outputs.contiguous().view(-1, outputs.size(-1))
            adv_targs = synt2_bow
            if args.similarity_as_weights:
                all_para_2_1_loss = para_criterion(outputs, targs)
                all_para_2_1_loss = all_para_2_1_loss.view(batch_size,
                                                           -1).contiguous()
                element_num = torch.sum(all_para_2_1_loss != 0 + 0).item()
                all_para_2_1_loss = torch.sum(all_para_2_1_loss, dim=-1)
                para_2_1_loss += torch.sum(all_para_2_1_loss * similarity *
                                           batch_size / element_num)

                all_adv_loss = adv_criterion(adv_outputs, adv_targs)
                # batch_size, element_num = all_adv_loss.shape
                all_adv_loss = torch.mean(all_adv_loss, dim=-1)
                adv_2_loss += torch.sum(all_adv_loss * similarity)
            else:
                para_2_1_loss += para_criterion(outputs, targs)
                adv_2_loss += adv_criterion(adv_outputs, adv_targs)

    return para_1_2_loss / len(valid_loader), para_2_1_loss / len(
        valid_loader), adv_1_loss / len(valid_loader), adv_2_loss / len(
            valid_loader)


print("==== preparing data ====")
make_path(args.cache_dir)
tokenizer = BartTokenizer.from_pretrained('facebook/bart-base',
                                          cache_dir=args.cache_dir)

with open('synt_vocab.pkl', 'rb') as f:
    synt_vocab = pickle.load(f)

print("==== loading data ====")
if args.mode == 'baseline':
    num = 1000000
    # num = 6000
    train_idxs, valid_idxs = random_split(
        range(num), [num - 5000, 5000],
        generator=torch.Generator().manual_seed(args.seed))
    print(f"number of train examples: {len(train_idxs)}")
    print(f"number of valid examples: {len(valid_idxs)}")
    train_loader = DataLoader(train_idxs,
                              batch_size=args.train_batch_size,
                              shuffle=True)
    valid_loader = DataLoader(valid_idxs,
                              batch_size=args.valid_batch_size,
                              shuffle=False)
    if os.path.exists('./data/baseline_data.pkl'):
        with open('./data/baseline_data.pkl', 'rb') as f:
            dataset = pickle.load(f)
    else:
        para_data = h5py.File(os.path.join(args.data_dir, 'data.h5'), 'r')
        dataset = prepare_dataset(para_data, tokenizer, num, args, synt_vocab)
        with open('./data/baseline_data.pkl', 'wb') as f:
            pickle.dump(dataset, f)
elif args.mode == 'structural':
    train_data = StructuralDataset('data/train_data_with_score.json', tokenizer,
                                   synt_vocab, args.max_sent_len,
                                   args.max_synt_len, args.use_linear_tree)
    valid_data = StructuralDataset('data/valid_data_with_score.json', tokenizer,
                                   synt_vocab, args.max_sent_len,
                                   args.max_synt_len, args.use_linear_tree)
    print(f"number of train examples: {len(train_data)}")
    print(f"number of valid examples: {len(valid_data)}")

    train_loader = DataLoader(train_data,
                              batch_size=args.train_batch_size,
                              shuffle=True,
                              collate_fn=collate)
    valid_loader = DataLoader(valid_data,
                              batch_size=args.valid_batch_size,
                              shuffle=False,
                              collate_fn=collate)

print("==== loading model ====")
config = BartConfig.from_pretrained('facebook/bart-base',
                                    cache_dir=args.cache_dir)
# pdb.set_trace()
config.word_dropout = args.word_dropout
config.max_sent_len = args.max_sent_len
config.max_synt_len = args.max_synt_len
config.syntax_encoder_layer_num = args.syntax_encoder_layer_num
config.rank = args.adv_rank
config.use_GAT = args.use_GAT

# NOTE: due to the consistance of tokenizer and embeddings layer in pre-trained Bart Model
# cannot use my own tokenizer.
# config.pad_token_id = vocab_dict[PAD_TOKEN]
# config.bos_token_id = vocab_dict[BOS_TOKEN]
# config.eos_token_id = vocab_dict[EOS_TOKEN]
# config.vocab_size = len(vocab_dict)

bart = BartModel.from_pretrained('facebook/bart-base', cache_dir=args.cache_dir)
if args.mode == 'baseline':
    model = ParaBart(config)
elif args.mode == 'structural':
    model = StructuralParaBart(config)

if os.path.exists(args.model_dir):
    checkpoint, last_epoch = last_checkpoint(args.model_dir)
    if checkpoint is None:
        model.load_state_dict(bart.state_dict(), strict=False)
        last_epoch = 0
    else:
        # state = torch.load(os.path.join(args.model_dir, checkpoint),
        #                    map_location='cpu')
        state = torch.load(checkpoint, map_location='cpu')
        model.load_state_dict(state)
else:
    model.load_state_dict(bart.state_dict(), strict=False)
    last_epoch = 0
model.zero_grad()
del bart

no_decay_params = []
no_decay_fast_params = []
fast_params = []
all_other_params = []
adv_no_decay_params = []
adv_all_other_params = []

for n, p in model.named_parameters():
    if 'adv' in n:
        if 'norm' in n or 'bias' in n:
            adv_no_decay_params.append(p)
        else:
            adv_all_other_params.append(p)
    elif 'linear' in n or 'synt' in n or 'decoder' in n:
        if 'bias' in n:
            no_decay_fast_params.append(p)
        else:
            fast_params.append(p)
    elif 'norm' in n or 'bias' in n:
        no_decay_params.append(p)
    else:
        all_other_params.append(p)

if args.mode == 'baseline':
    optimizer = optim.AdamW([{
        'params': fast_params,
        'lr': args.fast_lr
    }, {
        'params': no_decay_fast_params,
        'lr': args.fast_lr,
        'weight_decay': 0.0
    }, {
        'params': no_decay_params,
        'weight_decay': 0.0
    }, {
        'params': all_other_params
    }],
                            lr=args.lr,
                            weight_decay=args.weight_decay)

    adv_optimizer = optim.AdamW([{
        'params': adv_no_decay_params,
        'weight_decay': 0.0
    }, {
        'params': adv_all_other_params
    }],
                                lr=args.lr,
                                weight_decay=args.weight_decay)
elif args.mode == 'structural':
    optimizer = optim.AdamW([{
        'params': fast_params,
        'lr': args.fast_lr
    }, {
        'params': no_decay_fast_params,
        'lr': args.fast_lr,
        'weight_decay': 0.0
    }, {
        'params': no_decay_params,
        'weight_decay': 0.0
    }, {
        'params': all_other_params
    }],
                            lr=args.lr,
                            weight_decay=args.weight_decay)

    # adv_optimizer = optim.AdamW([{
    #     'params': adv_no_decay_params,
    #     'weight_decay': 0.0
    # }, {
    #     'params': adv_all_other_params
    # }],
    #                             lr=args.lr,
    #                             weight_decay=args.weight_decay)
    adv_optimizer = optim.AdamW(model.adversary.parameters(), lr=args.adv_lr)
    if args.scheduler:
        # t, T = 20000, 160000
        # gamma = 0.9998
        # # adv_lambda = lambda step: 1 if step < t else (step - t) / (
        # #     T - t) if step < T else 0
        # adv_lambda = lambda step: 1 if step < t else pow(gamma, (step - t))
        # adv_scheduler = optim.lr_scheduler.LambdaLR(adv_optimizer, adv_lambda)
        adv_scheduler = optim.lr_scheduler.ReduceLROnPlateau(adv_optimizer,
                                                             mode='min',
                                                             factor=0.1,
                                                             patience=0)

model = model.cuda()

if args.similarity_as_weights:
    para_criterion = nn.CrossEntropyLoss(ignore_index=model.config.pad_token_id,
                                         reduction='none').cuda()
    adv_bow_criterion = nn.BCEWithLogitsLoss(reduction='none').cuda()
else:
    para_criterion = nn.CrossEntropyLoss(
        ignore_index=model.config.pad_token_id).cuda()
    adv_bow_criterion = nn.BCEWithLogitsLoss().cuda()

adv_distance_criterion = L1DistanceLoss(model.device)
adv_depth_criterion = L1DepthLoss(model.device)
# adv_contractive_criterion

if args.mode == 'baseline':
    # adv_criterion = nn.BCEWithLogitsLoss().cuda()
    def adv_criterion(pred_bow, true_bow):
        return adv_bow_criterion(pred_bow, true_bow)
elif args.mode == 'structural':
    if args.adv_pred_bow and (not args.adv_pred_distance) and (
            not args.adv_pred_depth):

        def adv_criterion(pred_bow, true_bow):
            return adv_bow_criterion(pred_bow, true_bow)
    else:

        def adv_criterion(pred_dist, true_dist, pred_depth, true_depth,
                          pred_bow, true_bow):
            # pdb.set_trace()
            length_batch = torch.sum(true_depth != 0 + 0, dim=1)

            if args.adv_pred_distance:
                loss_distance, _ = adv_distance_criterion(
                    pred_dist, true_dist, length_batch)
            else:
                loss_distance = 0

            if args.adv_pred_depth:
                loss_depth, _ = adv_depth_criterion(pred_depth, true_depth,
                                                    length_batch)
            else:
                loss_depth = 0

            if args.adv_pred_bow:
                loss_bow = adv_bow_criterion(pred_bow, true_bow)
            else:
                loss_bow = 0

            # if args.adv_contractive
            #     loss_constractive = adv_contractive_criterion(embeds, similarity)

            return args.adv_coefficient * loss_distance + args.adv_coefficient * loss_depth + loss_bow


make_path(args.model_dir)

print("==== start training ====")

for epoch in range(last_epoch + 1, args.n_epoch + 1):
    if args.mode == 'baseline':
        train(epoch, model, tokenizer, optimizer, args)
    elif args.mode == 'structural':
        structural_train(epoch, model, tokenizer, optimizer, args)
    torch.save(
        model.state_dict(),
        os.path.join(args.model_dir, "model_epoch{:02d}.pt".format(epoch)))
