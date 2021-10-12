import torch
from tqdm import tqdm
from utils import deleaf

PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"
BOS_TOKEN = "<s>"
EOS_TOKEN = "</s>"


def prepare_dataset(para_data, tokenizer, num, args, synt_vocab):
    sents1 = list(para_data['train_sents1'][:num])
    synts1 = list(para_data['train_synts1'][:num])
    sents2 = list(para_data['train_sents2'][:num])
    synts2 = list(para_data['train_synts2'][:num])

    sent1_token_ids = torch.ones((num, args.max_sent_len + 2), dtype=torch.long)
    sent2_token_ids = torch.ones((num, args.max_sent_len + 2), dtype=torch.long)
    synt1_token_ids = torch.ones((num, args.max_synt_len + 2), dtype=torch.long)
    synt2_token_ids = torch.ones((num, args.max_synt_len + 2), dtype=torch.long)
    synt1_bow = torch.ones((num, 74))
    synt2_bow = torch.ones((num, 74))

    bsz = 64

    for i in tqdm(range(0, num, bsz)):
        sent1_inputs = tokenizer([
            sent.decode('utf-8') if isinstance(sent, bytes) else sent
            for sent in sents1[i:i + bsz]
        ],
                                 padding='max_length',
                                 truncation=True,
                                 max_length=args.max_sent_len + 2,
                                 return_tensors="pt")
        sent2_inputs = tokenizer([
            sent.decode('utf-8') if isinstance(sent, bytes) else sent
            for sent in sents2[i:i + bsz]
        ],
                                 padding='max_length',
                                 truncation=True,
                                 max_length=args.max_sent_len + 2,
                                 return_tensors="pt")
        sent1_token_ids[i:i + bsz] = sent1_inputs['input_ids']
        sent2_token_ids[i:i + bsz] = sent2_inputs['input_ids']

    for i in tqdm(range(num)):
        synt1 = [BOS_TOKEN] + deleaf(synts1[i]) + [EOS_TOKEN]
        synt1_token_ids[i, :len(synt1)] = torch.tensor(
            [synt_vocab[tag] for tag in synt1])[:args.max_synt_len + 2]
        synt2 = [BOS_TOKEN] + deleaf(synts2[i]) + [EOS_TOKEN]
        synt2_token_ids[i, :len(synt2)] = torch.tensor(
            [synt_vocab[tag] for tag in synt2])[:args.max_synt_len + 2]

        for tag in synt1:
            if tag != BOS_TOKEN and tag != EOS_TOKEN:
                synt1_bow[i][synt_vocab[tag] - 3] += 1
        for tag in synt2:
            if tag != BOS_TOKEN and tag != EOS_TOKEN:
                synt2_bow[i][synt_vocab[tag] - 3] += 1

    synt1_bow /= synt1_bow.sum(1, keepdim=True)
    synt2_bow /= synt2_bow.sum(1, keepdim=True)

    sum = 0
    for i in range(num):
        if torch.equal(synt1_bow[i], synt2_bow[i]):
            sum += 1

    return {
        'sent1': sent1_token_ids,
        'sent2': sent2_token_ids,
        'synt1': synt1_token_ids,
        'synt2': synt2_token_ids,
        'synt1bow': synt1_bow,
        'synt2bow': synt2_bow
    }


def collate(samples):
    batch_size = len(samples)
    sents1_token_ids = torch.stack([samples[i][0] for i in range(batch_size)],
                                   dim=0).long()
    synts1_token_ids = torch.stack([samples[i][1] for i in range(batch_size)],
                                   dim=0).long()
    # sents1_target_role_ids = torch.stack(
    #     [samples[i][2] for i in range(batch_size)], dim=0).long()
    synts1_distance = torch.stack([samples[i][2] for i in range(batch_size)],
                                  dim=0).long()
    synts1_depth = torch.stack([samples[i][3] for i in range(batch_size)],
                               dim=0).long()
    synts1_bow = torch.stack([samples[i][4] for i in range(batch_size)],
                             dim=0).float()
    # sents1_token_ids = torch.as_tensor(
    #     [samples[i][0].numpy().tolist() for i in range(batch_size)],
    #     dtype=torch.long)
    # synts1_token_ids = torch.as_tensor(
    #     [samples[i][1].numpy().tolist() for i in range(batch_size)],
    #     dtype=torch.long)
    # sents1_target_role_ids = torch.as_tensor(
    #     [samples[i][2].numpy().tolist() for i in range(batch_size)],
    #     dtype=torch.long)
    # synts1_distance = torch.as_tensor(
    #     [samples[i][3].numpy().tolist() for i in range(batch_size)],
    #     dtype=torch.long)
    # synts1_depth = torch.as_tensor(
    #     [samples[i][4].numpy().tolist() for i in range(batch_size)],
    #     dtype=torch.long)
    # synts1_bow = torch.as_tensor(
    #     [samples[i][5].numpy().tolist() for i in range(batch_size)],
    #     dtype=torch.float)
    graphs1 = [samples[i][5] for i in range(batch_size)]

    # sents2_token_ids = torch.as_tensor(
    #     [samples[i][7].numpy().tolist() for i in range(batch_size)],
    #     dtype=torch.long)
    # synts2_token_ids = torch.as_tensor(
    #     [samples[i][8].numpy().tolist() for i in range(batch_size)],
    #     dtype=torch.long)
    # sents2_target_role_ids = torch.as_tensor(
    #     [samples[i][9].numpy().tolist() for i in range(batch_size)],
    #     dtype=torch.long)
    # synts2_distance = torch.as_tensor(
    #     [samples[i][10].numpy().tolist() for i in range(batch_size)],
    #     dtype=torch.long)
    # synts2_depth = torch.as_tensor(
    #     [samples[i][11].numpy().tolist() for i in range(batch_size)],
    #     dtype=torch.long)
    # synts2_bow = torch.as_tensor(
    #     [samples[i][12].numpy().tolist() for i in range(batch_size)],
    #     dtype=torch.float)
    sents2_token_ids = torch.stack([samples[i][6] for i in range(batch_size)],
                                   dim=0).long()
    synts2_token_ids = torch.stack([samples[i][7] for i in range(batch_size)],
                                   dim=0).long()
    # sents2_target_role_ids = torch.stack(
    #     [samples[i][9] for i in range(batch_size)], dim=0).long()
    synts2_distance = torch.stack([samples[i][8] for i in range(batch_size)],
                                  dim=0).long()
    synts2_depth = torch.stack([samples[i][9] for i in range(batch_size)],
                               dim=0).long()
    synts2_bow = torch.stack([samples[i][10] for i in range(batch_size)],
                             dim=0).float()
    graphs2 = [samples[i][11] for i in range(batch_size)]

    similarity_score = [samples[i][12] for i in range(batch_size)]

    return sents1_token_ids, synts1_token_ids, synts1_distance, synts1_depth, synts1_bow, graphs1, sents2_token_ids, synts2_token_ids, synts2_distance, synts2_depth, synts2_bow, graphs2, similarity_score
