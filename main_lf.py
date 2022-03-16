import collections
import math

import numpy as np
import torch.cuda
from tqdm import tqdm
from torch.utils.data import DataLoader, IterableDataset
from transformers import BartConfig, AdamW

from model_bart import BartForConditionalGeneration

IS_DEBUG = True
DATASET = "TED"
SRC_LANG = "en"
TGT_LANG = "de"
MAX_LENGTH = 128
BATCH_SIZE = 2 if IS_DEBUG else 16
EMBEDDING_SIZE = 512
ENCODER_LAYER = 6
DECODER_LAYER = 6
ENCODER_ATT_HEAD = 8
DECODER_ATT_HEAD = 8
ENCODE_FFN_DIM = 2048
DECODER_FFN_DIM = 2048
DROPOUT = 0.3
LR = 0.0005
MAX_POS_EMBEDDING = 128
WARMUP_STEPS = 4000
PRINT_STEP = 10 if IS_DEBUG else 100
EVAL_STEP = 10 if IS_DEBUG else 3000

DATA_PATH = "data/" + DATASET + "/"
VOCAB_PATH = DATA_PATH + "vocab"
SRC_TRAIN_PATH = DATA_PATH + "train." + SRC_LANG
TGT_TRAIN_PATH = DATA_PATH + "train." + TGT_LANG
SRC_TEST_PATH = DATA_PATH + "test." + SRC_LANG
TGT_TEST_PATH = DATA_PATH + "test." + TGT_LANG

device = "cuda" if torch.cuda.is_available() else "cpu"


def self_collate_fn(batch):
    batch_max_length = 0
    for sample in batch:
        for ndarray in sample:
            batch_max_length = max(ndarray.size, batch_max_length)

    batch_max_length = min(MAX_LENGTH, batch_max_length)
    encoder_input_ids = torch.zeros((len(batch), batch_max_length))
    encoder_attention_mask = torch.zeros((len(batch), batch_max_length))
    decoder_input_ids = torch.zeros((len(batch), batch_max_length))
    decoder_attention_mask = torch.zeros((len(batch), batch_max_length))
    label_ids = torch.zeros((len(batch), batch_max_length))
    for i in range(len(batch)):
        encoder_input_ids[i] = torch.cat(
            [torch.from_numpy(batch[i][0]), torch.zeros((batch_max_length - batch[i][0].size))])
        encoder_attention_mask[i] = torch.cat(
            [torch.ones(batch[i][0].size), torch.zeros((batch_max_length - batch[i][0].size))])
        decoder_input_ids[i] = torch.cat(
            [torch.from_numpy(batch[i][1]), torch.zeros((batch_max_length - batch[i][1].size))])
        decoder_attention_mask[i] = torch.cat(
            [torch.ones(batch[i][1].size), torch.zeros((batch_max_length - batch[i][1].size))])
        label_ids[i] = torch.cat([torch.from_numpy(batch[i][2]), torch.zeros((batch_max_length - batch[i][2].size))])

    return encoder_input_ids.long(), encoder_attention_mask.long(), decoder_input_ids.long(), decoder_attention_mask.long(), label_ids.long()


def _get_ngrams(segment, max_order):
    """Extracts all n-grams upto a given maximum order from an input segment.
    Args:
      segment: text segment from which n-grams will be extracted.
      max_order: maximum length in tokens of the n-grams returned by this
          methods.
    Returns:
      The Counter containing all n-grams upto max_order in segment
      with a count of how many times each n-gram occurred.
    """
    ngram_counts = collections.Counter()
    for order in range(1, max_order + 1):
        for i in range(0, len(segment) - order + 1):
            ngram = tuple(segment[i:i + order])
            ngram_counts[ngram] += 1
    return ngram_counts


def compute_bleu(reference_corpus, translation_corpus, max_order=4,
                 smooth=False):
    """Computes BLEU score of translated segments against one or more references.
    Args:
      reference_corpus: list of lists of references for each translation. Each
          reference should be tokenized into a list of tokens.
      translation_corpus: list of translations to score. Each translation
          should be tokenized into a list of tokens.
      max_order: Maximum n-gram order to use when computing BLEU score.
      smooth: Whether or not to apply Lin et al. 2004 smoothing.
    Returns:
      3-Tuple with the BLEU score, n-gram precisions, geometric mean of n-gram
      precisions and brevity penalty.
    """
    matches_by_order = [0] * max_order
    possible_matches_by_order = [0] * max_order
    reference_length = 0
    translation_length = 0
    for (references, translation) in zip(reference_corpus,
                                         translation_corpus):
        reference_length += min(len(r) for r in references)
        translation_length += len(translation)

        merged_ref_ngram_counts = collections.Counter()
        for reference in references:
            merged_ref_ngram_counts |= _get_ngrams(reference, max_order)
        translation_ngram_counts = _get_ngrams(translation, max_order)
        overlap = translation_ngram_counts & merged_ref_ngram_counts
        for ngram in overlap:
            matches_by_order[len(ngram) - 1] += overlap[ngram]
        for order in range(1, max_order + 1):
            possible_matches = len(translation) - order + 1
            if possible_matches > 0:
                possible_matches_by_order[order - 1] += possible_matches

    precisions = [0] * max_order
    for i in range(0, max_order):
        if smooth:
            precisions[i] = ((matches_by_order[i] + 1.) /
                             (possible_matches_by_order[i] + 1.))
        else:
            if possible_matches_by_order[i] > 0:
                precisions[i] = (float(matches_by_order[i]) /
                                 possible_matches_by_order[i])
            else:
                precisions[i] = 0.0

    if min(precisions) > 0:
        p_log_sum = sum((1. / max_order) * math.log(p) for p in precisions)
        geo_mean = math.exp(p_log_sum)
    else:
        geo_mean = 0
    ratio = float(translation_length) / reference_length
    if ratio > 1.0:
        bp = 1.
    else:
        bp = math.exp(1 - 1. / ratio)
    bleu = geo_mean * bp
    return bleu, precisions, bp, ratio, translation_length, reference_length


def get_vocab(VOCAB_PATH):
    v2id = {}
    id2v = {}
    with open(VOCAB_PATH, "r", encoding="utf-8") as fr:
        lines = fr.readlines()
        for i, line in enumerate(lines):
            vocab = line[:-1]
            v2id[vocab] = i
            id2v[i] = vocab
    return v2id, id2v


class IterMTDataset(IterableDataset):
    def __init__(self, src_file_path: str, tgt_file_path: str, words_to_ids):
        super(IterMTDataset).__init__()
        self.src_file_path = src_file_path
        self.tgt_file_path = tgt_file_path
        self.start, self.end = self.get_file_start_end()
        self.words_to_ids = words_to_ids

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            iter_start = self.start
            iter_end = self.end
        else:
            per_worker_num = math.ceil((self.end - self.start) / worker_info.nums_workers)
            worker_id = worker_info.id
            iter_start = self.start + worker_id * per_worker_num
            iter_end = min(iter_start + per_worker_num, self.end)

        return self._sample_generator(iter_start, iter_end)

    def get_file_start_end(self):
        src_file_path = self.src_file_path
        tgt_file_path = self.tgt_file_path
        start = 0
        src_end = 0
        tgt_end = 0
        with open(src_file_path, "r") as src_fr, open(tgt_file_path, "r") as tgt_fr:
            for _ in enumerate(src_fr):
                src_end += 1
            for _ in enumerate(tgt_fr):
                tgt_end += 1
            assert src_end == tgt_end
        return start, src_end

    def _sample_generator(self, start, end):
        words_to_ids = self.words_to_ids
        with open(self.src_file_path, "r") as src_fr, open(self.tgt_file_path, "r") as tgt_fr:
            for i, (src_line, tgt_line) in enumerate(zip(src_fr, tgt_fr)):
                if i < start: continue
                if i >= end: return StopIteration()
                src_line = src_line[:-1]
                tgt_line = tgt_line[:-1]
                src_tokens = src_line.split()
                tgt_tokens = tgt_line.split()
                encoder_tokens = ["[CLS]"] + src_tokens + ["[SEP]"]
                decoder_tokens = ["[CLS]"] + tgt_tokens + ["[SEP]"]

                encoder_input_ids = []
                decoder_input_ids = []

                if len(encoder_tokens) > MAX_LENGTH or len(decoder_tokens) > MAX_LENGTH:
                    continue

                for token in encoder_tokens:
                    if token not in words_to_ids:
                        print(f"[UNK TOKEN] : {token}")
                        continue
                    encoder_input_ids.append(words_to_ids[token])

                for token in decoder_tokens:
                    if token not in words_to_ids:
                        print(f"[UNK TOKEN] : {token}")
                        continue
                    decoder_input_ids.append(words_to_ids[token])

                label_ids = decoder_input_ids[1:]
                decoder_input_ids = decoder_input_ids[:-1]

                sample = [
                    np.array(encoder_input_ids),
                    np.array(decoder_input_ids),
                    np.array(label_ids)
                ]
                yield sample

    def __len__(self):
        return self.end - self.start


class ReverseSqrtScheduler:
    def __init__(self, optimizer, lr, n_warmup_steps):
        self._optimizer = optimizer
        self.lr_mul = lr
        self.n_warmup_steps = n_warmup_steps
        self.n_steps = 0

        self.decay_factor = [_lr * n_warmup_steps ** 0.5 for _lr in lr]
        self.lr_step = [(_lr - 0) / n_warmup_steps for _lr in lr]

    def step_and_update_lr(self):
        self._update_learning_rate()
        self._optimizer.step()

    def zero_grad(self):
        self._optimizer.zero_grad()

    def _update_learning_rate(self):
        self.n_steps += 1
        if self.n_steps < self.n_warmup_steps:
            lr = [self.n_steps * _lr for _lr in self.lr_step]
        else:
            lr = [_decay_factor * self.n_steps ** -0.5 for _decay_factor in self.decay_factor]

        for i, param_group in enumerate(self._optimizer.param_groups):
            param_group['lr'] = lr[i]


def eval_the_model(model, test_dataloader, ids_to_words):
    eos = 2  # tokenizer.sep_token_id
    sos = 1  # tokenizer.cls_token_id

    preds = []
    targets = []
    with torch.no_grad():
        model.eval()
        for batch in tqdm(test_dataloader):

            encoder_input_ids, encoder_attention_mask, decoder_input_ids, decoder_attention_mask, label_ids = batch

            src_input_ids = encoder_input_ids.to(device)
            src_attention_mask = encoder_attention_mask.to(device)
            tgt_input_ids = torch.zeros((decoder_input_ids.size()[0], 1)).to(device).long() + sos
            tgt_label_ids = label_ids.to(device).tolist()

            outputs = model.generate(input_ids=src_input_ids,
                                     attention_mask=src_attention_mask,
                                     decoder_input_ids=tgt_input_ids,
                                     bos_token_id=sos,
                                     num_return_sequences=1,
                                     num_beams=5,
                                     max_length=MAX_LENGTH)  # use_cache=False)
            outputs = outputs.cpu().tolist()
            assert len(outputs) == len(tgt_label_ids)

            for i in range(len(outputs)):
                pred = outputs[i]
                if eos in pred:
                    index = pred.index(eos)
                    pred = pred[:index]

                pred_list = []
                for _id in pred[1:]:
                    pred_list.append(ids_to_words[_id])
                pred_str = " ".join(pred_list)
                pred_str = pred_str.replace("@@ ", "")

                target = tgt_label_ids[i]
                index = target.index(eos)
                target_list = []
                for _id in target[:index]:
                    target_list.append(ids_to_words[_id])
                target_str = " ".join(target_list)
                target_str = target_str.replace("@@ ", "")

                preds.append(pred_str.split())
                targets.append([target_str.split()])

    model.train()
    score = float(compute_bleu(targets, preds, max_order=4)[0])
    print(f"BLEU score: {score}\n")
    return score


words_to_ids, ids_to_words = get_vocab(VOCAB_PATH)
train_dataset = IterMTDataset(SRC_TRAIN_PATH, TGT_TRAIN_PATH, words_to_ids)
train_dataloader = DataLoader(dataset=train_dataset, shuffle=False, batch_size=BATCH_SIZE, collate_fn=self_collate_fn,
                              pin_memory=True)

test_dataset = IterMTDataset(SRC_TEST_PATH, TGT_TEST_PATH, words_to_ids)
test_dataloader = DataLoader(dataset=test_dataset, shuffle=False, batch_size=BATCH_SIZE, collate_fn=self_collate_fn,
                             pin_memory=True)

config = BartConfig(
    vocab_size=len(words_to_ids),
    d_model=EMBEDDING_SIZE,
    encoder_layers=ENCODER_LAYER,
    decoder_layers=DECODER_LAYER,
    encoder_attention_heads=ENCODER_ATT_HEAD,
    decoder_attention_heads=DECODER_ATT_HEAD,
    encoder_ffn_dim=ENCODE_FFN_DIM,
    decoder_ffn_dim=DECODER_FFN_DIM,
    dropout=DROPOUT,
    attention_dropout=DROPOUT,
    max_position_embeddings=MAX_POS_EMBEDDING,
    pad_token_id=0,
    bos_token_id=1,
    eos_token_id=2,
    decoder_start_token_id=1
)

model = BartForConditionalGeneration(config)
model.to(device)

optimizer = AdamW(model.parameters(), lr=LR, betas=(0.9, 0.98))
scheduler = ReverseSqrtScheduler(optimizer, [LR], WARMUP_STEPS)
for epoch in range(0, 100):
    model.train()
    total_loss = 0
    update_step = 0

    for batch in tqdm(train_dataloader):
        encoder_input_ids, encoder_attention_mask, decoder_input_ids, decoder_attention_mask, label_ids = batch
        encoder_input_ids = encoder_input_ids.to(device)
        encoder_attention_mask = encoder_attention_mask.to(device)
        decoder_input_ids = decoder_input_ids.to(device)
        decoder_attention_mask = decoder_attention_mask.to(device)
        label_ids = label_ids.to(device)

        assert encoder_input_ids.shape[-1] == encoder_attention_mask.shape[-1]
        assert decoder_input_ids.shape[-1] == encoder_input_ids.shape[-1]
        assert decoder_attention_mask.shape[-1] == encoder_input_ids.shape[-1]
        assert label_ids.shape[-1] == encoder_input_ids.shape[-1]
        assert encoder_input_ids.shape[-1] <= MAX_LENGTH

        loss = model(
            input_ids=encoder_input_ids,
            attention_mask=encoder_attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            labels=label_ids
        )["loss"]
        total_loss += loss.item()
        loss.backward()
        scheduler.step_and_update_lr()
        scheduler.zero_grad()
        update_step += 1
        if update_step % PRINT_STEP == 0:
            print(f"\nloss: {total_loss / update_step}")
        if update_step % EVAL_STEP == 0:
            eval_the_model(model, test_dataloader, ids_to_words)

    eval_the_model(model, test_dataloader, ids_to_words)
