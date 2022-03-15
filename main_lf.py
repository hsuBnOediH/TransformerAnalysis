import collections
import math
import pickle

import numpy as np
import torch.cuda
import os
from tqdm import trange, tqdm
from torch.utils.data import DataLoader, Dataset
from transformers import BartConfig, AdamW

from model_bart import BartForConditionalGeneration

DATASET = "TED"
SRC_LANG = "en"
TGT_LANG = "de"
MAX_LENGTH = 128
BATCH_SIZE = 2
EMBEDDING_SIZE = 512
ENCODER_LAYER = 6
DECODER_LAYER = 6
ENCODER_ATT_HEAD = 8
DECODER_ATT_HEAD = 8
ENCODE_FFN_DIM = 2048
DECODER_FFN_DIM = 2048
DROPOUT = 0.3
LR = 0.0005
# TODO? Why 104?
MAX_POS_EMBEDDING = 128
WARMUP_STEPS = 4000
PRINT_STEP = 200
EVAL_STEP = 1000

DATA_PATH = "data/" + DATASET + "/"
VOCAB_PATH = DATA_PATH + "vocab"
SRC_TRAIN_PATH = DATA_PATH + "train." + SRC_LANG
TGT_TRAIN_PATH = DATA_PATH + "train." + TGT_LANG
SRC_TEST_PATH = DATA_PATH + "test." + SRC_LANG
TGT_TEST_PATH = DATA_PATH + "test." + TGT_LANG

device = "cuda" if torch.cuda.is_available() else "cpu"


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
        # Why Smooth?
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


def get_vocab(vocab_file_path):
    v2id = {}
    id2v = {}
    with open(VOCAB_PATH, "r", encoding="utf-8") as fr:
        lines = fr.readlines()
        for i, line in enumerate(lines):
            vocab = line[:-1]
            v2id[vocab] = i
            id2v[i] = vocab
    return v2id, id2v


def process_data(source_file_path, target_file_path, word_to_ids):
    dataset = []
    pickle_file_name = source_file_path.split('/')[-1].split('.')[0]
    pickle_file_path = DATA_PATH + pickle_file_name + "_cache"

    if os.path.exists(pickle_file_path):
        with open(pickle_file_path, "rb") as fr:
            dataset = pickle.load(fr)
            return dataset

    with open(source_file_path, "r") as source_file_reader, open(target_file_path, "r") as target_file_reader:
        src_docs = source_file_reader.readlines()
        tgt_docs = target_file_reader.readlines()
        assert len(src_docs) == len(tgt_docs)

        for i in trange(len(src_docs)):
            src_doc = src_docs[i][:-1]
            tgt_doc = tgt_docs[i][:-1]

            src_sentences = src_doc.split("</s>")
            tgt_sentences = tgt_doc.split("</s>")
            for (src_sentence, tgt_sentence) in zip(src_sentences, tgt_sentences):
                encoder_tokens = src_sentence.split()
                decoder_tokens = tgt_sentence.split()

                # truncate if too much token
                # if len(encoder_tokens) > MAX_LENGTH - 2:
                #     encoder_tokens = encoder_tokens[:MAX_LENGTH - 2]
                # if len(decoder_tokens) > MAX_LENGTH - 2:
                #     decoder_tokens = decoder_tokens[:MAX_LENGTH - 2]
                # add [cls] and [sep]
                encoder_tokens = ["[CLS]"] + encoder_tokens + ["[SEP]"]
                decoder_tokens = ["[CLS]"] + decoder_tokens + ["[SEP]"]

                encoder_input_id = []
                decoder_input_id = []

                for token in encoder_tokens:
                    if token not in word_to_ids:
                        print(f"[UNK TOKEN] : {token}")
                        continue
                    encoder_input_id.append(word_to_ids[token])

                for token in decoder_tokens:
                    if token not in word_to_ids:
                        print(f"[UNK TOKEN] : {token}")
                        continue
                    decoder_input_id.append(word_to_ids[token])

                # TODO: the last token not need to be input into model
                decoder_input_id = decoder_input_id[1:]

                encoder_attention_mask = [1 for _ in encoder_input_id]
                decoder_attention_mask = [1 for _ in decoder_input_id]
                # TODO: the first token of decoder don't need to be predicted
                # label_id = decoder_input_id[1:]
                label_id = decoder_input_id[:]

                if len(encoder_input_id) > MAX_LENGTH or len(decoder_input_id) > MAX_LENGTH:
                    continue

                while len(encoder_input_id) < MAX_LENGTH:
                    # encoder_tokens.append("[PAD]")
                    encoder_input_id.append(0)
                    encoder_attention_mask.append(0)
                while len(decoder_input_id) < MAX_LENGTH:
                    # decoder_tokens.append("[PAD]")
                    decoder_input_id.append(0)
                    decoder_attention_mask.append(0)
                    # TODO? meaning end?
                    label_id.append(-100)
                dataset.append({
                    "encoder_input_id": np.array(encoder_input_id),
                    "encoder_attention_mask": np.array(encoder_attention_mask),
                    "decoder_input_id": np.array(decoder_input_id),
                    "decoder_attention_mask": np.array(decoder_attention_mask),
                    "label_id": np.array(label_id)
                })
    with open(pickle_file_path, 'wb') as fw:
        pickle.dump(dataset, fw)
    return dataset


class MTDataset(Dataset):
    def __init__(self, ds):
        self.dataset = ds

    def __getitem__(self, item):
        return self.dataset[item]

    def __len__(self):
        return len(self.dataset)


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


def eval_the_model(model, test_data_loader, ids_to_words):
    END_OF_SENTENCE = 2
    START_OF_SENTENCE = 1
    h_l = []
    y_l = []
    the_test_index= 0
    with torch.no_grad():
        model.eval()

        for batch in tqdm(test_data_loader):
            encoder_input_id = batch["encoder_input_id"].to(device)
            encoder_attention_mask = batch["encoder_attention_mask"].to(device)
            decoder_input_id = batch["decoder_input_id"].to(device)
            decoder_attention_mask = batch["decoder_attention_mask"].to(device)
            label_id = batch["label_id"].to(device)

            outputs = model.generate(
                input_ids=encoder_input_id,
                attention_mask=encoder_attention_mask,
                bos_token_id=START_OF_SENTENCE,
                num_return_sequences=1,
                num_beams=5,
                max_length=MAX_LENGTH,
                use_cache=True
            )
            outputs = outputs.to("cpu").tolist()
            assert len(outputs) == len(label_id)

            for i in range(len(outputs)):
                predict = outputs[i]
                if END_OF_SENTENCE in predict:
                    index = predict.index(END_OF_SENTENCE)
                    predict = predict[:index]
                predict_tokens = []
                for _id in predict[1:]:
                    predict_tokens.append(ids_to_words[_id])
                predict_str = " ".join(predict_tokens)
                    # TODO: what?
                predict_str = predict_str.replace("@@", "")

                target = label_id[i]
                target = target.to("cpu").tolist()
                index = target.index(END_OF_SENTENCE)
                target = target[:index]
                target_tokens = []
                for _id in target:
                    target_tokens.append(ids_to_words[_id])
                target_str = " ".join(target_tokens)
                target_str = target_str.replace("@@", "")

                h_l.append(predict_str.split())
                y_l.append(target_str.split())
            the_test_index += 1
            if the_test_index == 5:
                break
    model.train()
    score = float(compute_bleu(y_l, h_l, 4, False)[0])
    print(f"bleu score: {score}")


words_to_ids, ids_to_words = get_vocab(VOCAB_PATH)
train_dataset = process_data(SRC_TRAIN_PATH, TGT_TRAIN_PATH, words_to_ids)
train_data_loader = DataLoader(dataset=MTDataset(train_dataset), shuffle=True, batch_size=BATCH_SIZE, pin_memory=True)
test_dataset = process_data(SRC_TEST_PATH, TGT_TEST_PATH, words_to_ids)
test_data_loader = DataLoader(dataset=MTDataset(test_dataset), shuffle=True, batch_size=BATCH_SIZE, pin_memory=True)
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

# TODO Beta?
optimizer = AdamW(model.parameters(), lr=LR, betas=(0.9, 0.98))
scheduler = ReverseSqrtScheduler(optimizer, [LR], WARMUP_STEPS)
for epoch in range(0, 100):
    model.train()
    total_loss = 0
    update_step = 0

    for batch in tqdm(train_data_loader):
        encoder_input_id = batch["encoder_input_id"].to(device)
        encoder_attention_mask = batch["encoder_attention_mask"].to(device)
        decoder_input_id = batch["decoder_input_id"].to(device)
        decoder_attention_mask = batch["decoder_attention_mask"].to(device)
        label_id = batch["label_id"].to(device)

        # print(encoder_input_id, encoder_input_id.shape)
        # print(encoder_attention_mask, encoder_attention_mask.shape)
        # print(decoder_input_id,decoder_input_id.shape)
        # print(decoder_attention_mask,decoder_attention_mask.shape)
        # print(label_id,label_id.shape)
        assert encoder_input_id.shape[-1] == 128
        assert encoder_attention_mask.shape[-1] == 128
        assert decoder_input_id.shape[-1] == 128
        assert decoder_attention_mask.shape[-1] == 128
        assert label_id.shape[-1] == 128

        loss = model(
            input_ids=encoder_input_id,
            attention_mask=encoder_attention_mask,
            decoder_input_ids=decoder_input_id,
            decoder_attention_mask=decoder_attention_mask,
            labels=label_id
        )["loss"]
        total_loss += loss.item()
        loss.backward()
        scheduler.step_and_update_lr()
        scheduler.zero_grad()
        update_step += 1
        if update_step % PRINT_STEP == 0:
            print(f"loss: {total_loss / update_step}")
        if update_step % EVAL_STEP == 0:
            eval_the_model(model, test_data_loader, ids_to_words)


    eval_the_model(model, test_data_loader, ids_to_words)


