"""
dataset.py  ─  Multi30k German→English dataset utilities
"""

import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from datasets import load_dataset
from collections import Counter
import spacy, os


# ── tokenizers ─────────────────────────────────────────────────────
def get_tokenizers():
    try:
        spacy_de = spacy.load("de_core_news_sm")
    except OSError:
        os.system("python -m spacy download de_core_news_sm")
        spacy_de = spacy.load("de_core_news_sm")
    try:
        spacy_en = spacy.load("en_core_web_sm")
    except OSError:
        os.system("python -m spacy download en_core_web_sm")
        spacy_en = spacy.load("en_core_web_sm")

    tokenize_de = lambda text: [tok.text.lower() for tok in spacy_de.tokenizer(text)]
    tokenize_en = lambda text: [tok.text.lower() for tok in spacy_en.tokenizer(text)]
    return tokenize_de, tokenize_en


# ── vocabulary ─────────────────────────────────────────────────────
SPECIAL_TOKENS = ["<pad>", "<bos>", "<eos>", "<unk>"]
PAD_IDX, BOS_IDX, EOS_IDX, UNK_IDX = 0, 1, 2, 3


def build_vocab(sentences, tokenize_fn, min_freq=2):
    counter = Counter()
    for sent in sentences:
        counter.update(tokenize_fn(sent))
    vocab = {tok: idx for idx, tok in enumerate(SPECIAL_TOKENS)}
    for word, freq in counter.items():
        if freq >= min_freq:
            vocab[word] = len(vocab)
    return vocab


def numericalize(tokens, vocab):
    return [BOS_IDX] + [vocab.get(t, UNK_IDX) for t in tokens] + [EOS_IDX]


# ── Dataset ────────────────────────────────────────────────────────
class TranslationDataset(Dataset):
    def __init__(self, data, src_vocab, tgt_vocab, tokenize_src, tokenize_tgt):
        self.data        = data
        self.src_vocab   = src_vocab
        self.tgt_vocab   = tgt_vocab
        self.tok_src     = tokenize_src
        self.tok_tgt     = tokenize_tgt

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        src = self.data[idx]["de"]
        tgt = self.data[idx]["en"]
        src_ids = numericalize(self.tok_src(src), self.src_vocab)
        tgt_ids = numericalize(self.tok_tgt(tgt), self.tgt_vocab)
        return torch.tensor(src_ids, dtype=torch.long), \
               torch.tensor(tgt_ids, dtype=torch.long)


def collate_fn(batch):
    src_batch, tgt_batch = zip(*batch)
    src_pad = pad_sequence(src_batch, batch_first=True, padding_value=PAD_IDX)
    tgt_pad = pad_sequence(tgt_batch, batch_first=True, padding_value=PAD_IDX)
    return src_pad, tgt_pad


# ── main loader ────────────────────────────────────────────────────
def get_dataloaders(batch_size=128, min_freq=2):
    raw = load_dataset("bentrevett/multi30k")
    tokenize_de, tokenize_en = get_tokenizers()

    train_data = raw["train"]
    val_data   = raw["validation"]
    test_data  = raw["test"]

    src_vocab = build_vocab(train_data["de"], tokenize_de, min_freq)
    tgt_vocab = build_vocab(train_data["en"], tokenize_en, min_freq)

    # save for Transformer.init to load
    torch.save(src_vocab, "src_vocab.pt")
    torch.save(tgt_vocab, "tgt_vocab.pt")

    make_ds = lambda split: TranslationDataset(split, src_vocab, tgt_vocab, tokenize_de, tokenize_en)

    train_loader = DataLoader(make_ds(train_data), batch_size=batch_size,
                              shuffle=True,  collate_fn=collate_fn, num_workers=2)
    val_loader   = DataLoader(make_ds(val_data),   batch_size=batch_size,
                              shuffle=False, collate_fn=collate_fn, num_workers=2)
    test_loader  = DataLoader(make_ds(test_data),  batch_size=batch_size,
                              shuffle=False, collate_fn=collate_fn, num_workers=2)

    return train_loader, val_loader, test_loader, src_vocab, tgt_vocab
