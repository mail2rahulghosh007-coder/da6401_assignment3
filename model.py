"""
Transformer for German → English Machine Translation
DA6401 Assignment 3
"""

import math, os, subprocess, sys
import torch
import torch.nn as nn
import torch.nn.functional as F

# ─────────────────────────────────────────────────────────────────────────────
# FILL IN YOUR GOOGLE DRIVE FILE ID FOR best_model.pt BELOW
# ─────────────────────────────────────────────────────────────────────────────
_BEST_MODEL_GDRIVE_ID = "1qm_baz2s1NNd2OJc375AAWGMQLyoU5UQ"


def _gdrive_download(gdrive_id, dest_path):
    if os.path.exists(dest_path):
        return
    if gdrive_id.startswith("PASTE"):
        return
    try:
        import gdown
        print(f"Downloading {os.path.basename(dest_path)} from Google Drive …")
        gdown.download(id=gdrive_id, output=dest_path, quiet=False)
    except Exception as e:
        print(f"Warning: download failed: {e}")


def _ensure_spacy():
    """Install spacy models if missing. Called in __init__ before any timer."""
    import spacy as _spacy
    for model_name in ["de_core_news_sm", "en_core_web_sm"]:
        try:
            _spacy.load(model_name)
        except OSError:
            print(f"Installing {model_name}...")
            subprocess.check_call(
                [sys.executable, "-m", "spacy", "download", model_name],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            )


# ─────────────────────────────────────────────────────────────────────────────
# 1. Scaled Dot-Product Attention
# ─────────────────────────────────────────────────────────────────────────────
def scaled_dot_product_attention(Q, K, V, mask=None):
    d_k    = Q.size(-1)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask, float('-inf'))
    attn = torch.nan_to_num(F.softmax(scores, dim=-1), nan=0.0)
    return torch.matmul(attn, V), attn


# ─────────────────────────────────────────────────────────────────────────────
# 2. Multi-Head Attention  — returns TENSOR only (not tuple)
# ─────────────────────────────────────────────────────────────────────────────
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0
        self.num_heads    = num_heads
        self.d_k          = d_model // num_heads
        self.W_q          = nn.Linear(d_model, d_model, bias=False)
        self.W_k          = nn.Linear(d_model, d_model, bias=False)
        self.W_v          = nn.Linear(d_model, d_model, bias=False)
        self.W_o          = nn.Linear(d_model, d_model, bias=False)
        self.attn_weights = None  # stored for visualisation

    def split_heads(self, x):
        B, S, _ = x.size()
        return x.view(B, S, self.num_heads, self.d_k).transpose(1, 2)

    def forward(self, query, key, value, mask=None):
        Q = self.split_heads(self.W_q(query))
        K = self.split_heads(self.W_k(key))
        V = self.split_heads(self.W_v(value))
        out, self.attn_weights = scaled_dot_product_attention(Q, K, V, mask)
        B, H, S, d_k = out.size()
        out = out.transpose(1, 2).contiguous().view(B, S, H * d_k)
        return self.W_o(out)   # TENSOR only


# ─────────────────────────────────────────────────────────────────────────────
# 3. Feed-Forward
# ─────────────────────────────────────────────────────────────────────────────
class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )
    def forward(self, x): return self.net(x)


# ─────────────────────────────────────────────────────────────────────────────
# 4. Sinusoidal Positional Encoding
# ─────────────────────────────────────────────────────────────────────────────
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe       = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return self.dropout(x + self.pe[:, :x.size(1), :])


# ─────────────────────────────────────────────────────────────────────────────
# 5. Encoder Layer
# ─────────────────────────────────────────────────────────────────────────────
class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.ff    = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.drop1 = nn.Dropout(dropout)
        self.drop2 = nn.Dropout(dropout)

    def forward(self, x, src_mask=None):
        x = self.norm1(x + self.drop1(self.self_attn(x, x, x, src_mask)))
        x = self.norm2(x + self.drop2(self.ff(x)))
        return x


# ─────────────────────────────────────────────────────────────────────────────
# 6. Decoder Layer
# ─────────────────────────────────────────────────────────────────────────────
class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn  = MultiHeadAttention(d_model, num_heads)
        self.cross_attn = MultiHeadAttention(d_model, num_heads)
        self.ff    = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.drop1 = nn.Dropout(dropout)
        self.drop2 = nn.Dropout(dropout)
        self.drop3 = nn.Dropout(dropout)

    def forward(self, x, enc_out, tgt_mask=None, src_mask=None):
        x = self.norm1(x + self.drop1(self.self_attn(x, x, x, tgt_mask)))
        x = self.norm2(x + self.drop2(self.cross_attn(x, enc_out, enc_out, src_mask)))
        x = self.norm3(x + self.drop3(self.ff(x)))
        return x


# ─────────────────────────────────────────────────────────────────────────────
# 7. Encoder Stack
# ─────────────────────────────────────────────────────────────────────────────
class Encoder(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, d_ff, num_layers, dropout=0.1, max_len=5000):
        super().__init__()
        self.embed  = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pe     = PositionalEncoding(d_model, max_len, dropout)
        self.scale  = math.sqrt(d_model)
        self.layers = nn.ModuleList(
            [EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)]
        )

    def forward(self, src, src_mask=None):
        x = self.pe(self.embed(src) * self.scale)
        for layer in self.layers:
            x = layer(x, src_mask)
        return x


# ─────────────────────────────────────────────────────────────────────────────
# 8. Decoder Stack
# ─────────────────────────────────────────────────────────────────────────────
class Decoder(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, d_ff, num_layers, dropout=0.1, max_len=5000):
        super().__init__()
        self.embed  = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pe     = PositionalEncoding(d_model, max_len, dropout)
        self.scale  = math.sqrt(d_model)
        self.layers = nn.ModuleList(
            [DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)]
        )

    def forward(self, tgt, enc_out, tgt_mask=None, src_mask=None):
        x = self.pe(self.embed(tgt) * self.scale)
        for layer in self.layers:
            x = layer(x, enc_out, tgt_mask, src_mask)
        return x


# ─────────────────────────────────────────────────────────────────────────────
# 9. Mask helpers
# ─────────────────────────────────────────────────────────────────────────────
def make_pad_mask(seq, pad_idx=0):
    return (seq == pad_idx).unsqueeze(1).unsqueeze(2)

def make_causal_mask(size, device):
    return torch.triu(
        torch.ones(size, size, device=device), diagonal=1
    ).bool().unsqueeze(0).unsqueeze(0)


# ─────────────────────────────────────────────────────────────────────────────
# 10. Full Transformer
# ─────────────────────────────────────────────────────────────────────────────
class Transformer(nn.Module):
    def __init__(
        self,
        src_vocab_size  = None,
        tgt_vocab_size  = None,
        d_model         = 256,
        num_heads       = 8,
        d_ff            = 512,
        num_layers      = 3,
        dropout         = 0.1,
        max_len         = 256,
        pad_idx         = 0,
        model_save_path = "best_model.pt",
    ):
        super().__init__()
        self.device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.pad_idx = pad_idx
        self.max_len = max_len

        # ── Install spacy models NOW (before any 3-sec timer starts) ─────────
        _ensure_spacy()
        import spacy as _spacy
        self._spacy_de = _spacy.load("de_core_news_sm")
        self._spacy_en = _spacy.load("en_core_web_sm")

        # ── Locate & download best_model.pt ──────────────────────────────────
        _here        = os.path.dirname(os.path.abspath(__file__))
        abs_ckpt     = os.path.join(_here, model_save_path)
        if not os.path.exists(abs_ckpt):
            abs_ckpt = model_save_path
        _gdrive_download(_BEST_MODEL_GDRIVE_ID, abs_ckpt)

        # ── Load checkpoint ───────────────────────────────────────────────────
        ckpt = None
        if os.path.exists(abs_ckpt):
            ckpt = torch.load(abs_ckpt, map_location="cpu", weights_only=False)

        # ── Load vocab (from checkpoint, then separate files) ─────────────────
        self.src_vocab = None
        self.tgt_vocab = None
        if isinstance(ckpt, dict):
            self.src_vocab = ckpt.get("src_vocab", None)
            self.tgt_vocab = ckpt.get("tgt_vocab", None)

        for attr, fname in [("src_vocab", "src_vocab.pt"), ("tgt_vocab", "tgt_vocab.pt")]:
            if getattr(self, attr) is None:
                for path in [os.path.join(_here, fname), fname]:
                    if os.path.exists(path):
                        setattr(self, attr, torch.load(path, weights_only=False))
                        break

        # ── Resolve vocab sizes ───────────────────────────────────────────────
        if src_vocab_size is None and self.src_vocab is not None:
            src_vocab_size = len(self.src_vocab)
        if tgt_vocab_size is None and self.tgt_vocab is not None:
            tgt_vocab_size = len(self.tgt_vocab)
        src_vocab_size = src_vocab_size or 8000
        tgt_vocab_size = tgt_vocab_size or 8000

        # ── Build layers ──────────────────────────────────────────────────────
        self.encoder = Encoder(src_vocab_size, d_model, num_heads, d_ff, num_layers, dropout, max_len)
        self.decoder = Decoder(tgt_vocab_size, d_model, num_heads, d_ff, num_layers, dropout, max_len)
        self.fc_out  = nn.Linear(d_model, tgt_vocab_size)
        self._init_weights()

        # ── Load weights ──────────────────────────────────────────────────────
        if ckpt is not None:
            state = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
            self.load_state_dict(state)
            print(f"Loaded weights from {abs_ckpt}")

        self.to(self.device)

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    # ── Tokenizers using spacy (same as training) ─────────────────────────────
    def tokenize_de(self, text):
        return [tok.text.lower() for tok in self._spacy_de.tokenizer(text)]

    def tokenize_en(self, text):
        return [tok.text.lower() for tok in self._spacy_en.tokenizer(text)]

    def forward(self, src, tgt):
        src_mask = make_pad_mask(src, self.pad_idx).to(src.device)
        tgt_mask = (
            make_pad_mask(tgt, self.pad_idx).to(tgt.device)
            | make_causal_mask(tgt.size(1), tgt.device)
        )
        enc_out = self.encoder(src, src_mask)
        dec_out = self.decoder(tgt, enc_out, tgt_mask, src_mask)
        return self.fc_out(dec_out)

    def greedy_decode(self, src_tensor, max_len=None):
        if max_len is None:
            max_len = self.max_len
        self.eval()
        with torch.no_grad():
            src_mask = make_pad_mask(src_tensor, self.pad_idx).to(self.device)
            enc_out  = self.encoder(src_tensor, src_mask)
            bos      = self.tgt_vocab["<bos>"]
            eos      = self.tgt_vocab["<eos>"]
            ids      = [bos]
            for _ in range(max_len):
                t   = torch.tensor(ids, dtype=torch.long, device=self.device).unsqueeze(0)
                tm  = (
                    make_pad_mask(t, self.pad_idx).to(self.device)
                    | make_causal_mask(t.size(1), self.device)
                )
                nxt = self.fc_out(
                    self.decoder(t, enc_out, tm, src_mask)
                )[0, -1, :].argmax(-1).item()
                ids.append(nxt)
                if nxt == eos:
                    break
        return ids[1:]

    def infer(self, german_sentence: str) -> str:
        assert self.src_vocab is not None, "src_vocab not loaded"
        assert self.tgt_vocab is not None, "tgt_vocab not loaded"
        self.eval()
        tokens = ["<bos>"] + self.tokenize_de(german_sentence) + ["<eos>"]
        src_t  = torch.tensor(
            [self.src_vocab.get(t, self.src_vocab["<unk>"]) for t in tokens],
            dtype=torch.long, device=self.device
        ).unsqueeze(0)
        i2w   = {v: k for k, v in self.tgt_vocab.items()}
        words = []
        for idx in self.greedy_decode(src_t):
            tok = i2w.get(idx, "<unk>")
            if tok in ("<eos>", "<pad>"):
                break
            if tok not in ("<bos>",):
                words.append(tok)
        # Detokenize: attach punctuation to previous word (sacrebleu style)
        result = []
        for w in words:
            if w in (".", ",", "!", "?", ";", ":", "'s", "n't", "'re", "'ve", "'ll", "'d", "'m") \
                    and result:
                result[-1] = result[-1] + w
            else:
                result.append(w)
        return " ".join(result)
