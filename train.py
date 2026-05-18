"""
train.py  ─  DA6401 Assignment 3 Training Script
"""

import os, math, copy
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
import sacrebleu as _sacrebleu
from tqdm import tqdm

from dataset      import get_dataloaders, PAD_IDX, BOS_IDX, EOS_IDX, UNK_IDX
from model        import Transformer, Encoder, Decoder, make_pad_mask, make_causal_mask
from lr_scheduler import NoamScheduler


# ── Config ────────────────────────────────────────────────────────────────────
CFG = dict(
    d_model      = 256,
    num_heads    = 8,
    d_ff         = 512,
    num_layers   = 3,
    dropout      = 0.1,
    max_len      = 256,
    batch_size   = 128,
    warmup_steps = 4000,
    epochs       = 30,
    clip         = 1.0,
    label_smooth = 0.1,
    min_freq     = 2,
)


# ── Label Smoothing Loss ──────────────────────────────────────────────────────
class LabelSmoothedCE(nn.Module):
    def __init__(self, vocab_size, eps=0.1, pad_idx=0):
        super().__init__()
        self.eps        = eps
        self.pad_idx    = pad_idx
        self.vocab_size = vocab_size

    def forward(self, logits, target):
        V         = logits.size(-1)
        log_probs = torch.log_softmax(logits, dim=-1)
        with torch.no_grad():
            smooth = torch.full_like(log_probs, self.eps / (V - 2))
            smooth[:, self.pad_idx] = 0
            smooth.scatter_(1, target.unsqueeze(1), 1.0 - self.eps)
            pad_mask = target.eq(self.pad_idx)
            smooth[pad_mask] = 0
        loss     = -(smooth * log_probs).sum(dim=-1)
        n_tokens = (~pad_mask).sum()
        return loss.sum() / n_tokens


# ── Learned PE (experiment 2.4) ───────────────────────────────────────────────
class _LearnedPE(nn.Module):
    def __init__(self, d_model, max_len, dropout=0.1):
        super().__init__()
        self.embed   = nn.Embedding(max_len, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        pos = torch.arange(x.size(1), device=x.device).unsqueeze(0)
        return self.dropout(x + self.embed(pos))


# ── Build model ───────────────────────────────────────────────────────────────
def build_model(src_vocab, tgt_vocab, cfg, device, learned_pos=False):
    """Build Transformer directly without triggering __init__ download logic."""
    import spacy as _spacy, types

    model = Transformer.__new__(Transformer)
    nn.Module.__init__(model)
    model.device    = device
    model.pad_idx   = PAD_IDX
    model.max_len   = cfg["max_len"]
    model.src_vocab = src_vocab
    model.tgt_vocab = tgt_vocab
    model._spacy_de = None
    model._spacy_en = None

    try:
        model._spacy_de = _spacy.load("de_core_news_sm")
        model._spacy_en = _spacy.load("en_core_web_sm")
    except OSError:
        pass   # Kaggle has these installed; fine to skip if missing

    model.encoder = Encoder(len(src_vocab), cfg["d_model"], cfg["num_heads"],
                            cfg["d_ff"], cfg["num_layers"], cfg["dropout"], cfg["max_len"])
    model.decoder = Decoder(len(tgt_vocab), cfg["d_model"], cfg["num_heads"],
                            cfg["d_ff"], cfg["num_layers"], cfg["dropout"], cfg["max_len"])
    model.fc_out  = nn.Linear(cfg["d_model"], len(tgt_vocab))

    if learned_pos:
        for part in [model.encoder, model.decoder]:
            part.pe = _LearnedPE(cfg["d_model"], cfg["max_len"], cfg["dropout"])

    # Bind methods
    for name in ["_init_weights", "forward", "greedy_decode", "infer"]:
        setattr(model, name, types.MethodType(getattr(Transformer, name), model))

    model.tokenize_de = lambda t: t.lower().split()
    model.tokenize_en = lambda t: t.lower().split()
    model._init_weights()
    return model.to(device)


# ── BLEU evaluation ───────────────────────────────────────────────────────────
def evaluate_bleu(model, loader, tgt_vocab, device, max_samples=500):
    idx2word   = {v: k for k, v in tgt_vocab.items()}
    hypotheses = []
    references = []
    model.eval()
    n = 0
    with torch.no_grad():
        for src, tgt in loader:
            src = src.to(device)
            for i in range(src.size(0)):
                if n >= max_samples: break
                pred_ids    = model.greedy_decode(src[i:i+1])
                pred_tokens = [idx2word.get(t, "<unk>") for t in pred_ids
                               if idx2word.get(t, "<unk>") not in ("<eos>", "<pad>", "<bos>")]
                ref_tokens  = [idx2word.get(t.item(), "<unk>") for t in tgt[i]
                               if idx2word.get(t.item(), "<unk>") not in ("<eos>", "<pad>", "<bos>", "<unk>")]
                hypotheses.append(" ".join(pred_tokens))
                references.append(" ".join(ref_tokens))
                n += 1
            if n >= max_samples: break
    return _sacrebleu.corpus_bleu(hypotheses, [references]).score


# ── One training epoch ────────────────────────────────────────────────────────
def train_epoch(model, loader, optimizer, criterion, scheduler, device, clip,
                log_grad_norm=False):
    model.train()
    total_loss, total_tok = 0, 0
    grad_norms = []
    for src, tgt in tqdm(loader, leave=False):
        src, tgt = src.to(device), tgt.to(device)
        tgt_in   = tgt[:, :-1]
        tgt_out  = tgt[:, 1:]
        logits   = model(src, tgt_in)
        B, T, V  = logits.size()
        loss     = criterion(logits.reshape(B*T, V), tgt_out.reshape(B*T))
        optimizer.zero_grad()
        loss.backward()
        if clip:
            nn.utils.clip_grad_norm_(model.parameters(), clip)
        if log_grad_norm:
            norm = sum(p.grad.norm().item()**2 for p in model.parameters() if p.grad is not None) ** 0.5
            grad_norms.append(norm)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        n_tok      = tgt_out.ne(PAD_IDX).sum().item()
        total_loss += loss.item() * n_tok
        total_tok  += n_tok
    return total_loss / max(total_tok, 1), grad_norms


def val_epoch(model, loader, criterion, device):
    model.eval()
    total_loss, total_tok = 0, 0
    with torch.no_grad():
        for src, tgt in loader:
            src, tgt = src.to(device), tgt.to(device)
            tgt_in   = tgt[:, :-1]
            tgt_out  = tgt[:, 1:]
            logits   = model(src, tgt_in)
            B, T, V  = logits.size()
            loss     = criterion(logits.reshape(B*T, V), tgt_out.reshape(B*T))
            n_tok    = tgt_out.ne(PAD_IDX).sum().item()
            total_loss += loss.item() * n_tok
            total_tok  += n_tok
    return total_loss / max(total_tok, 1)


# ── Main training loop — saves checkpoint on BEST BLEU ───────────────────────
def train_model(model, train_loader, val_loader, cfg, device,
                run_name="run", use_noam=True, label_smooth=0.1,
                log_grad_norm=False, epochs=None, save_path="best_model.pt"):

    epochs    = epochs or cfg["epochs"]
    criterion = LabelSmoothedCE(len(model.tgt_vocab), eps=label_smooth, pad_idx=PAD_IDX)
    optimizer = optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.98), eps=1e-9)

    if use_noam:
        scheduler = NoamScheduler(optimizer, d_model=cfg["d_model"], warmup_steps=cfg["warmup_steps"])
    else:
        scheduler = None
        for pg in optimizer.param_groups:
            pg["lr"] = 1e-4

    best_bleu = -1.0   # ← save on BEST BLEU, not best loss

    for epoch in range(1, epochs + 1):
        tr_loss, gnorms = train_epoch(model, train_loader, optimizer, criterion,
                                      scheduler, device, cfg["clip"], log_grad_norm)
        val_loss = val_epoch(model, val_loader, criterion, device)
        val_bleu = evaluate_bleu(model, val_loader, model.tgt_vocab, device, max_samples=500)
        cur_lr   = optimizer.param_groups[0]["lr"]

        log = {"epoch": epoch, "train_loss": tr_loss, "val_loss": val_loss,
               "val_bleu": val_bleu, "lr": cur_lr}
        if gnorms:
            log["grad_norm_mean"] = sum(gnorms) / len(gnorms)
        wandb.log(log)
        print(f"[{run_name}] Epoch {epoch:02d}  "
              f"train_loss={tr_loss:.4f}  val_loss={val_loss:.4f}  "
              f"val_bleu={val_bleu:.2f}  lr={cur_lr:.6f}")

        # Save on best BLEU (not best loss)
        if val_bleu > best_bleu:
            best_bleu = val_bleu
            torch.save({
                "model_state_dict": model.state_dict(),
                "src_vocab":        model.src_vocab,
                "tgt_vocab":        model.tgt_vocab,
                "cfg":              cfg,
                "epoch":            epoch,
                "val_bleu":         val_bleu,
            }, save_path)
            print(f"  ✅ Saved best model (BLEU={val_bleu:.2f})")

    # Restore best checkpoint
    ckpt = torch.load(save_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"Training done. Best BLEU = {best_bleu:.2f}")
    return model


# ── Experiment runners ────────────────────────────────────────────────────────
def exp_main(cfg, train_loader, val_loader, test_loader, src_vocab, tgt_vocab, device):
    wandb.init(project="da6401_a3", name="main_noam", config=cfg)
    model = build_model(src_vocab, tgt_vocab, cfg, device)
    model = train_model(model, train_loader, val_loader, cfg, device,
                        run_name="main", use_noam=True,
                        label_smooth=cfg["label_smooth"], save_path="best_model.pt")
    test_bleu = evaluate_bleu(model, test_loader, tgt_vocab, device, max_samples=1000)
    wandb.log({"test_bleu": test_bleu})
    print(f"Test BLEU: {test_bleu:.2f}")
    wandb.finish()


def exp_noam_vs_fixed(cfg, train_loader, val_loader, src_vocab, tgt_vocab, device):
    for use_noam, name in [(True, "noam"), (False, "fixed_lr_1e-4")]:
        wandb.init(project="da6401_a3", name=f"exp_noam_{name}", config=cfg)
        model = build_model(src_vocab, tgt_vocab, cfg, device)
        train_model(model, train_loader, val_loader, cfg, device,
                    run_name=name, use_noam=use_noam,
                    label_smooth=cfg["label_smooth"],
                    epochs=15,
                    save_path=f"model_{name}.pt")
        wandb.finish()


def exp_scaling_ablation(cfg, train_loader, val_loader, src_vocab, tgt_vocab, device):
    import model as model_module
    from model import scaled_dot_product_attention as orig_sdpa

    def sdpa_no_scale(Q, K, V, mask=None):
        import torch.nn.functional as F
        scores = torch.matmul(Q, K.transpose(-2, -1))   # no sqrt(d_k)
        if mask is not None:
            scores = scores.masked_fill(mask, float('-inf'))
        attn = torch.nan_to_num(F.softmax(scores, dim=-1), nan=0.0)
        return torch.matmul(attn, V), attn

    for use_scale, name in [(True, "with_scale"), (False, "no_scale")]:
        if not use_scale:
            model_module.scaled_dot_product_attention = sdpa_no_scale
        else:
            model_module.scaled_dot_product_attention = orig_sdpa
        wandb.init(project="da6401_a3", name=f"exp_scale_{name}",
                   config={**cfg, "scaling": use_scale})
        m = build_model(src_vocab, tgt_vocab, cfg, device)
        train_model(m, train_loader, val_loader, cfg, device,
                    run_name=name, use_noam=True,
                    label_smooth=cfg["label_smooth"],
                    log_grad_norm=True, epochs=5,
                    save_path=f"model_{name}.pt")
        model_module.scaled_dot_product_attention = orig_sdpa
        wandb.finish()


def exp_attn_rollout(cfg, val_loader, src_vocab, tgt_vocab, device):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    model = build_model(src_vocab, tgt_vocab, cfg, device)
    if os.path.exists("best_model.pt"):
        ckpt = torch.load("best_model.pt", map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    idx2src = {v: k for k, v in src_vocab.items()}
    src, _  = next(iter(val_loader))
    src     = src[:1].to(device)
    src_tokens = [idx2src.get(i.item(), "?") for i in src[0] if i.item() != PAD_IDX]

    with torch.no_grad():
        src_mask  = make_pad_mask(src, PAD_IDX).to(device)
        x         = model.encoder.pe(model.encoder.embed(src) * math.sqrt(cfg["d_model"]))
        last_attn = None
        for layer in model.encoder.layers:
            _ = layer.self_attn(x, x, x, src_mask)
            last_attn = layer.self_attn.attn_weights[0].cpu().numpy()
            x = layer.norm1(x + layer.drop1(layer.self_attn(x, x, x, src_mask)))
            x = layer.norm2(x + layer.drop2(layer.ff(x)))

    S = len(src_tokens)
    wandb.init(project="da6401_a3", name="attn_rollout", config=cfg)
    for h in range(last_attn.shape[0]):
        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(last_attn[h, :S, :S], cmap="viridis")
        ax.set_xticks(range(S)); ax.set_xticklabels(src_tokens, rotation=90, fontsize=8)
        ax.set_yticks(range(S)); ax.set_yticklabels(src_tokens, fontsize=8)
        ax.set_title(f"Encoder Last Layer — Head {h+1}")
        plt.colorbar(im); plt.tight_layout()
        wandb.log({f"attn_head_{h+1}": wandb.Image(fig)})
        plt.close(fig)
    wandb.finish()


def exp_pos_encoding(cfg, train_loader, val_loader, src_vocab, tgt_vocab, device):
    for learned, name in [(False, "sinusoidal"), (True, "learned_pe")]:
        wandb.init(project="da6401_a3", name=f"exp_pos_{name}", config=cfg)
        m = build_model(src_vocab, tgt_vocab, cfg, device, learned_pos=learned)
        train_model(m, train_loader, val_loader, cfg, device,
                    run_name=name, use_noam=True,
                    label_smooth=cfg["label_smooth"],
                    epochs=15,
                    save_path=f"model_{name}.pt")
        wandb.finish()


def exp_label_smoothing(cfg, train_loader, val_loader, src_vocab, tgt_vocab, device):
    for eps, name in [(0.1, "ls_0.1"), (0.0, "ls_0.0")]:
        wandb.init(project="da6401_a3", name=f"exp_ls_{name}",
                   config={**cfg, "label_smooth": eps})
        m = build_model(src_vocab, tgt_vocab, cfg, device)
        train_model(m, train_loader, val_loader, cfg, device,
                    run_name=name, use_noam=True,
                    label_smooth=eps,
                    epochs=15,
                    save_path=f"model_{name}.pt")
        wandb.finish()
