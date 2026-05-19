DA6401 Assignment 3 — Transformer for German→English Machine Translation
Implementation of the Transformer architecture from "Attention Is All You Need" (Vaswani et al., 2017) trained on the Multi30k dataset for German→English neural machine translation.

Model Architecture
The model follows the original Transformer paper with the following configuration:
ParameterValued_model256num_heads8d_ff512num_layers3 (encoder + decoder each)d_k32 (d_model / num_heads)dropout0.1max_len256
Key implementation details:

Scaled dot-product attention: Attention(Q,K,V) = softmax(QKᵀ/√d_k)V
Multi-head attention implemented from scratch without torch.nn.MultiheadAttention
Sinusoidal positional encoding registered as a non-trainable buffer
Post-LayerNorm after each Add & Norm step
Xavier uniform initialization for all weight matrices
Padding mask and causal (look-ahead) mask for the decoder
Greedy decoding at inference time


Training Configuration
ParameterValueOptimizerAdam (β1=0.9, β2=0.98, ε=1e-9)Learning rate scheduleNoam (warmup_steps=4000)Label smoothingε = 0.1Batch size128Epochs30Gradient clipping1.0Min token frequency2
The best checkpoint is saved based on validation BLEU score, not validation loss.

Dataset

Dataset: Multi30k (bentrevett/multi30k via HuggingFace datasets)
Language pair: German → English
Split: 29,000 train / 1,014 validation / 1,000 test sentence pairs
Tokenization: spaCy (de_core_news_sm for German, en_core_web_sm for English)
Vocabulary: built from training set only with min_freq=2


Results
MetricValueTest BLEU40.80Best Validation BLEU~37–38

Project Structure
model.py          # Full Transformer architecture (MHA, encoder, decoder, PE, masks)
train.py          # Training loop and all 5 experiment runners
dataset.py        # Multi30k data loading, vocabulary building, DataLoader
lr_scheduler.py   # Noam learning rate scheduler
best_model.pt     # Best checkpoint (includes model weights, src_vocab, tgt_vocab)

Experiments
Five experiments were conducted and logged to Weights & Biases:
2.1 Noam Scheduler vs Fixed LR — Noam scheduler with warmup converges faster and to a lower final loss compared to a fixed learning rate of 1e-4. The warmup phase prevents early divergence in the self-attention layers.
2.2 Scaling Factor Ablation — Without the 1/√d_k scaling factor, W_Q and W_K gradient norms are 2–2.5× larger and significantly more unstable (spiky), directly demonstrating the vanishing gradient problem described in Section 3.2.1 of the paper. Final training loss is 4.2 without scaling vs 3.9 with scaling.
2.3 Attention Head Visualization — Attention weights from the last encoder layer are logged as heatmaps for all 8 heads. Different heads specialize in local self-attention, noun phrase grouping, long-range dependencies, and next-token prediction. Partial head redundancy is observed between heads 1 and 7.
2.4 Sinusoidal vs Learned PE — Sinusoidal positional encoding achieves ~36 test BLEU compared to ~34 for learned nn.Embedding — a 2 point gap despite learned PE achieving slightly lower validation loss, indicating overfitting.
2.5 Label Smoothing — Training with ε=0.1 reduces prediction confidence (0.58 vs 0.62) compared to standard cross-entropy, acting as a regularizer that prevents overconfidence and improves generalization.
