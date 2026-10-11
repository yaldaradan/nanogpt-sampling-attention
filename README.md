# nanoGPT Sampling & Attention Experiments

A character-level GPT trained on Tiny Shakespeare, built on [Andrej Karpathy's nanoGPT](https://github.com/karpathy/nanoGPT). I extended it with a multi-head self-attention module that has a causal toggle, attention-map visualizations, and greedy, top-k and top-p (nucleus) sampling. I then compared how temperature and the sampling strategy change the generated text.

## What I added

- **Self-attention with a causal toggle**: `SelfAttentionModule` takes a `causal` flag. When it is on, an upper-triangular mask blocks each token from attending to future tokens. When it is off, the module computes full bidirectional attention. It returns the attention weights along with the output so they can be plotted.
- **Attention-map visualization**: plots the attention matrix for each head as a query × key heatmap, in both causal and non-causal mode, so the effect of the mask is visible.
- **Sampling strategies in `GPT.generate`**:
  - `greedy=True`: always picks the most likely next character.
  - `top_k`: samples only from the k most likely characters.
  - `top_p`: nucleus sampling. Keeps the smallest set of characters whose cumulative probability exceeds p, always keeping at least the top one.
  - All three work with `temperature` scaling.
- **Sampling comparison**: generates text from the prompt `"Juliet: "` for every combination of temperature (0.7, 1.0, 1.3) and strategy (greedy, top-k=50, top-p=0.9).

From nanoGPT (adapted): the GPT model (attention, MLP, LayerNorm, blocks), the char-level tokenizer, and the training loop with AdamW and cosine learning-rate decay.

## Setup

```bash
pip install torch numpy matplotlib gdown
```

A GPU is recommended for training. To train on a Mac, set `device = 'mps'` in the training cell; to train on a CPU, set `device = 'cpu'`.

## Run

Open `nanogpt_sampling_attention.ipynb` and run the cells in order:

1. **Attention module and plots**: builds the standalone attention module and plots per-head attention maps, causal and non-causal.
2. **Data**: downloads Tiny Shakespeare and builds the character vocabulary (90/10 train/val split).
3. **Model**: defines the GPT.
4. **Train**: 3 layers, 8 heads, 768-dim embeddings, block size 128, 10,000 iterations.
5. **Sample**: runs the temperature × strategy comparison.


## Credits

Model and training code adapted from [nanoGPT](https://github.com/karpathy/nanoGPT) by Andrej Karpathy (MIT License).
