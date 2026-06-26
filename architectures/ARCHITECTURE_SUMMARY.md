# Architecture Summary - Data Flow Analysis

## Overview

All models follow this general pattern:
```
Input: (x_ts, x_static, observed_mask)
  x_ts: [B, T, C_ts] - time series features (already z-score normalized)
  x_static: [B, C_static] - static features (already z-score normalized)
  observed_mask: [B, T] - boolean mask for valid timesteps

Processing:
  1. Truncate to effective_seq_len = seq_len - max(lags_sequence)
  2. Architecture-specific temporal processing
  3. Pool over time dimension → [B, pooled_dim]
  4. Concatenate with static features → [B, pooled_dim + C_static]
  5. Regression head → [B, 1] prediction in z-score space

Output: [B, 1] prediction (z-score space, trend added later)
```

---

## LINEAR MODELS (linearLayer.py)

### NLinearYieldModel
**Paper:** "Are Transformers Effective for Time Series Forecasting?" (Zeng et al., AAAI 2023)

**Data Flow:**
```
x_ts [B, T, C] → truncate to effective_seq_len
  ↓
Find last valid value per channel (using observed_mask)
  last_val [B, 1, C] = gather(x_ts, last_valid_idx)
  ↓
Subtract last value (distribution shift normalization)
  x_shifted [B, T, C] = x_ts - last_val
  ↓
Transpose + Linear layer per channel
  x_t [B, C, T] → temporal_linear(T→1) → out [B, C, 1]
  ↓
Add last value back (undo shift)
  out [B, C, 1] = out + last_val^T
  ↓
Squeeze + Concat with static
  pooled [B, C] → cat([pooled, x_static]) → [B, C + C_static]
  ↓
Regression head → prediction [B, 1]
```

**Key Innovation:** Last-value subtraction for distribution shift robustness

**Status:** ✓ Correct

---

### DLinearYieldModel
**Paper:** "Are Transformers Effective for Time Series Forecasting?" (Zeng et al., AAAI 2023)

**Data Flow:**
```
x_ts [B, T, C] → truncate to effective_seq_len
  ↓
Transpose to [B, C, T]
  ↓
Extract trend via moving average (mask-aware)
  trend [B, C, T] = moving_avg(x_padded, kernel_size)
  remainder [B, C, T] = x_t - trend
  ↓
Separate linear layers for trend and remainder
  trend_out [B, C, 1] = trend_linear(trend)
  remainder_out [B, C, 1] = remainder_linear(remainder)
  ↓
Add + Squeeze
  pooled [B, C] = (trend_out + remainder_out).squeeze(-1)
  ↓
Concat with static → Regression head → prediction [B, 1]
```

**Key Innovation:** Trend-remainder decomposition

**Status:** ✓ Correct

---

### RLinearYieldModel
**Paper:** "An Analysis of Linear Time Series Forecasting Models" (Li et al. 2024)

**Data Flow:**
```
x_ts [B, T, C] → truncate to effective_seq_len
  ↓
Apply RevIN (per-instance normalization)
  x_revin [B, T, C] = (x_ts - instance_mean) / instance_std
  (Also applies affine params gamma, beta if enabled)
  ↓
Transpose + Linear layer per channel
  x_t [B, C, T] → temporal_linear(T→1) → out [B, C, 1]
  ↓
Squeeze + Concat with static → Regression head → prediction [B, 1]
```

**Key Innovation:** RevIN instance normalization

**Status:** ✓ Correct

---

### XLinearYieldModel
**Paper:** "A Lightweight and Accurate MLP-Based Model for Long-Term Time Series Forecasting with Exogenous Inputs" (Chen et al., AAAI 2026)

**Data Flow:**
```
x_ts [B, T, C] → truncate to effective_seq_len
  ↓
Build endogenous series:
  If lag_years > 0: lag_yield [B, 1] → broadcast → endo [B, T, 1]
  Else: mean(x_ts, dim=-1) → endo [B, T, 1]
  ↓
Apply RevIN if enabled (skip for constant lag_yield)
  ↓
Embed endogenous:
  h_endo [B, T, hidden] = endo_embed(endo)
  ↓
Initialize global token:
  G [B, 1, hidden] = global_token.expand(B)
  ↓
Embed exogenous channels:
  x_ts reshaped to [B*C, T, 1] → exo_embed → h_exo [B, C, T, hidden]
  ↓
Apply TGM (Temporal Gating Module) to h_endo
  h_tgm [B, T, hidden] = TGM(h_endo)
  ↓
Update global token (masked mean of h_tgm)
  G [B, 1, hidden] = G + mean(h_tgm)
  ↓
Apply VGM (Variate Gating Module): G × each exo channel
  h_vgm [B, T, C, hidden] = VGM(cat([G_expanded, h_exo]))
  ↓
Pool (masked mean):
  endo_pooled [B, hidden] = mean(h_tgm)
  exo_pooled [B, C*hidden] = mean(h_vgm)
  ↓
Concat: [endo_pooled, exo_pooled, x_static] → Regression head → prediction [B, 1]
```

**Key Innovation:** Cross-attention between endogenous patches and exogenous channels via gating

**Status:** ⚠️ Known Limitation - Endogenous series (lag_yield broadcast) is constant over time, which limits the effectiveness of patching. Semantically correct (lag_yield is the target variable) but architecturally constrained.

---

### OLinearYieldModel
**Paper:** "OLinear: A Linear Orthogonal Transformation for Time Series Forecasting" (Yue et al., 2025)

**Data Flow:**
```
x_ts [B, T, C] → truncate to effective_seq_len
  ↓
Initialize channel correlation matrix (first training batch only, frozen)
  channel_corr_mat [C, C] = softmax(correlation(x_ts))
  ↓
Apply RevIN normalization
  x_norm [B, T, C] = (x_ts - instance_mean) / instance_std
  ↓
Token embedding with dimension expansion:
  x_emb [B, C, T, D] = tokenEmb(x_norm, embeddings)
  ↓
Transpose: [B, C, T, D] → [B, C, D, T]
  ↓
Flatten: [B, C, D*T]
  ↓
Apply ortho_trans MLP:
  encoded [B*C, D*T] → ortho_trans → [B*C, embed_size]
  ↓
Reshape: [B, C, embed_size]
  ↓
Apply channel correlation matrix (softmax-normalized):
  encoded [B, C, embed_size] = corr_weight [B, C, C] @ encoded [B, C, embed_size]
  ↓
Pool over embed_size:
  pooled [B, C] = mean(encoded, dim=-1)
  ↓
Concat with static → Regression head → prediction [B, 1]
```

**Key Innovation:** Orthogonal transformation + channel correlation matrix

**Status:** ✓ Correct (with simplification: correlation matrix computed once and frozen)

---

## TRANSFORMER MODELS (tstLayer.py)

### AutoformerYieldModel
**Paper:** "Autoformer: Decomposition Transformers for Auto-correlation-aware Time Series Forecasting" (Wu et al., 2021)

**Data Flow:**
```
x_ts [B, T, C] → truncate to context_length
  ↓
Call HuggingFace Autoformer with:
  past_values = x_ts
  past_time_features = zeros(B, T, 0)
  past_observed_mask = observed_mask expanded to [B, T, C]
  future_values = zeros(B, 1, C)
  future_time_features = zeros(B, 1, 0)
  ↓
Extract hidden state:
  h [B, seq_len, d_model] or [B, n_channels, n_patches, d_model]
  ↓
Pool hidden state:
  If 3D: mean over seq_len → [B, d_model]
  If 4D: mean over patches, flatten channels → [B, n_channels*d_model]
  ↓
Concat with static → Regression head → prediction [B, 1]
```

**Key Innovation:** Auto-correlation mechanism

**Status:** ✓ Correct

---

### PatchTSTModel
**Paper:** "A Time Series is Worth 64 Words" (Nie et al., 2023)

**Data Flow:**
```
x_ts [B, T, C] → truncate to context_length
  ↓
Call HuggingFace PatchTST with:
  past_values = x_ts
  past_observed_mask = observed_mask expanded to [B, T, C]
  future_values = None
  ↓
Extract hidden state:
  h [B, n_channels, n_patches, d_model]
  ↓
Pool over patches, flatten channels:
  pooled [B, n_channels*d_model] = mean(h, dim=2).reshape(B, -1)
  ↓
Concat with static → Regression head → prediction [B, 1]
```

**Key Innovation:** Patching + channel independence

**Status:** ✓ Correct

---

### iTransformerYieldModel
**Paper:** "iTransformer: Inverted Transformers Are Effective for Time Series Forecasting" (Liu et al., 2023)

**Data Flow:**
```
x_ts [B, T, C] → truncate to context_length
  ↓
Apply RevIN if enabled:
  x_ts [B, T, C] = (x_ts - instance_mean) / instance_std
  ↓
Inverted embedding (channels as tokens):
  x_ts_permuted [B, C, T] = x_ts.permute(0, 2, 1)
  embedded [B, C, hidden] = inverted_embedding(x_ts_permuted)
  ↓
Transformer encoder (channels attend to each other):
  enc_out [B, C, hidden] = encoder(embedded)
  ↓
Project each channel to scalar:
  channel_scalars [B, C] = channel_projection(enc_out).squeeze(-1)
  ↓
Handle invalid channels (mask-based validity check):
  filled_channels [B, C] = where(channel_validity, channel_scalars, mean)
  ↓
Concat with static → Regression head → prediction [B, 1]
```

**Key Innovation:** Inverted embedding (channels as tokens, not timesteps)

**Status:** ✓ Correct

---

### TimeXerYieldModel
**Paper:** "TimeXer: Empowering Transformers for Time Series Forecasting with Exogenous Variables" (Wang et al., 2024)

**Data Flow:**
```
x_ts [B, T, C] → truncate to context_length
  ↓
Build endogenous series:
  If lag_years > 0: lag_yield [B, 1] → broadcast → endo [B, T, 1]
  Else: mean(x_ts, dim=-1) → endo [B, T, 1]
  ↓
Endogenous patching:
  endo unfolded → [B, patch_num, patch_len]
  endo_patches [B*C, patch_num, hidden] = endo_patch_embedding(endo)
  Add positional encoding
  Add global token → endo_with_glb [B*C, patch_num+1, hidden]
  ↓
Exogenous inverted embedding:
  x_ts_permuted [B, C, T] = x_ts.permute(0, 2, 1)
  exo_embed [B, C, hidden] = exo_inverted_embedding(x_ts_permuted)
  exo_expanded [B*C, C, hidden] for cross-attention
  ↓
Encoder with self- and cross-attention:
  enc_out [B*C, patch_num+1, hidden] = encoder(endo_with_glb, exo_expanded)
  ↓
Flatten and project per channel:
  enc_out [B, C, patch_num+1, hidden]
  flat [B, C, hidden*(patch_num+1)] = flatten(enc_out)
  channel_repr [B, C] = channel_projection(flat)
  ↓
Handle invalid channels (fill with mean):
  filled_channels [B, C] = where(channel_validity, channel_repr, mean)
  ↓
Concat with static → Regression head → prediction [B, 1]
```

**Key Innovation:** Cross-attention between endogenous patches and exogenous channels

**Status:** ⚠️ Known Limitation - Same issue as XLinear: endogenous series (lag_yield broadcast) is constant over time, limiting the effectiveness of patching.

---

### TimesNetModel
**Paper:** "TimesNet: Temporal 2D-Variation Modeling for General Time Series Analysis"

**Data Flow:**
```
x_ts [B, T, C] → truncate to context_length
  ↓
Input embedding:
  enc_out [B, T, d_model] = enc_embedding(x_ts)
  ↓
Apply TimesNet blocks (FFT-based period detection + 2D conv):
  for block in times_net_blocks:
    enc_out [B, T, d_model] = block(enc_out)
    enc_out [B, T, d_model] = layer_norm(enc_out)
  ↓
Pool over time:
  pooled [B, d_model] = mean(enc_out, dim=1)
  ↓
Concat with static → Regression head → prediction [B, 1]
```

**Key Innovation:** FFT-based period detection + 2D convolution

**Status:** ✓ Correct

---

## SHARED COMPONENTS

### BaseTimeSeriesModel (Both files)

**Shared Processing Pipeline:**

1. **_normalize_time_series**: Z-score normalization using training statistics
   - Handles each feature by name
   - Re-zeros padded positions after normalization
   - Mask-aware

2. **_normalize_and_impute_static**: Z-score normalization for static features
   - Normalizes first, then imputes NaN → 0.0 (correct order)
   - Mask-aware

3. **_shared_step**: Training step
   - Normalize inputs
   - Forward pass
   - Add trend (if use_residual_trend) with .detach()
   - Compute weighted MSE loss
   - Update metrics

4. **_eval_step_with_clipping**: Evaluation/validation step
   - Same as _shared_step but:
   - Denormalizes predictions to original scale
   - Clips predictions to ≥ 0
   - Logs clip rate as diagnostic
   - Computes metrics on clipped predictions

5. **test_step**: Test step
   - Handles recursive lag prediction if enabled
   - Accumulates per-year predictions for CSV results

6. **predict**: Inference-only prediction
   - Returns dict with predictions, targets, metadata

### Context Length Standardization

All models use:
```python
lags_sequence = [1] if config.lag_years > 0 else [0]
effective_seq_len = seq_len - max(lags_sequence)
```

**Result:**
- lag_years = 0: effective_seq_len = seq_len
- lag_years > 0: effective_seq_len = seq_len - 1

This ensures fair comparison across architectures.

---

## REGRESSION HEAD (All models)

Standardized 2-layer MLP:
```python
nn.Sequential(
    nn.Linear(input_dim, input_dim // 2),
    nn.LayerNorm(input_dim // 2),
    nn.ReLU() or nn.GELU(),
    nn.Dropout(0.1),
    nn.Linear(input_dim // 2, 1)
)
```

**input_dim varies by model:**
- Linear models (NLinear, DLinear, RLinear): n_ts_features + n_static_features
- Transformer HF models: d_model (64) + n_static_features
- iTransformer, TimeXer: n_ts_features + n_static_features
- TimesNet: d_model (64) + n_static_features

---

## KNOWN LIMITATIONS

### XLinear / TimeXer Endogenous Series

**Issue:** The endogenous series is constructed by broadcasting lag_yield (a scalar) over all timesteps, creating a constant series with no temporal variation.

**Why:** For crop yield prediction, we don't have historical yield observations at seasonal time resolution to use as true endogenous data.

**Impact:** The patching mechanism (core to these architectures) operates on constant data, limiting its effectiveness.

**Fallback:** When lag_years = 0, uses mean of exogenous channels (has temporal variation).

**Trade-off:**
- **Current approach (lag_yield):** Semantically correct (endogenous = target variable), but architecturally limited
- **Alternative (NDVI):** Has temporal patterns, but semantically incorrect (NDVI is an input feature, not the target)
- **Alternative (mean of exogenous):** Has temporal variation, but not "endogenous" in any meaningful sense

Current implementation uses lag_yield for semantic correctness.

---

## FIXED BUGS

1. ✓ **linearLayer.py:506** - Added `return loss` in test_step (recursive lag branch)
2. ✓ **tstLayer.py:635** - Added `.detach()` on trends in _eval_step_with_clipping
3. ✓ **tstLayer.py:857** - Added `.detach()` on trends in predict()
4. ✓ **tstLayer.py:733,734,771,772** - Standardized epsilon to 1e-8 with clamp()

---

## FINAL STATUS

| Model | Status | Notes |
|-------|--------|-------|
| NLinear | ✓ Correct | Core innovation preserved |
| DLinear | ✓ Correct | Core innovation preserved |
| RLinear | ✓ Correct | Core innovation preserved |
| XLinear | ⚠️ Limitation | Constant endogenous (known trade-off) |
| OLinear | ✓ Correct | Simplified correlation update |
| Autoformer | ✓ Correct | Core innovation preserved |
| PatchTST | ✓ Correct | Core innovation preserved |
| iTransformer | ✓ Correct | Core innovation preserved |
| TimeXer | ⚠️ Limitation | Constant endogenous (known trade-off) |
| TimesNet | ✓ Correct | Core innovation preserved |
| TSMixer | ✓ Correct | Core innovation preserved |
| Informer | ✓ Correct | Core innovation preserved |
| TST | ✓ Correct | Core innovation preserved |

All architectures are correctly implemented for the crop yield prediction task, with documented limitations for XLinear and TimeXer regarding endogenous series construction.
