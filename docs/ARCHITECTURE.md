# SpliNet architecture

SpliNet uses the FNet masked-language-model scaffold while replacing the Fourier mixing operation with a fixed multi-head cardinal B-spline inverse prefilter.

For the released model, the spline order is fixed at 2 and the transform is single-sided: it is applied along the sequence axis only. Hidden features are reshaped into 12 heads of dimension 64, but no operation mixes information across head boundaries inside the spline mixer.

For each batch/head/feature signal, the sequence transform uses the stable quadratic cardinal B-spline pole

`z = -3 + 2 sqrt(2)`

and the symmetric truncated inverse kernel

`h[k] = sqrt(2) z^|k|`, for `|k| <= 16`.

The implementation evaluates all `B * H * Dh` sequence signals in one reflected-padding `conv1d` call. The mixer itself therefore has zero learned parameters.

Each encoder layer is:

```text
input
  -> fixed SpliNet mixer
  -> residual + LayerNorm
  -> 3072-wide feed-forward block
  -> residual + LayerNorm
```

The embedding stack and masked-language-model prediction head are inherited from the FNet implementation. The released model uses 12 layers, hidden size 768, 12 heads, sequence length 512 and a 32k vocabulary.
