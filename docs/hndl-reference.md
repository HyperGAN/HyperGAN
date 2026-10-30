# HNDL (HyperGAN Network Description Language) Reference

HyperGAN Network Description Language (HNDL) is a domain-specific architecture description language for defining neural network components. HNDL programs compile into standard PyTorch `nn.Module` computational graphs with static shape inference, dimension tracking, and automatic parameter management.

---

## 1. Syntax & Core Concepts

An HNDL file (`.hndl`) or inline `source` block describes a series of operations applied sequentially or across named tensor ports.

### 1.1 Port Bindings and Sequential Flow

- Sequential flow passes the output of each operation directly as the primary input (`x`) of the following line:
  ```hndl
  linear(64)
  relu()
  linear(10)
  ```
- Named bindings allow multiple inputs and branching graphs:
  ```hndl
  skip = identity()
  h = conv(64, kernel=3, padding=1)
  h = relu(h)
  out = add(h, skip)
  ```

### 1.2 Shape Contracts & Dimension Symbols

Every HNDL network specifies explicit input and output shape contracts:
- `['B', 4]` denotes a batch of 4-dimensional vectors.
- `['B', 3, 32, 32]` denotes a batch of 3-channel 32x32 spatial feature maps.
- `'B'` is always the leading batch dimension.
- Spatial and sequence dimensions are statically checked at compile time via `hndl.resolve()`.

---

## 2. Operator Catalog (85 Operators)

HyperGAN includes 85 registered operators categorized into 13 functional domains.

### 2.1 Activation Operators (15)

Non-linear activation functions applied element-wise.

1. **`relu()`**: Rectified Linear Unit ($\max(0, x)$).
2. **`leaky_relu(negative_slope=0.2)`**: Leaky ReLU with slope for $x < 0$.
3. **`gelu(approximate='none')`**: Gaussian Error Linear Unit.
4. **`quick_gelu()`**: Fast Sigmoid approximation: $x \cdot \sigma(1.702 \cdot x)$.
5. **`silu()`**: Sigmoid Linear Unit / Swish ($x \cdot \sigma(x)$).
6. **`tanh()`**: Hyperbolic tangent $(-1 \dots 1)$.
7. **`sigmoid()`**: Logistic sigmoid $(0 \dots 1)$.
8. **`dy_tanh()`**: Dynamic hyperbolic tangent with learned per-channel saturation.
9. **`elu(alpha=1.0)`**: Exponential Linear Unit.
10. **`hardswish()`**: Piecewise linear approximation of Swish.
11. **`identity()`**: Passthrough identity function.
12. **`mish()`**: Self-regularized non-monotonic activation: $x \cdot \tanh(\text{softplus}(x))$.
13. **`snake(frequency=1.0)`**: Periodic activation for continuous signals: $x + \frac{1}{a} \sin^2(a x)$.
14. **`softmax(dim=-1)`**: Normalized exponential distribution over specified axis.
15. **`softplus(beta=1.0, threshold=20.0)`**: Smooth approximation to ReLU: $\frac{1}{\beta} \log(1 + e^{\beta x})$.

### 2.2 Arithmetic Operators (10)

Tensor math and learned affine transformations.

16. **`add(a, b)`**: Element-wise addition of two tensors with matching shapes.
17. **`sub(a, b)`**: Element-wise subtraction ($a - b$).
18. **`mul(a, b)`**: Element-wise multiplication ($a \cdot b$).
19. **`broadcast_add(a, b)`**: Addition broadcasting singleton dimensions.
20. **`broadcast_mul(a, b)`**: Multiplication broadcasting singleton dimensions.
21. **`clamp(min=None, max=None)`**: Clamps tensor values within $[min, max]$.
22. **`constant(value, shape)`**: Emits a constant scalar or tensor literal.
23. **`learned_scale(init=1.0)`**: Learnable scalar or channel-wise multiplier parameter.
24. **`matmul(a, b)`**: Matrix multiplication over final two dimensions.
25. **`scale(factor)`**: Fixed scalar multiplication.

### 2.3 Convolution Operators (4)

Spatial 1D and 2D convolution and transposed convolution layers.

26. **`conv(channels, kernel=3, stride=1, padding=0, groups=1, bias=True)`**: 2D spatial convolution.
27. **`conv1d(channels, kernel=3, stride=1, padding=0, groups=1, bias=True)`**: 1D temporal/sequence convolution.
28. **`deconv(channels, kernel=4, stride=2, padding=1, bias=True)`**: 2D transposed convolution (spatial upsampling).
29. **`deconv1d(channels, kernel=4, stride=2, padding=1, bias=True)`**: 1D transposed convolution.

### 2.4 Core & Linear Operators (3)

Fundamental tensor transformations and conditional branching.

30. **`linear(out_features, bias=True)`**: Fully-connected affine transformation.
31. **`cast(dtype)`**: Type conversion ('float32', 'float16', 'bfloat16', 'int64').
32. **`where(condition, x, y)`**: Selects elements from $x$ or $y$ depending on boolean condition.

### 2.5 Memory & Recurrence (1)

Associative and attractor memory.

33. **`hopfield(dim, heads=4, pattern_dim=None)`**: Modern Hopfield continuous associative memory retrieval.

### 2.6 Normalization Operators (9)

Batch, channel, group, layer, and adaptive normalization layers.

34. **`adaln_zero(cond, gate=True)`**: Adaptive LayerNorm-Zero conditioning modulation for DiT architectures.
35. **`adaptive_norm(cond, norm='layer')`**: Adaptive style/latent modulation (AdaIN).
36. **`batch_norm(momentum=0.1, eps=1e-5)`**: Spatial batch normalization across mini-batch.
37. **`grn(eps=1e-6)`**: Global Response Normalization across spatial feature channels.
38. **`group_norm(num_groups=32, eps=1e-5)`**: Group normalization dividing channels into sub-groups.
39. **`instance_norm(eps=1e-5, affine=False)`**: Contrast normalization over spatial dimensions per sample.
40. **`l2_norm(dim=-1, eps=1e-12)`**: Euclidean unit-norm projection along specified dimension.
41. **`layer_norm(eps=1e-5)`**: Normalization over hidden feature dimensions.
42. **`rms_norm(eps=1e-8)`**: Root Mean Square layer normalization without mean centering.

### 2.7 Pretrained Providers (1)

External foundation backbone model integration.

43. **`pretrained(source, provider="", output="features", layer="", layers=(), readout="")`**:
    Loads frozen vision/text foundation models (e.g., `torchvision_resnet18`, `dinov3_vits16`) with static symbolic shape resolution.

### 2.8 Regularization & Augmentation (3)

Stochastic regularization and differentiable data augmentation.

44. **`diff_augment(policy="color,translation,cutout")`**: Differentiable Data Augmentation for GAN stability.
45. **`drop_path(drop_prob=0.1)`**: Stochastic depth / drop path per sample in residual networks.
46. **`dropout(p=0.5)`**: Standard random feature dropout.

### 2.9 Sequence & Attention Operators (18)

Transformer attention blocks, positional embeddings, and state-space layers.

47. **`attention(heads=8, dim_head=64)`**: Multi-head self-attention.
48. **`cls_token(dim)`**: Prepends learnable class classification token to sequence.
49. **`cross_attention(context, heads=8)`**: Multi-head cross-attention over external context.
50. **`embedding(num_embeddings, embedding_dim)`**: Discrete token embedding table lookup.
51. **`feed_forward(hidden_dim, activation='gelu')`**: Standard transformer two-layer MLP block.
52. **`fourier_features(num_bands=16, max_freq=10.0)`**: Random Fourier positional encoding.
53. **`gated_delta_attention(dim, heads=4)`**: Linear attention with recurrent delta rule.
54. **`gather_token(index=0)`**: Extracts single token from sequence by index.
55. **`joint_attention(context, heads=8)`**: Symmetrical joint attention between two sequences.
56. **`latent_attention(latents=16, heads=8)`**: Perceiver-style cross-attention querying learned latent queries.
57. **`mamba2(d_state=64, d_conv=4, expand=2)`**: Selective State Space Model (SSM) layer.
58. **`moe(num_experts=8, top_k=2)`**: Mixture of Experts feed-forward block.
59. **`pool_tokens(mode='mean')`**: Reduces sequence length via mean, max, or attention pooling.
60. **`pos_embed(dim, max_len=1024)`**: Learnable 1D additive positional embeddings.
61. **`rope(dim)`**: Rotary Position Embedding applied to query/key tensors.
62. **`rope_2d(dim, height=None, width=None)`**: 2D Rotary Position Embedding applying sinusoidal frequency rotations across spatial height and width axes in vision transformers.
63. **`sinusoidal_embed(dim, max_period=10000)`**: Fixed sinusoidal positional encodings.
64. **`swiglu(dim)`**: Swish-Gated Linear Unit feed-forward block ($x \cdot \text{sigmoid}(\beta x) \otimes W_g x$).
65. **`transformer_block(heads=8, mlp_ratio=4.0)`**: Complete transformer layer (norm, attention, residual, FFN).

### 2.10 Shape & Layout Operators (10)

Tensor reshaping, permuting, concatenation, and reduction.

65. **`chunk(chunks, dim=1)`**: Splits tensor into equal-sized chunks along specified dimension.
66. **`concat(tensors, dim=1)`**: Concatenates multiple tensors along an axis.
67. **`flatten(start_dim=1, end_dim=-1)`**: Flattens contiguous range of dimensions.
68. **`mean(dim=None, keepdim=False)`**: Reduces tensor along dimensions via arithmetic mean.
69. **`pad(pad, mode='constant', value=0.0)`**: Pads tensor boundaries (e.g. `pad=(1, 1, 1, 1)`).
70. **`permute(dims)`**: Rearranges tensor axes according to specified permutation.
71. **`reshape(shape)`**: Reshapes tensor to new dimension tuple (preserves total elements).
72. **`split(split_size, dim=1)`**: Splits tensor into specific section lengths.
73. **`sum(dim=None, keepdim=False)`**: Computes sum across specified dimensions.
74. **`transpose(dim0, dim1)`**: Swaps two dimensions.

### 2.11 Spatial & Pooling Operators (9)

Spatial sampling, pooling, and coordinates.

75. **`adaptive_avg_pool(output_size)`**: Pools spatial dimensions to fixed $(H, W)$ target.
76. **`avg_pool(kernel_size=2, stride=2, padding=0)`**: Spatial 2D average pooling.
77. **`coordinate_grid(align_corners=False)`**: Appends normalized $[-1, 1]$ $(x, y)$ coordinate channels.
78. **`global_avg_pool()`**: Reduces spatial $(B, C, H, W)$ to $(B, C)$ feature vector.
79. **`grid_sample(grid, mode='bilinear', padding_mode='zeros')`**: Samples 2D input at grid locations.
80. **`max_pool(kernel_size=2, stride=2, padding=0)`**: Spatial 2D maximum pooling.
81. **`pixel_shuffle(upscale_factor=2)`**: Sub-pixel convolution rearranging $(B, C \cdot r^2, H, W) \to (B, C, H \cdot r, W \cdot r)$.
82. **`spatial_attention(kernel_size=7)`**: Inter-spatial attention feature modulation.
83. **`upsample(scale_factor=2, mode='nearest')`**: Spatial upsampling (nearest, bilinear, bicubic).

### 2.12 Vision Architecture Blocks (2)

High-level compound vision building blocks.

84. **`patch_embed(patch_size=4, in_channels=3, embed_dim=128)`**: Vision Transformer patch projection.
85. **`resblock(channels, stride=1)`**: Pre-activation residual convolutional block with shortcut connection.

---

## 3. End-to-End Network Examples

### 3.1 DCGAN Generator (`networks.generator`)
```hndl
# Input: ['B', 64] -> Output: ['B', 3, 32, 32]
linear(256 * 4 * 4)
reshape(['B', 256, 4, 4])
group_norm(32)
relu()
deconv(128, kernel=4, stride=2, padding=1)
group_norm(32)
relu()
deconv(64, kernel=4, stride=2, padding=1)
group_norm(16)
relu()
deconv(3, kernel=4, stride=2, padding=1)
tanh()
```

### 3.2 Vision Transformer Discriminator (`networks.critic`)
```hndl
# Input: ['B', 3, 32, 32] -> Output: ['B', 1]
diff_augment("color,translation")
patch_embed(patch_size=4, in_channels=3, embed_dim=128)
pos_embed(128)
transformer_block(heads=4, mlp_ratio=2.0)
transformer_block(heads=4, mlp_ratio=2.0)
pool_tokens(mode="mean")
linear(1)
```

### 3.3 Conditional U-Net Colorizer with RoPE + SwiGLU Attention
```hndl
# Input: ['B', 1, 64, 64] -> Output: ['B', 3, 64, 64]
# Encoder
c1 = conv(64, kernel=3, padding=1)
h1 = relu(c1)
c2 = conv(128, kernel=4, stride=2, padding=1, x=h1)
h2 = group_norm(32, x=c2)

# Bottleneck with SwiGLU & 2D RoPE
b1 = swiglu(128, x=h2)
b2 = rope_2d(128, height=32, width=32, x=b1)

# Decoder with skip-connection
u1 = deconv(64, kernel=4, stride=2, padding=1, x=b2)
u1_cat = concat([u1, h1], dim=1)
out = conv(3, kernel=3, padding=1, x=u1_cat)
tanh(out)
```

---

## 4. Formal Grammar & Shape Inference

### 4.1 EBNF Grammar Specification

HNDL programs follow a clean, pythonic grammar:

```ebnf
program         = { statement ( "\n" | ";" ) } ;
statement       = [ assignment | loop | expr ] ;
assignment      = IDENTIFIER "=" expr ;
loop            = "for" IDENTIFIER "in" "range(" INTEGER ")" ":" NEWLINE INDENT { statement } DEDENT ;
expr            = call | literal | IDENTIFIER ;
call            = IDENTIFIER "(" [ arg_list ] ")" ;
arg_list        = positional_args [ "," keyword_args ] | keyword_args ;
positional_args = expr { "," expr } ;
keyword_args    = IDENTIFIER "=" expr { "," IDENTIFIER "=" expr } ;
literal         = NUMBER | STRING | shape_literal | list_literal ;
shape_literal   = "[" ( "'B'" | IDENTIFIER | INTEGER ) { "," ( IDENTIFIER | INTEGER ) } "]" ;
list_literal    = "[" [ expr { "," expr } ] "]" ;
```

Bounded loops (`for _ in range(N):`) are statically unrolled during AST compilation, suffixing operator node identifiers (`block_0`, `block_1`) to generate deterministic PyTorch module graphs.

### 4.2 Symbolic Dimension Tracking & Shape Propagation

The HNDL compiler propagates symbolic dimensions through graph AST nodes:
- `'B'`: Dynamic batch size dimension.
- `'H'`, `'W'`: Spatial height and width dimensions.
- `'C'`: Channel depth dimension.
- `'S'`: Sequence or token length dimension.
- `'D'`: Embedding / hidden feature dimension.

For spatial convolution and pooling operations, spatial dimensions propagate according to:
$$O = \left\lfloor \frac{I - K + 2P}{S} \right\rfloor + 1$$
For transposed convolutions (`deconv`):
$$O = (I - 1) \cdot S - 2P + K + P_{\text{out}}$$
For sub-pixel convolutions (`pixel_shuffle(r)`):
$$(B, C \cdot r^2, H, W) \longrightarrow (B, C, H \cdot r, W \cdot r)$$

Foundation backbone operators (`pretrained(...)`) query static provider shape registries, enabling full offline graph validation and dimension checking without downloading multi-gigabyte weight checkpoints.

---

## 5. Static Error Diagnostics & Code Reference

The HNDL parser and AST resolver perform strict validation before graph instantiation. Errors provide line and column offsets, context snippets, and typo suggestions:

| Error Code | Category | Cause & Resolution |
| :--- | :--- | :--- |
| **`E_SYNTAX_ERROR`** | Syntax | Malformed syntax, unmatched parentheses, or illegal tokens. |
| **`E_UNKNOWN_OPERATOR`** | Lookup | Unregistered operator name. HyperGAN matches against the 85-operator catalog and suggests close spellings via `difflib`. |
| **`E_SHAPE_MISMATCH`** | Tensor Shape | Operation received a tensor whose rank or dimension violates input constraints (e.g. 2D conv on 1D tensor). |
| **`E_UNBOUND_INPUT`** | Binding | Graph references a named input port that was not bound in the model file's `inputs` table. |
| **`E_DIMENSION_MISMATCH`** | Channels | Channel mismatch between adjacent layers (e.g. `linear(128)` followed by `conv(64)` without `reshape`). |
| **`E_CYCLE_DETECTED`** | Graph AST | Directed cycle detected in named variable assignments. |

