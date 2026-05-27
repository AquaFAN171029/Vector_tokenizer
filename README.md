# SVG Vector Pipeline

The current pipeline is:

```text
SVG drawing
  -> parser JSON
  -> tokenized primitive records
  -> channel embeddings
  -> fused 256-dimensional primitive embeddings
```

## Directory Layout

```text
vector_pipeline/
  examples/
    single_file/
      input.svg
      parsed.json
      tokenized.json
      embeddings.json

  svg_parser/
    svg_parser_v2.py

  tokenizer/
    channels/
      type.py
      geom.py
      style.py
      position.py
      text.py
    core/
      dataset_reader.py
      sequence_builder.py
      primitive_tokenizer.py
    embeddings/
      hash_embedding.py
      flow_embedding.py
      qwen_text.py
    cli/
      run_tokenizer.py
      run_embedding.py
      run_flow_embedding.py
    run_tokenizer.py        # compatibility wrapper
    run_embedding.py        # compatibility wrapper
    run_flow_embedding.py   # compatibility wrapper

  pipeline/
    run_svg_dataset_pipeline.py
    README.md

  modeling/
    prediction_heads.py
    # reusable prediction heads for encoder outputs

  experiments/
    splits/
      build_file_split.py
      # file-level train/val/test split generation

    pairwise_relation/
      relation_labels.py
      build_pairwise_labels.py
      train_pairwise_smoke.py
      train_pairwise_head.py
      evaluate_pairwise_head.py
      # pseudo-label generation, training, and evaluation for relation prediction

  docs/
    generated/
      pipeline_architecture_tokenizer_embedder.docx
      build_pipeline_doc.py
      doc_assets/
    reference/
      tokenizer.docx
      tokenizer.pdf
      tokenizer_requirement.pdf

  outputs/
    processed_dataset_sample/
    processed_dataset_smoke/
```

## Quick Start

Run the pipeline:

```bash
python3 vector_pipeline/svg_parser/svg_parser_v2.py \
  vector_pipeline/examples/single_file/input.svg \
  -o vector_pipeline/examples/single_file/parsed.json

python3 -m vector_pipeline.tokenizer.run_tokenizer \
  vector_pipeline/examples/single_file/parsed.json \
  -o vector_pipeline/examples/single_file/tokenized.json

python3 -m vector_pipeline.tokenizer.run_embedding \
  vector_pipeline/examples/single_file/tokenized.json \
  -o vector_pipeline/examples/single_file/embeddings.json
```

Run embedding with Qwen text embeddings for the text channel:

```bash
export DASHSCOPE_API_KEY="your_api_key_here"

python3 -m vector_pipeline.tokenizer.run_embedding \
  vector_pipeline/examples/single_file/tokenized.json \
  -o vector_pipeline/examples/single_file/embeddings.json \
  --text-backend qwen \
  --text-dim 64
```

Run the single-file hybrid flow embedding path:

```bash
export DASHSCOPE_API_KEY="your_api_key_here"

python3 -m vector_pipeline.tokenizer.run_flow_embedding \
  vector_pipeline/examples/single_file/parsed.json \
  -o vector_pipeline/examples/single_file/flow.embeddings.json \
  --text-backend qwen
```

Run a small dataset test:

```bash
python3 -m vector_pipeline.pipeline.run_svg_dataset_pipeline --input-dir "SVG Dataset" --output-dir outputs/processed_dataset_sample --limit 3 --overwrite
```

Run the full SVG dataset:

```bash
python3 -m vector_pipeline.pipeline.run_svg_dataset_pipeline --input-dir "SVG Dataset" --output-dir outputs/processed_dataset --overwrite
```

Run the full dataset with Qwen text embeddings:

```bash
export DASHSCOPE_API_KEY="your_api_key_here"

python3 -m vector_pipeline.pipeline.run_svg_dataset_pipeline \
  --input-dir "SVG Dataset" \
  --output-dir outputs/processed_dataset_qwen \
  --text-backend qwen \
  --qwen-model text-embedding-v4 \
  --qwen-base-url https://dashscope-intl.aliyuncs.com/compatible-mode/v1 \
  --qwen-api-key-env DASHSCOPE_API_KEY
```

Batch outputs are written under `vector_pipeline/outputs/` when a relative `--output-dir` is used.

```text
vector_pipeline/outputs/processed_dataset/
  parsed/
  tokenized/
  embeddings/
  manifest.json
  summary.json
  embedding_dataset.jsonl
```

## Current End-to-End Workflow

This is the current recommended workflow for reproducing the full Qwen embedding and pairwise relation head experiment from scratch.

All commands below assume they are run from the parent workspace directory:

```bash
cd "/Users/aquafan/Library/Mobile Documents/com~apple~CloudDocs/POLYU STUDY/McGill"
```

### 1. Generate Full Qwen Embeddings

The current main embedding run uses official Qwen/DashScope `text-embedding-v4` for the text channel and keeps the final fused primitive embedding at 256 dimensions.

```bash
export DASHSCOPE_API_KEY="your_api_key_here"

python3 -m vector_pipeline.pipeline.run_svg_dataset_pipeline \
  --input-dir "SVG Dataset" \
  --output-dir outputs/processed_dataset_qwen \
  --text-backend qwen \
  --qwen-model text-embedding-v4 \
  --qwen-base-url https://dashscope-intl.aliyuncs.com/compatible-mode/v1 \
  --qwen-api-key-env DASHSCOPE_API_KEY
```

If the run is interrupted, rerun the same command without `--overwrite`; existing files will be skipped.

Check the result:

```bash
python3 -m json.tool vector_pipeline/outputs/processed_dataset_qwen/summary.json

find vector_pipeline/outputs/processed_dataset_qwen/embeddings \
  -name "*.embeddings.json" | wc -l
```

Expected full dataset size:

```text
3760 embedding JSON files
```

### 2. Build a File-Level Train/Val/Test Split

The split unit is the SVG file, not primitive pairs. This prevents leakage where pairs from the same drawing appear in both train and test.

```bash
python3 -m vector_pipeline.experiments.splits.build_file_split \
  --manifest vector_pipeline/outputs/processed_dataset_qwen/manifest.json \
  --output-json outputs/splits/file_split_qwen_seed13.json \
  --train-ratio 0.8 \
  --val-ratio 0.1 \
  --test-ratio 0.1 \
  --seed 13
```

Output:

```text
vector_pipeline/outputs/splits/file_split_qwen_seed13.json
```

### 3. Generate Split-Specific Pairwise Labels

Labels are generated from `parsed/` geometry and `tokenized/` order. The embeddings are not used to generate labels; embeddings are used later as model inputs.

Train labels:

```bash
python3 -m vector_pipeline.experiments.pairwise_relation.build_pairwise_labels \
  --parsed-dir vector_pipeline/outputs/processed_dataset_qwen/parsed \
  --tokenized-dir vector_pipeline/outputs/processed_dataset_qwen/tokenized \
  --split-json vector_pipeline/outputs/splits/file_split_qwen_seed13.json \
  --split train \
  --output-jsonl vector_pipeline/outputs/pairwise_relation_qwen_train/pairwise_labels.jsonl \
  --summary-json vector_pipeline/outputs/pairwise_relation_qwen_train/summary.json \
  --max-pairs-per-file 1000
```

Validation labels:

```bash
python3 -m vector_pipeline.experiments.pairwise_relation.build_pairwise_labels \
  --parsed-dir vector_pipeline/outputs/processed_dataset_qwen/parsed \
  --tokenized-dir vector_pipeline/outputs/processed_dataset_qwen/tokenized \
  --split-json vector_pipeline/outputs/splits/file_split_qwen_seed13.json \
  --split val \
  --output-jsonl vector_pipeline/outputs/pairwise_relation_qwen_val/pairwise_labels.jsonl \
  --summary-json vector_pipeline/outputs/pairwise_relation_qwen_val/summary.json \
  --max-pairs-per-file 1000
```

Test labels:

```bash
python3 -m vector_pipeline.experiments.pairwise_relation.build_pairwise_labels \
  --parsed-dir vector_pipeline/outputs/processed_dataset_qwen/parsed \
  --tokenized-dir vector_pipeline/outputs/processed_dataset_qwen/tokenized \
  --split-json vector_pipeline/outputs/splits/file_split_qwen_seed13.json \
  --split test \
  --output-jsonl vector_pipeline/outputs/pairwise_relation_qwen_test/pairwise_labels.jsonl \
  --summary-json vector_pipeline/outputs/pairwise_relation_qwen_test/summary.json \
  --max-pairs-per-file 1000
```

The current full run produced:

```text
train rows: 2,902,228
val rows:   362,158
test rows:  365,137
```

### 4. Smoke Test the Prediction Head

Run this before full training to verify that labels and embeddings align.

```bash
python3 -m vector_pipeline.experiments.pairwise_relation.train_pairwise_smoke \
  --embeddings-dir vector_pipeline/outputs/processed_dataset_qwen/embeddings \
  --labels-jsonl vector_pipeline/outputs/pairwise_relation_qwen_train/pairwise_labels.jsonl \
  --max-samples 5000 \
  --batch-size 128 \
  --epochs 1 \
  --hidden-dim 128
```

Successful smoke output should end with:

```text
Smoke training finished.
```

### 5. Train the Pairwise Relation Head

Full training command:

```bash
python3 -m vector_pipeline.experiments.pairwise_relation.train_pairwise_head \
  --embeddings-dir vector_pipeline/outputs/processed_dataset_qwen/embeddings \
  --train-labels-jsonl vector_pipeline/outputs/pairwise_relation_qwen_train/pairwise_labels.jsonl \
  --val-labels-jsonl vector_pipeline/outputs/pairwise_relation_qwen_val/pairwise_labels.jsonl \
  --output-dir outputs/checkpoints/pairwise_relation_head_qwen_full \
  --batch-size 256 \
  --epochs 25 \
  --hidden-dim 256 \
  --lr 1e-3
```

For a faster development run, add sample limits:

```bash
python3 -m vector_pipeline.experiments.pairwise_relation.train_pairwise_head \
  --embeddings-dir vector_pipeline/outputs/processed_dataset_qwen/embeddings \
  --train-labels-jsonl vector_pipeline/outputs/pairwise_relation_qwen_train/pairwise_labels.jsonl \
  --val-labels-jsonl vector_pipeline/outputs/pairwise_relation_qwen_val/pairwise_labels.jsonl \
  --output-dir outputs/checkpoints/pairwise_relation_head_qwen_50k \
  --max-train-samples 50000 \
  --max-val-samples 10000 \
  --batch-size 128 \
  --epochs 5 \
  --hidden-dim 256 \
  --lr 1e-3
```

Resume training from the best checkpoint:

```bash
python3 -m vector_pipeline.experiments.pairwise_relation.train_pairwise_head \
  --embeddings-dir vector_pipeline/outputs/processed_dataset_qwen/embeddings \
  --train-labels-jsonl vector_pipeline/outputs/pairwise_relation_qwen_train/pairwise_labels.jsonl \
  --val-labels-jsonl vector_pipeline/outputs/pairwise_relation_qwen_val/pairwise_labels.jsonl \
  --output-dir outputs/checkpoints/pairwise_relation_head_qwen_full \
  --resume-checkpoint vector_pipeline/outputs/checkpoints/pairwise_relation_head_qwen_full/best.pt \
  --batch-size 256 \
  --epochs 25 \
  --hidden-dim 256 \
  --lr 1e-3
```

When resuming, `--epochs` is the target total epoch count. For example, if `best.pt` is from epoch 15 and `--epochs 25`, training continues from epoch 16 to 25.

Training writes:

```text
vector_pipeline/outputs/checkpoints/pairwise_relation_head_qwen_full/
  best.pt
  last.pt
  config.json
  history.json
```

### 6. Evaluate on Test

Only run test evaluation after model selection is finished on validation.

```bash
python3 -m vector_pipeline.experiments.pairwise_relation.evaluate_pairwise_head \
  --checkpoint vector_pipeline/outputs/checkpoints/pairwise_relation_head_qwen_full/best.pt \
  --embeddings-dir vector_pipeline/outputs/processed_dataset_qwen/embeddings \
  --labels-jsonl vector_pipeline/outputs/pairwise_relation_qwen_test/pairwise_labels.jsonl \
  --output-json outputs/evaluations/pairwise_relation_head_qwen_test_full.json \
  --batch-size 256
```

The evaluation JSON contains overall metrics and per-type-pair metrics.

## Stage 1: Parser

The parser converts SVG elements into a primitive schema.

Supported primitive types:

```text
LINE
ARC
POLYLINE
TEXT
ANNOTATION
```

Parser output root:

```json
{
  "source_file": "path/to/input.svg",
  "viewBox": [0.0, 0.0, 100.0, 100.0],
  "num_primitives": 499,
  "primitives": []
}
```

Each primitive follows this schema:

```json
{
  "primitive_id": "p_000001",
  "type": "LINE",
  "subtype": null,
  "geometry": {
    "coords": [],
    "geom_mask": []
  },
  "style": {
    "layer": "layer0",
    "stroke": "rgb(0,0,178)",
    "stroke_width": 0.1,
    "fill": "none",
    "opacity": 1.0
  },
  "position": {
    "cx": 89.79,
    "cy": 74.10,
    "w": 20.40,
    "h": 0.0
  },
  "text": {
    "normalized_text": null,
    "english_text": null,
    "font_family": null,
    "font_size": null
  },
  "meta": {
    "parent_group_id": "layer0",
    "instance_id": null,
    "semantic_id": null
  }
}
```

## Stage 2: Tokenizer

The tokenizer converts each primitive into a token record.

Tokenizer output per primitive:

```json
{
  "primitive_id": "p_000005",
  "type_token": "LINE",
  "geom_tokens": ["G_185", "G_14", "G_185", "G_57", "G_31", "G_255", "G_128", "PAD"],
  "geom_mask": [1, 1, 1, 1, 1, 1, 1, 0],
  "style_tokens": ["LAYER_layer0", "STROKE_rgb(0,178,0)", "WIDTH_0", "FILL_none", "OP_10"],
  "pos_tokens": ["CX_185", "CY_36", "W_0", "H_43"],
  "text_tokens": [],
  "group_token": "GROUP_layer0"
}
```

### Sequence Ordering

Before tokenization, primitives are sorted into a sequence:

```text
parent_group_id -> cy -> cx -> primitive_id
```

Sorting Rules:

1. Group by `meta.parent_group_id`.
2. Within each group, sort top-to-bottom by `position.cy`.
3. Break ties left-to-right by `position.cx`.
4. Use `primitive_id` as a final stable tie-breaker.

### Type Token Mapping

Input:

```text
primitive.type
```

Rule:

```text
LINE       -> LINE
ARC        -> ARC
POLYLINE   -> POLYLINE
TEXT       -> TEXT
ANNOTATION -> ANNOTATION
other      -> UNK_TYPE
```

Output:

```text
type_token
```

### Geometry Token Mapping

Input:

```text
geometry.coords
geometry.geom_mask
viewBox = [vx, vy, vw, vh]
```

| Primitive type | `coords` slot meaning |
|---|---|
| `LINE` | `[x1, y1, x2, y2, length, sin(angle), cos(angle), 0]` |
| `POLYLINE` | `[start_x, start_y, end_x, end_y, total_length, num_points, closed_flag, 0]` |
| `ARC` | `[cx, cy, rx, ry, sin(start_angle), cos(start_angle), sin(end_angle), cos(end_angle)]` |
| `TEXT` / `ANNOTATION` | `[x, y, width_est, height_est, sin(rotation), cos(rotation), font_size, 0]` |

Normalization rules:

| Value type | Rule |
|---|---|
| x coordinate | `(x - vx) / vw` |
| y coordinate | `(y - vy) / vh` |
| width or x-radius | `value / vw` |
| height or y-radius | `value / vh` |
| line/polyline length | `length / sqrt(vw^2 + vh^2)` |
| sine/cosine value | `(value + 1) / 2` |
| polyline point count | `min(num_points, 255) / 255` |
| closed flag | `1.0 if value >= 0.5 else 0.0` |
| font size | `font_size / vh` |

Quantization rule:

```text
bin = round(clamp(normalized_value, 0, 1) * (geom_bins - 1))
```

Default:

```text
geom_bins = 256
```

Token rule:

```text
if geom_mask[i] == 1:
    geom_tokens[i] = "G_<bin>"
else:
    geom_tokens[i] = "PAD"
```

Output:

```text
geom_tokens
geom_mask
```

### Style Token Mapping

Input:

```text
style.layer
style.stroke
style.stroke_width
style.fill
style.opacity
```

Output format:

```text
[
  "LAYER_<cleaned_layer>",
  "STROKE_<cleaned_stroke>",
  "WIDTH_<width_bin>",
  "FILL_<cleaned_fill>",
  "OP_<opacity_bin>"
]
```

Rules:

| Field | Rule |
|---|---|
| `layer` | Clean string; missing value becomes `NO_LAYER` |
| `stroke` | Clean string; missing value becomes `NO_STROKE` |
| `fill` | Clean string; missing value becomes `NO_FILL` |
| `stroke_width` | Quantize from `[0, 10]` into `width_bins` |
| `opacity` | Quantize from `[0, 1]` into `opacity_bins` |

Defaults:

```text
width_bins = 32
opacity_bins = 11
```

### Position Token Mapping

Input:

```text
position.cx
position.cy
position.w
position.h
viewBox = [vx, vy, vw, vh]
```

Normalization:

```text
CX = (cx - vx) / vw
CY = (cy - vy) / vh
W  = w / vw
H  = h / vh
```

Quantization:

```text
bin = round(clamp(value, 0, 1) * (pos_bins - 1))
```

Default:

```text
pos_bins = 256
```

Output:

```text
["CX_<bin>", "CY_<bin>", "W_<bin>", "H_<bin>"]
```

### Text Token Mapping

Input:

```text
text.normalized_text
```

Rules:

```text
missing or empty text -> []
text frequency >= min_text_freq -> ["TEXT_<cleaned_text>"]
text frequency < min_text_freq  -> ["UNK_TEXT"]
```

Default:

```text
min_text_freq = 1
```

With the default, every observed text string is kept as a text token.

The tokenized primitive also stores:

```text
text_value
```

`text_value` is the string used by the embedder when `--text-backend qwen` is enabled. It is selected as:

```text
english_text if available, otherwise normalized_text
```

### Group Token Mapping

Input:

```text
meta.parent_group_id
```

Rule:

```text
missing group -> GROUP_NO_GROUP
existing group -> GROUP_<cleaned_parent_group_id>
```

Output:

```text
group_token
```

## Stage 3: Embedder

The embedder maps each tokenized primitive into one fused 256 dimensional vector.

Current embedding shape:

```text
type_emb  = 32
geom_emb  = 128
style_emb = 32
pos_emb   = 64
text_emb  = 64
group_emb = 16

concat_dim = 32 + 128 + 32 + 64 + 64 + 16 = 336
output_dim = 256
```

Each channel dimension is selected from:

```text
{16, 32, 64, 128}
```

### Token-to-Vector Rule

For the default prototype backend, each symbolic token is mapped to a deterministic vector using:

```text
SHA-256(seed, token, dim) -> pseudo-random vector -> L2 normalization
```

This gives stable, reproducible vectors without training.

Important:

```text
PAD tokens are ignored during channel pooling.
```

### Channel Pooling Rule

Each channel can contain one or more tokens. The embedder averages token vectors inside each channel:

```text
channel_emb = average(non_PAD_token_vectors)
```

If a channel has no valid token, it becomes a zero vector.

### Qwen Text Embedding Backend

The text channel can optionally use Qwen `text-embedding-v4` instead of the default hash-based text token vector.

This backend is enabled with:

```bash
--text-backend qwen
```

The text embedding dimension is fixed to:

```text
text_dim = 64
```

The API request uses:

```text
model = text-embedding-v4
dimensions = 64
```

The Qwen backend reads the API key from:

```text
DASHSCOPE_API_KEY
```

Example:

```bash
export DASHSCOPE_API_KEY="your_api_key_here"

python3 -m vector_pipeline.tokenizer.run_embedding \
  vector_pipeline/examples/single_file/tokenized.json \
  -o vector_pipeline/examples/single_file/embeddings.json \
  --text-backend qwen \
  --text-dim 64
```

Text strings are deduplicated and cached locally to avoid repeated API calls.

Default cache:

```text
vector_pipeline/outputs/cache/qwen_text_embeddings_64.json
```

If a primitive has no text, its text channel remains a zero vector.

### Fusion Rule

The final primitive embedding is produced by concatenating all channel embeddings and projecting to 256 dimensions:

```text
concat = [
  type_emb,
  geom_emb,
  style_emb,
  pos_emb,
  text_emb,
  group_emb
]

z_i = L2_normalize(tanh(W * concat))
```

Output per primitive:

```json
{
  "primitive_id": "p_000005",
  "group_token": "GROUP_layer0",
  "embedding": [0.01, -0.03, 0.12]
}
```

The real `embedding` list has 256 numbers.

## Batch Dataset Output

The batch pipeline produces three levels of intermediate output:

```text
parsed/       # parser JSON per SVG
tokenized/    # tokenized JSON per SVG
embeddings/   # embedding JSON per SVG
```

It also writes summary files:

| File | Meaning |
|---|---|
| `manifest.json` | Per-file processing metadata and primitive counts |
| `summary.json` | Dataset-level statistics, type counts, group counts, failures |
| `embedding_dataset.jsonl` | One JSON line per SVG, suitable for downstream dataset loading |

`embedding_dataset.jsonl` format:

```json
{
  "source_file": "SVG Dataset/0000-0002.svg",
  "num_primitives": 1076,
  "primitive_ids": ["p_000028", "p_000047"],
  "group_tokens": ["GROUP_layer0", "GROUP_layer0"],
  "embeddings": [[0.01, -0.03], [0.04, 0.09]]
}
```

Each vector inside `embeddings` has 256 dimensions.

## Prediction Heads

The first reusable prediction head is implemented in:

```text
vector_pipeline/modeling/prediction_heads.py
```

### Typed Pairwise Relation Head

The current experiment uses `TypePairRelationHead`, which is encoder-agnostic. It does not care whether primitive embeddings come from the current deterministic embedder or a future trainable encoder.

Expected input:

```text
e_i: [B, D] or [..., D]
e_j: [B, D] or [..., D]
D = embedding dimension, default 256
```

Pair feature rule:

```text
pair_feature = concat(e_i, e_j, |e_i - e_j|, e_i * e_j)
```

Therefore:

```text
pair_feature_dim = 4D
```

The typed head keeps a separate classifier for each geometric type pair:

```text
LINE-LINE
LINE-ARC
LINE-POLYLINE
ARC-ARC
ARC-POLYLINE
POLYLINE-POLYLINE
```

Output:

```text
relation_logits: [B, num_labels_for_this_type_pair]
```

Example:

```python
from vector_pipeline.modeling.prediction_heads import TypePairRelationHead

head = TypePairRelationHead(embedding_dim=256)
logits = head(e_i, e_j, type_pair="LINE-LINE")
```

For sequence-level encoder outputs:

```python
E = encoder(batch)               # [B, N, D]
pair_indices = ...               # [B, P, 2]
logits = head.forward_from_sequence(E, pair_indices, type_pair="LINE-LINE")
```

This interface is designed so that a future trainable encoder can be connected directly:

```text
tokenized primitives -> encoder -> E [B, N, D] -> pairwise relation head
```

### Typed Pairwise Relation Pseudo-Labels

The first pairwise relation label builder is implemented in:

```text
vector_pipeline/experiments/pairwise_relation/
  relation_labels.py
  build_pairwise_labels.py
```

The current label builder creates typed multi-label pseudo-labels for supported geometric type pairs:

```text
LINE-LINE
LINE-ARC
LINE-POLYLINE
ARC-ARC
ARC-POLYLINE
POLYLINE-POLYLINE
```

Each type pair has its own relation vocabulary.

Current relation labels:

| Type pair | Labels |
|---|---|
| `LINE-LINE` | `connected`, `intersect`, `collinear_overlap`, `perpendicular`, `parallel`, `near` |
| `LINE-ARC` | `connected`, `intersect`, `tangent`, `near` |
| `LINE-POLYLINE` | `connected`, `intersect`, `overlap_segment`, `perpendicular_to_segment`, `parallel_to_segment`, `near` |
| `ARC-ARC` | `connected`, `intersect`, `overlap`, `concentric`, `tangent`, `near` |
| `ARC-POLYLINE` | `connected`, `intersect`, `tangent_to_segment`, `near` |
| `POLYLINE-POLYLINE` | `connected`, `intersect`, `overlap_segment`, `perpendicular_segment`, `parallel_segment`, `containment`, `near` |

The target format is multi-label:

```text
label_vector = [0, 1, 0, ...]
```

If no relation is detected, the sample is treated as a negative example with an all-zero `label_vector`.

Important geometric thresholds:

```text
parallel_dot = 0.95
perpendicular_dot = 0.15
joined_diag_ratio = 0.01
near_diag_ratio = 0.03
overlap_diag_ratio = 0.003
tangent_diag_ratio = 0.01
concentric_diag_ratio = 0.01
```

All distance thresholds are scaled by the SVG `viewBox` diagonal.

Build labels for a split:

```bash
python3 -m vector_pipeline.experiments.pairwise_relation.build_pairwise_labels \
  --parsed-dir vector_pipeline/outputs/processed_dataset_qwen/parsed \
  --tokenized-dir vector_pipeline/outputs/processed_dataset_qwen/tokenized \
  --split-json vector_pipeline/outputs/splits/file_split_qwen_seed13.json \
  --split train \
  --output-jsonl vector_pipeline/outputs/pairwise_relation_qwen_train/pairwise_labels.jsonl \
  --summary-json vector_pipeline/outputs/pairwise_relation_qwen_train/summary.json \
  --max-pairs-per-file 1000
```

Output JSONL format:

```json
{
  "source_file": "vector_pipeline/outputs/processed_dataset_qwen/parsed/0000-0002.parsed.json",
  "i": 224,
  "j": 470,
  "primitive_id_i": "p_000238",
  "primitive_id_j": "p_000414",
  "type_i": "LINE",
  "type_j": "ARC",
  "type_pair": "LINE-ARC",
  "labels": ["near"],
  "label_ids": [3],
  "label_vector": [0, 0, 0, 1]
}
```

The indices `i` and `j` follow the tokenized/embedding sequence order, so they can be used directly with:

```text
embeddings[i]
embeddings[j]
```

## Current Validation

The current full Qwen relation-head experiment used:

```text
embedding backend: Qwen text-embedding-v4 text channel, fused to 256-D primitive embeddings
split unit: SVG source file
train labels: 2,902,228
val labels:   362,158
test labels:  365,137
checkpoint: epoch 25
best val loss: 0.129006
```

Final full test result:

```text
test loss:        0.136190
micro precision:  0.877056
micro recall:     0.698169
micro F1:         0.777455
```

Per type-pair test result:

| Type pair | Samples | Loss | Micro precision | Micro recall | Micro F1 |
|---|---:|---:|---:|---:|---:|
| `LINE-LINE` | 322,921 | 0.134371 | 0.882883 | 0.713288 | 0.789076 |
| `LINE-ARC` | 36,662 | 0.138127 | 0.487367 | 0.217590 | 0.300859 |
| `ARC-ARC` | 5,554 | 0.229198 | 0.441006 | 0.111328 | 0.177778 |


## Notes

- The parser currently focuses on `LINE`, `ARC`, `POLYLINE`, `TEXT`, and `ANNOTATION`.
- The tokenizer is type-aware for geometry because the eight geometry slots have different meanings for different primitive types.
- The final 256-dimensional vector is fused; after fusion, individual dimensions no longer correspond directly to one channel.
- To inspect channel-specific information, read the tokenized JSON or modify the embedder to also save pre-fusion channel embeddings.
- Last update: 27 May 2026.
