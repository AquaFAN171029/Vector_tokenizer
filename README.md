# CAD/SVG Vector Pipeline

This repository contains a prototype pipeline for converting CAD-derived SVG drawings into primitive-level vector embeddings.

The current pipeline is:

```text
SVG drawing
  -> parser JSON
  -> tokenized primitive records
  -> channel embeddings
  -> fused 256-dimensional primitive embeddings
```

The implementation is intentionally simple and explicit. It is designed to make the representation pipeline easy to inspect before replacing the prototype embedder with a trainable neural module.

## Directory Layout

```text
vector_pipeline/
  svg_parser/
    svg_parser_v2.py
    input.svg
    output.json

  tokenizer/
    run_tokenizer.py
    run_embedding.py
    dataset_reader.py
    sequence_builder.py
    primitive_tokenizer.py
    type_tokenizer.py
    geom_tokenizer.py
    style_tokenizer.py
    pos_tokenizer.py
    text_tokenizer.py
    embedding_generator.py

  pipeline/
    run_svg_dataset_pipeline.py
    README.md

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

Run the single-file pipeline:

```bash
python3 vector_pipeline/svg_parser/svg_parser_v2.py vector_pipeline/svg_parser/input.svg -o vector_pipeline/svg_parser/output.json
python3 -m vector_pipeline.tokenizer.run_tokenizer vector_pipeline/svg_parser/output.json -o vector_pipeline/tokenizer/output.tokenized.json
python3 -m vector_pipeline.tokenizer.run_embedding vector_pipeline/tokenizer/output.tokenized.json -o vector_pipeline/tokenizer/output.embeddings.json
```

Run a small dataset smoke test:

```bash
python3 -m vector_pipeline.pipeline.run_svg_dataset_pipeline --input-dir "SVG Dataset" --output-dir outputs/processed_dataset_sample --limit 3 --overwrite
```

Run the full SVG dataset:

```bash
python3 -m vector_pipeline.pipeline.run_svg_dataset_pipeline --input-dir "SVG Dataset" --output-dir outputs/processed_dataset --overwrite
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

## Stage 1: Parser

The parser converts SVG path/text elements into a canonical primitive schema.

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

The tokenizer converts each primitive into a symbolic token record.

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

Before tokenization, primitives are sorted into a deterministic sequence:

```text
parent_group_id -> cy -> cx -> primitive_id
```

This means:

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

The parser always writes an 8-slot geometry vector. Different primitive types interpret the slots differently.

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

The group token is preserved because grouping/layer structure is important for region-aware sequence construction.

## Stage 3: Embedder

The embedder maps each tokenized primitive into one fused 256-dimensional vector.

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

For the current prototype, each symbolic token is mapped to a deterministic vector using:

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

Note: this embedder is a deterministic prototype. For training, this module can be replaced by trainable `nn.Embedding` layers and a trainable MLP while keeping the same channel structure.

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

## Current Validation

The current smoke test successfully processed one SVG file after the directory cleanup:

```text
Success: 1
Failed: 0
Total primitives: 1076
Embedding dimension: 256
```

The earlier three-file sample produced:

```text
Success: 3
Failed: 0
Total primitives: 3374
```

## Notes

- The parser currently focuses on `LINE`, `ARC`, `POLYLINE`, `TEXT`, and `ANNOTATION`.
- The tokenizer is type-aware for geometry because the eight geometry slots have different meanings for different primitive types.
- The final 256-dimensional vector is fused; after fusion, individual dimensions no longer correspond directly to one channel.
- To inspect channel-specific information, read the tokenized JSON or modify the embedder to also save pre-fusion channel embeddings.

