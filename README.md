# DRBTKG

**Drug Repurposing Prediction Driven by Biomedical Text-Integrated Knowledge Graph Representations**

DRBTKG is a heterogeneous graph learning framework for drug-disease indication prediction. It integrates BioBERT-derived drug-text embeddings with heterogeneous graph representations to rank candidate drug-disease associations.

## Overview

DRBTKG formulates drug repurposing as relation-aware link prediction on a heterogeneous biomedical knowledge graph. Text-derived drug features are aligned with structural graph representations and decoded into ranking scores for candidate indications.

## ✨ Methodological Design

- **Text-enriched drug representations:** Fixed embeddings from drug descriptions, mechanisms of action, and pharmacodynamics add pharmacological context beyond graph topology.
- **Task-adaptive feature fusion:** A two-layer nonlinear projection aligns text embeddings with the graph space before fusion with structural drug features.
- **Relation-aware heterogeneous propagation:** Explicit modelling of node and relation types preserves the multi-scale semantics of the biomedical graph.
- **Target-edge isolation:** Indication relations are retained only as supervision labels, while indication and contraindication relations are removed from message passing, preventing direct target-edge information leakage.
- **Type-specific residual updates:** Node-type-specific residual mappings preserve node identity and stabilise layer-wise aggregation.

## Repository structure

```text
DRBTKG_GitHub/
├── data/
│   ├── kg.csv
│   ├── train_data/
│   │   ├── train1.pkl ... train5.pkl
│   │   └── test1.pkl  ... test5.pkl
│   └── drug_text_embeddings/
│       ├── drug_text_embeddings_64d.pt
│       └── drug_text_mappings.pkl
├── model.py
├── train.py
├── requirements.txt
├── LICENSE
└── README.md
```

## Requirements

- Python 3.9
- PyTorch 2.6.0 with CUDA 11.8
- PyTorch Geometric 2.7.0
- NumPy 1.26.4
- pandas 2.3.3
- scikit-learn 1.7.2

```bash
pip install -r requirements.txt
```

## Training

```bash
python train.py
```

## Licence

This repository's source code and author-generated documentation are released under the [MIT License](LICENSE).
