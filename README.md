# Nested Learning - Hope Architecture

This repository contains a PyTorch implementation of the **Nested Learning** paradigm, as described in the paper "Nested Learning: The Illusion of Deep Learning Architectures".

## Overview

Nested Learning (NL) represents machine learning models as coherent systems of nested, multi-level optimization problems. This implementation focuses on:
- **Hope Architecture**: A self-referential Titans model followed by a Continuum Memory System (CMS).
- **M3 Optimizer**: Multi-scale Momentum Muon optimizer that manages momentums at different time scales.
- **Continuum Memory System (CMS)**: A spectrum of memory blocks with varying update frequencies to mitigate catastrophic forgetting.

## Project Structure

```text
.
├── nested_learning/
│   ├── modules/
│   │   ├── cms.py                   # Continuum Memory System
│   │   └── self_referential_titans.py # Self-Referential Titans
│   ├── optimizers/
│   │   └── m3.py                    # Multi-scale Momentum Muon Optimizer
│   └── models/
│       └── hope.py                  # Hope Model Assembly
├── demo_training.py                 # Training verification script
├── requirements.txt                 # Dependencies
└── README.md                        # Documentation
```

## Installation

Ensure you have Python 3.10+ installed. Install the requirements:

```bash
pip install -r requirements.txt
```

## Usage

Run the training demo to verify the implementation:

```bash
export PYTHONPATH=$PYTHONPATH:.
python3 demo_training.py
```

The demo initializes a Hope model and trains it on a synthetic sequence shift task using the M3 optimizer, showing a steady decrease in loss.

## Acknowledgments

Based on the paper "Nested Learning: The Illusion of Deep Learning Architectures" (NeurIPS 2025).
