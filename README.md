# Neural Cryptanalysis for Key Recovery Attack

A comprehensive implementation of neural network-based cryptanalysis using deep learning models to perform key recovery attacks.

## Overview

This project implements two main approaches for cryptographic key recovery using neural networks:

1. **RNN-Based Deep Learning Models**
   - LSTM (Long Short-Term Memory)
   - BiLSTM (Bidirectional LSTM)
   - Standard RNN
   - GRU (Gated Recurrent Unit)

2. **Fully-Connected Neural Networks**
   - Standard Multi-Layer Perceptron (MLP)
   - Custom FC with Batch Normalization
   - Dropout and Regularization

## Features

- **Synthetic Data Generation**: Creates realistic cryptographic data with power consumption and timing side-channels
- **Multiple Model Architectures**: Supports various neural network types optimized for different attack scenarios
- **Side-Channel Simulation**: Implements power analysis and timing analysis simulations
- **Comprehensive Metrics**: Includes accuracy, loss, guessing entropy, and ranking metrics
- **Training Visualization**: Generates plots for training history and model comparison

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Quick Start

```bash
python key_recovery_attack.py
```

This will:
1. Generate synthetic cryptographic data
2. Train all available neural network models
3. Evaluate performance on test data
4. Generate visualization plots
5. Save results to ./results/ directory