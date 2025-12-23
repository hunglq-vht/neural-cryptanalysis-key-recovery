"""Utility functions for cryptanalysis"""

import numpy as np
from scipy import stats
import os

def create_directories():
    """Create necessary directories"""
    dirs = ['./data', './models', './results']
    for d in dirs:
        os.makedirs(d, exist_ok=True)

def bits_to_vector(data, bits=8):
    """Convert bits to vector representation"""
    if isinstance(data, int):
        data = np.uint32(data)
    binary = np.unpackbits(np.uint8(data))
    return binary.astype(np.float32)

def vector_to_bits(vector):
    """Convert vector back to bits"""
    binary = (vector > 0.5).astype(np.uint8)
    return np.packbits(binary)

def hamming_weight(x):
    """Calculate Hamming weight"""
    return bin(x).count('1')

def hamming_distance(x, y):
    """Calculate Hamming distance between two values"""
    return hamming_weight(x ^ y)

def power_consumption_model(plaintext, key, ciphertext, leakage_factor=0.1):
    """Simulate power consumption leakage based on key and plaintext"""
    intermediate = plaintext[0] ^ key[0]
    power = hamming_weight(intermediate) * leakage_factor
    noise = np.random.normal(0, 0.05)
    return power + noise

def correlation_analysis(traces, hypothesis):
    """Perform correlation analysis between traces and hypothesis"""
    correlation = np.corrcoef(traces, hypothesis)[0, 1]
    return correlation

def normalize_data(data, axis=None):
    """Normalize data to [0, 1] range"""
    min_val = np.min(data, axis=axis, keepdims=True)
    max_val = np.max(data, axis=axis, keepdims=True)
    return (data - min_val) / (max_val - min_val + 1e-8)

def calculate_guessing_entropy(predictions, true_key):
    """Calculate guessing entropy metric"""
    correct_position = np.argmax(predictions) if np.argmax(predictions) == true_key else len(predictions)
    return correct_position

def matrix_multiply_gf2(A, x):
    """Matrix-vector multiplication in GF(2)"""
    result = np.zeros(A.shape[0], dtype=np.uint8)
    for i in range(A.shape[0]):
        result[i] = np.sum(A[i] * x) % 2
    return result

def print_metrics(history, model_name):
    """Print training metrics"""
    print(f"\n{'='*60}")
    print(f"Training Results for {model_name}")
    print(f"{'='*60}")
    for key, value in history.items():
        print(f"{key}: {value}")
    print(f"{'='*60}\n")