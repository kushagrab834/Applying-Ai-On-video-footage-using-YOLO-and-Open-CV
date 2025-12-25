import tensorflow as tf
import numpy as np

print("--- Step 1: Tensor Basics ---")

# 1. SCALAR (Rank-0 Tensor)
# A single number.
# dtype=tf.float32 defines the data type (32-bit floating point number).
scalar = tf.constant(7, dtype=tf.float32)
print(f"Scalar: {scalar}")
print(f"Scalar Shape: {scalar.shape}") # Output: () because it has no dimensions

# 2. VECTOR (Rank-1 Tensor)
# A list of numbers.
vector = tf.constant([10, 10], dtype=tf.float32)
print(f"\nVector: {vector}")
print(f"Vector Shape: {vector.shape}") # Output: (2,) -> It has 2 elements

# 3. MATRIX (Rank-2 Tensor)
# A list of lists (rows and columns).
matrix = tf.constant([[1, 2],
                      [3, 4]], dtype=tf.float32)
print(f"\nMatrix:\n{matrix}")
print(f"Matrix Shape: {matrix.shape}") # Output: (2, 2) -> 2 rows, 2 columns

# 4. VARIABLE
# Unlike 'constants', Variables can be changed (updated) during training.
# This is used for Weights and Biases in a Neural Network.
trainable_var = tf.Variable([[1.0, 2.0], [3.0, 4.0]])
print(f"\nVariable (before change):\n{trainable_var.numpy()}")

# Modifying a variable (e.g., updating weights)
trainable_var.assign([[0.0, 0.0], [1.0, 1.0]])
print(f"Variable (after change):\n{trainable_var.numpy()}")