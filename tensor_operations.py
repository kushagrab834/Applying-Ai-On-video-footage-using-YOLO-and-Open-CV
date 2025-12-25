print("\n--- Step 2: Math Operations ---")

tensor_a = tf.constant([[1, 2], 
                        [3, 4]])

tensor_b = tf.constant([[5, 6], 
                        [7, 8]])

# 1. Element-wise Addition
# Adds corresponding elements: (1+5), (2+6), etc.
add_result = tf.add(tensor_a, tensor_b)
# OR simply: tensor_a + tensor_b
print(f"Addition Result:\n{add_result.numpy()}")

# 2. Element-wise Multiplication
# Multiplies corresponding elements: (1*5), (2*6), etc.
mult_result = tf.multiply(tensor_a, tensor_b)
# OR simply: tensor_a * tensor_b
print(f"\nElement-wise Multiplication:\n{mult_result.numpy()}")

# 3. Matrix Multiplication (Dot Product)
# This is the most critical operation in Deep Learning (Input * Weights).
# Row of A * Column of B.
matmul_result = tf.matmul(tensor_a, tensor_b)
print(f"\nMatrix Multiplication (Dot Product):\n{matmul_result.numpy()}")