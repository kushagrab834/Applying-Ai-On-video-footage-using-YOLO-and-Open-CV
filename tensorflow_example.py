import numpy as np
from tensorflow import keras

print("\n--- Step 3: Building a Neural Network ---")

# 1. PREPARE DATA
# We use numpy arrays because TensorFlow understands them easily.
xs = np.array([-1.0,  0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)
ys = np.array([-3.0, -1.0, 1.0, 3.0, 5.0, 7.0], dtype=float)

# 2. BUILD THE MODEL
# Sequential: Defines a stack of layers where data flows from one to the next.
model = keras.Sequential([
    # Dense Layer: A standard layer of neurons.
    # units=1: This layer has 1 neuron (because our output is a single number).
    # input_shape=[1]: The model expects 1 input number at a time (x).
    keras.layers.Dense(units=1, input_shape=[1])
])

# 3. COMPILE THE MODEL
# optimizer='sgd': Stochastic Gradient Descent. It tells the model HOW to improve.
# loss='mean_squared_error': Measures how wrong the model is. (Guess - Real Answer)^2
model.compile(optimizer='sgd', loss='mean_squared_error')

print("Training model... (this takes a moment)")

# 4. TRAIN THE MODEL (FIT)
# epochs=500: The model will look at the data and try to learn 500 times.
# verbose=0: Hides the logs so the output is clean.
model.fit(xs, ys, epochs=500, verbose=0)

# 5. PREDICT
# Now we ask the model to predict 'y' for a new 'x' (e.g., 10.0).
# The correct answer should be (2 * 10) - 1 = 19.
prediction = model.predict(np.array([10.0]))
print(f"\nPrediction for x=10: {prediction[0][0]:.4f}")
print("Correct answer is 19. The model is close!")