#from: https://youtu.be/kCc8FmEb1nY

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Parameters
vocab_size = 10000  # Vocabulary size
max_seq_length = 256  # Maximum sequence length
embedding_dim = 128  # Embedding dimension
num_heads = 8  # Number of attention heads
num_transformer_blocks = 4  # Number of transformer blocks
dense_dim = 512  # Dense layer dimension

# Input layer
inputs = layers.Input(shape=(max_seq_length,))

# Embedding layer
embedding_layer = layers.Embedding(input_dim=vocab_size, output_dim=embedding_dim)(inputs)

# Transformer blocks
x = embedding_layer
for _ in range(num_transformer_blocks):
    # Self-attention layer
    attention_output = layers.MultiHeadAttention(key_dim=embedding_dim, num_heads=num_heads)(x, x)
    attention_output = layers.Dropout(0.1)(attention_output)
    x = layers.Add()([x, attention_output])
    x = layers.LayerNormalization(epsilon=1e-6)(x)

    # Feed-forward neural network
    ffn = keras.Sequential(
        [layers.Dense(dense_dim, activation="relu"), layers.Dense(embedding_dim),]
    )
    x = ffn(x)
    x = layers.Dropout(0.1)(x)
    x = layers.Add()([x, embedding_layer])
    x = layers.LayerNormalization(epsilon=1e-6)(x)

# Output layer
outputs = layers.Dense(vocab_size, activation="softmax")(x)

# GPT-like model
model = keras.Model(inputs, outputs)

# Compile the model (you can use appropriate loss and optimizer based on your task)
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy")
