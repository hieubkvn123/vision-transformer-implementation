import numpy as np

### Tensorflow dependecies ### 
import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam


# Positional embedding layer
def pos_embedding_matrix(batch_size=16, d_model = 128, seq_len = 16):
    # Batch-size position embedding
    batch_pos_embs = []
    
    # Get the pos embedding matrix
    pos_embs = []
    
    # Position 0 is reserved for classification token
    for i in range(0, seq_len + 1):
        pos_vec = np.array([x for x in range(1, d_model + 1)])
        even_mask = np.array([1 if x % 2 == 0 else 0 for x in range(1, d_model + 1)])
        odd_mask = np.array([0 if x % 2 == 0 else 1 for x in range(1, d_model + 1)])
        
        pos_even = np.sin(i/(10000 ** (pos_vec / d_model))) * even_mask
        pos_odd = np.cos(i/(10000 ** ((pos_vec - 1) / d_model))) * odd_mask
        
        pos_emb = pos_even + pos_odd
        pos_embs.append(pos_emb)
        
    for i in range(batch_size):
        batch_pos_embs.append(pos_embs)

    return np.array(batch_pos_embs)


# Single-headed self-attention module
class SelfAttention(Model):
    def __init__(self, d_model=128):
        super(SelfAttention, self).__init__()
        self.d_model = d_model
        
    def build(self, input_shape):
        self.W_q = self.add_weight(shape=(input_shape[-1], self.d_model),
                                  initializer='identity',
                                  trainable=True)
        
        self.W_k = self.add_weight(shape=(input_shape[-1], self.d_model),
                                  initializer='identity',
                                  trainable=True)
        
        self.W_v = self.add_weight(shape=(input_shape[-1], self.d_model),
                                  initializer='identity',
                                  trainable=True)
    
    def call(self, X):
        Q = tf.matmul(X, self.W_q) 
        K = tf.matmul(X, self.W_k) 
        V = tf.matmul(X, self.W_v) 

        q_k_T = tf.matmul(Q, K, transpose_b=True)
        d_k = tf.cast(self.d_model, dtype=tf.float32)

        scores = tf.nn.softmax(q_k_T / tf.math.sqrt(d_k)) 
        outputs = tf.matmul(scores, V)
        
        return outputs
    

