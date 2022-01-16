import os
import glob
import tqdm
import time
import wandb
import numpy as np
import matplotlib.pyplot as plt

from dataloader import DataLoader

### Tensorflow dependecies ### 
import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam

# No @tf.function because we don't want to compute all
# symbolic tensors in the training step.
def train_step(model, patches, labels, opt):
    with tf.GradientTape() as tape:
        logits = model(patches)
        loss = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels)
        
        correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
        accuracy = tf.cast(correct_prediction, dtype=tf.float32)
        
    gradients = tape.gradient(loss, model.trainable_variables)
    opt.apply_gradients(zip(gradients, model.trainable_variables))

    return tf.reduce_mean(loss), tf.reduce_mean(accuracy)

def val_step(model, patches, labels): 
    logits = model(patches)
    loss = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels)

    correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
    accuracy = tf.cast(correct_prediction, dtype=tf.float32)
    
    return tf.reduce_mean(loss), tf.reduce_mean(accuracy)
    
def train(model, data_dir, batch_size=16, d_model=128, img_size=64, num_classes=2, epochs=100, lr=0.0001, wandb_log=None):
    bce = tf.keras.losses.BinaryCrossentropy()
    acc = tf.keras.metrics.Accuracy()
    opt = tf.keras.optimizers.Adam(lr=lr, amsgrad=True)

    loader = DataLoader(data_dir, batch_size=batch_size, img_size=img_size)
    train_steps = loader.train_steps
    val_steps = loader.val_steps
    num_classes = loader.n_classes
    
    losses = {'train' : [], 'val' : []}
    accs = {'train' : [], 'val' : []}
    
    # Initialize wandb 
    if(wandb_log is not None):
        wandb.init(project=wandb_log['project'], entity=wandb_log['entity'])
        wandb.config.update({
            'd_model' : d_model,
            'img_size' : img_size, 
            'num_classes' : num_classes,
            'epochs' : epochs,
            'lr' : lr,
            'batch_size' : batch_size,
            'L' : model.L,
            'num_attn_heads' : model.num_attn_heads
        })

    try:
        for i in range(epochs):
            print(f'Epoch #[{i+1}/{epochs}]')
            time.sleep(1.0)

            with tqdm.tqdm(total=train_steps, colour='green') as pbar:
                for j in range(train_steps):
                    # Extract data batch
                    _, patches, labels = loader.get_batch(train=True)

                    # Perform trainstep and compute loss + accuracy
                    loss, accuracy = train_step(model, patches, labels, opt)
                    loss, accuracy = loss.numpy(), accuracy.numpy()

                    # Log message
                    pbar.set_postfix({
                        'batch_id' : f'{j+1}/{train_steps}',
                        'loss' : f'{loss:.6f}',
                        'accuracy' : f'{accuracy:.6f}'
                    })
                    
                    # Store for plotting
                    losses['train'].append(loss)
                    accs['train'].append(accuracy)

                    # Log to wandb
                    if(wandb_log is not None) : wandb.log({'train_loss' : loss, 'train_acc' : accuracy})

                    # Update progress bar
                    pbar.update(1)

            with tqdm.tqdm(total=val_steps) as pbar:
                for j in range(val_steps): 
                    # Extract data batch 
                    _, patches, labels  = loader.get_batch(train=False)

                    # Perform valstep and compute loss + accuracy
                    loss, accuracy = val_step(model, patches, labels)
                    loss, accuracy = loss.numpy(), accuracy.numpy()

                    # Log message
                    pbar.set_postfix({
                        'batch_id' : f'{j+1}/{train_steps}',
                        'loss' : f'{loss:.6f}',
                        'accuracy' : f'{accuracy:.6f}'
                    })
                    
                    # Store for plotting
                    losses['val'].append(loss)
                    accs['val'].append(accuracy)

                    # Log to wandb
                    if(wandb_log is not None) : wandb.log({'val_loss' : loss, 'val_acc' : accuracy})

                    # Update progress bar
                    pbar.update(1)
                    
    except KeyboardInterrupt:
        print('[INFO] Training halted ... ')
                
    if(wandb_log is None):
        fig, ax = plt.subplots(2, 1, figsize=(30, 15))
        
        ax[0].plot(losses['train'], label='Train loss', color='orange')
        ax[0].plot(losses['val'], label='Val loss', color='blue')
        
        ax[1].plot(accs['train'], label='Train accuracy', color='orange')
        ax[1].plot(accs['val'], label='Val accuracy', color='blue')
        
        ax[0].set_title('Losses')
        ax[1].set_title('Accuracy')
    
        if(not os.path.exists('logs')):
            os.mkdir('logs')

        print('[INFO] Saving figures to logs/training_result.png')
        plt.savefig('logs/training_result.png')
