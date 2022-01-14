import os
import glob
import numpy as np
import tensorflow as tf

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

class DataLoader:
    def __init__(self, directory, batch_size=32, split_ratio=0.2,
                 img_size=64, shuffle=True):
        self.directory = directory
        self.shuffle = shuffle
        self.batch_size = batch_size 
        self.split_ratio = 0.2
        self.img_size = img_size
        self.patch_size = img_size // 4
        self.n_classes = 10 # To be adjusted when dataset is parsed
        
        self.train_dataset, self.val_dataset = None, None
        self.train_paths, self.train_labels = None, None
        self.val_paths, self.val_labels = None, None
        
        self.parse_dataset()
        
    def map_fn(self, img, img_size):
        # After the image has been decoded into TF tensor
        # Regular image decoding
        img = tf.image.resize(img, [img_size, img_size])
        img = tf.clip_by_value(img, 0.0, 255.0)
        
        img = img / 127.5 - 1
        
        return img
        
    def parse_fn_with_label(self, path, label):
        img = tf.io.read_file(path)
        img = tf.io.decode_png(img, 3)
        
        img = self.map_fn(img, self.img_size)
        label = tf.one_hot(label, depth=self.n_classes)
        
        return img, label
        
        
    def parse_dataset(self):
        all_imgs = glob.glob(os.path.join(self.directory, "*", "*.jpg"))
        
        img_paths = []
        img_labels = []
        for entry in all_imgs:
            class_name = entry.split('/')[-2]
            
            img_labels.append(class_name)
            img_paths.append(entry)
        
        img_labels = np.array(img_labels)
        img_paths = np.array(img_paths)
        
        img_labels = LabelEncoder().fit_transform(img_labels).flatten()
        
        self.n_classes = len(np.unique(img_labels))
        self.all_paths = img_paths
        self.all_labels = img_labels
        
        # Config sets of train - val img paths and labels
        self.train_paths, self.val_paths, self.train_labels, self.val_labels = train_test_split(
            self.all_paths, self.all_labels, test_size = self.split_ratio)
        
        # Get train and val dataset
        self.train_dataset = self.get_train_dataset()
        self.val_dataset = self.get_val_dataset()
        
        # Get train and val size
        self.train_steps = len(self.train_dataset)
        self.val_steps = len(self.val_dataset)
        
    def get_train_dataset(self):
        if(self.train_dataset is None):
            self.train_dataset = tf.data.Dataset.from_tensor_slices((self.train_paths, self.train_labels))
            
            if(self.shuffle):
                self.train_dataset = self.train_dataset.shuffle(40000)
            
            self.train_dataset = self.train_dataset.map(self.parse_fn_with_label)
            self.train_dataset = self.train_dataset.batch(self.batch_size)
            self.train_dataset = self.train_dataset.repeat(1).prefetch(1)
            
        return self.train_dataset
    
    def get_val_dataset(self):
        if(self.val_dataset is None):
            self.val_dataset = tf.data.Dataset.from_tensor_slices((self.val_paths, self.val_labels))
            
            if(self.shuffle):
                self.val_dataset = self.val_dataset.shuffle(40000)
                
            self.val_dataset = self.val_dataset.map(self.parse_fn_with_label)
            self.val_dataset = self.val_dataset.batch(self.batch_size)
            self.val_dataset = self.val_dataset.repeat(1).prefetch(1)
            
        return self.val_dataset
            
    def get_batch(self, train=True):
        if(train):
            images, labels = next(iter(self.train_dataset))
        else:
            images, labels = next(iter(self.val_dataset))
            
        # Flattening images into patches
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )
        
        patch_dims = patches.shape[-1]
        patches = tf.reshape(patches, [self.batch_size, -1, patch_dims])
        
        return images, patches, labels

