# Constants
from config import (
    VAE_WEIGHTS,
    VAE_TRAINING_DATA, 
    FRAME_DIM, 
    N_EPISODES,
    N_TIMESTEPS
)

# Model Import
from Conv_VAE import Conv_VAE

# Other Imports
import os
import numpy as NP

# Trains the Convolutional-Variational Autoencoder
def train_model():
    # Initializing the model & training data.
    ConvVAE = Conv_VAE()
    data = allocate_data()
    
    # Training the model and saving results.
    ConvVAE.train(data)
    ConvVAE.save_weights(VAE_WEIGHTS)
    
    # Closing & removing the data file.
    del data
    os.remove('TEMP.dat')
    
# Allocates the training data for use by Tensorflow.
def allocate_data():
    # Gathering the list of files.
    files = os.listdir(VAE_TRAINING_DATA)
    
    # Ensuring enough training samples have been generated.
    if len(files) != N_EPISODES:
        raise Exception(f'Actual/Desired Training Samples: {len(files)}/{N_EPISODES}')
    
    # Initializing the data array.
    data = NP.memmap(
        'TEMP.dat',
        mode = 'w+',
        shape = (N_EPISODES * N_TIMESTEPS,) + FRAME_DIM, 
        dtype = NP.float32
    )
    
    # Iterating through training data files.
    for count, file in enumerate(files):
        # Determining index to place file data.
        START_INDEX = count * N_TIMESTEPS
        END_INDEX = (count + 1) * N_TIMESTEPS
        
        # Getting full file path. 
        file_path = os.path.join(VAE_TRAINING_DATA, file)
        
        # Storing the file data.
        observations = NP.load(file_path)['observations']
        data[START_INDEX:END_INDEX, :, :, :] = observations
        
    # Flushing changes to disk & returning allocated data.
    del data
    return NP.memmap(
        'TEMP.dat',
        shape = (N_EPISODES * N_TIMESTEPS,) + FRAME_DIM,
        dtype = NP.float32
    )
        
if __name__ == '__main__':
    train_model()