# Constants
from config import (
    RNN_WEIGHTS,
    RNN_TRAINING_DATA, 
    LATENT_FRAME_DIM,
    ACTION_DIM,
    N_EPISODES,
    N_TIMESTEPS
)

# Model Import
from MDN_RNN import MDN_RNN

# Other Imports
import os
import numpy as NP

def train_model():
    # Initializing the model & training data.
    MDNRNN = MDN_RNN()
    _input, _output = load_data()
    
    # Training the model and saving results.
    MDNRNN.train(_input[:, :-2, :], _output[:, 2:, :])
    MDNRNN.save_weights(RNN_WEIGHTS)
    
    # Closing & removing the data files.
    del _input; del _output;
    os.remove('TEMP_1.dat'); os.remove('TEMP_2.dat');
    
# Loads the training data.
def load_data():
    # Gathering the list of files.
    files = os.listdir(RNN_TRAINING_DATA)
    
    # Ensuring enough training samples have been generated.
    if len(files) != N_EPISODES:
        raise Exception(f'Actual/Desired Training Samples: {len(files)}/{N_EPISODES}')
    
    # Initializing the data arrays.
    _input = NP.memmap(
        'TEMP_1.dat',
        mode = 'w+',
        shape = (N_EPISODES, N_TIMESTEPS, (LATENT_FRAME_DIM + ACTION_DIM)), 
        dtype = NP.float32
    )
    _output = NP.memmap(
        'TEMP_2.dat',
        mode = 'w+',
        shape = (N_EPISODES, N_TIMESTEPS, LATENT_FRAME_DIM),
        dtype = NP.float32
    )
    
    # Iterating through training data files.
    for INDEX, file in enumerate(files):        
        # Getting full file path. 
        file_path = os.path.join(RNN_TRAINING_DATA, file)
        
        # Storing the file data.
        data = NP.load(file_path)
        _input_INDEX = NP.concatenate(
            (data['observations'], data['actions']), axis = 1)
        _input[INDEX, :, :] = _input_INDEX
        _output[INDEX, :, :] = data['observations']
        
    # Flushing changes to disk & returning allocated data.
    del _input, _output
    return (
        NP.memmap(
            'TEMP_1.dat',
            shape = (N_EPISODES, N_TIMESTEPS, (LATENT_FRAME_DIM + ACTION_DIM)), 
            dtype = NP.float32
        ),
        NP.memmap(
            'TEMP_2.dat',
            shape = (N_EPISODES, N_TIMESTEPS, LATENT_FRAME_DIM),
            dtype = NP.float32
        )
    )

if __name__ == '__main__':
    train_model()