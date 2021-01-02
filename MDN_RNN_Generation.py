# Constants
from config import (
    VAE_WEIGHTS,
    VAE_TRAINING_DATA, 
    RNN_TRAINING_DATA,
    FRAME_DIM,
    N_EPISODES,
    N_TIMESTEPS, 
    RENDER_MODE
)

# Model Import
from Conv_VAE import Conv_VAE

# Other Imports
import os
import numpy as NP
import matplotlib.pyplot as PLT

# Generates training data for the MDN-RNN.
def generate_data():
    # Initializing the VAE model & loading weights.
    ConvVAE = Conv_VAE()
    try: ConvVAE.load_weights(VAE_WEIGHTS)
    except: raise Exception('Train the ConvVAE first.')
    
    # Gathering the list of files.
    files = os.listdir(VAE_TRAINING_DATA)
    
    # Ensuring enough training samples have been generated.
    if len(files) < N_EPISODES:
        raise Exception(f'Actual/Desired Training Samples: {len(files)}/{N_EPISODES}')
    
    for N, file in enumerate(files):
        # Logging the current episode.
        print(f'EPISODE {N}')
        
        # Encoding the data for an episode.
        encoded_data = generate_episode_data(file, ConvVAE)
        observations, actions, rewards, done = encoded_data
        
        # Saving the data.
        save_file = os.path.join(RNN_TRAINING_DATA, file)
        NP.savez_compressed(
            save_file,
            observations = observations,
            actions = actions,
            rewards = rewards,
            done = done
        )

        
# Generates training data for a single instance.
def generate_episode_data(file, ConvVAE):
    # Getting the full file path.
    file_path = os.path.join(VAE_TRAINING_DATA, file)
        
    # Loading the data and encoding the observations.
    data = NP.load(file_path)
    observations = data['observations'].reshape((N_TIMESTEPS,) + FRAME_DIM)
    encoded_observations = ConvVAE.Encoder.predict(observations)
    
    # Visualizing the difference between original and encoded data.
    if RENDER_MODE:
        observations = observations.astype(NP.float32)
        visualize_data(observations[N_TIMESTEPS // 2], ConvVAE.Trainer)
    
    # Returning the encoded data.
    return (
        encoded_observations, 
        data['actions'], 
        data['rewards'].astype(int), 
        data['done'].astype(int)
    )

# Visualizes the difference between the original & processed input frames.
def visualize_data(original, Model):
    # Getting the prediction by passing the image through the entire autoencoder.
    prediction = Model.predict(NP.array([original]))[0]
    
    # Creating a new PLT figure.
    PLT.clf(); PLT.cla(); PLT.close();
    figure, axes = PLT.subplots(1, 2)
    
    # Plotting the data.
    axes[0].imshow(original)
    axes[1].imshow(prediction)
    PLT.show()
    
if __name__ == '__main__':        
    generate_data()