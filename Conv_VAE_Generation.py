# Constants
from config import (
    VAE_TRAINING_DATA, 
    N_EPISODES, 
    N_TIMESTEPS, 
    REFRESH_RATE, 
    RENDER_MODE
)

# Environment Import
from car_racing import CarRacing

# Other Imports
import os
import random
import numpy as NP

# Generates training data for the Conv-VAE.
def generate_data():
    environment = CarRacing(verbose = RENDER_MODE)
    
    for episode in range(N_EPISODES):  
        # Logging the current episode.
        print(f'EPISODE {episode}')
        
        # Creating a new environment instance.
        environment.reset()
        
        # Gathering data from the instance.
        observations, actions, rewards, done = generate_episode_data(environment)
        
        # Closing the environment instance.
        environment.close()
        
        # Saving the gathered data.
        save_file = os.path.join(VAE_TRAINING_DATA, f'{episode}.npz')
        NP.savez_compressed(
            save_file,
            observations = observations,
            actions = actions,
            rewards = rewards,
            done = done
        )
    
# Generates training data for a single instance.
def generate_episode_data(environment):
    # Data Containers
    observations, actions, rewards, done_sequence = [], [], [], []
        
    for step in range(N_TIMESTEPS):
        # Renders the environment.
        if RENDER_MODE:
            environment.render()
            
        # Generating an action.
        if step % REFRESH_RATE == 0:
            action = get_action(step)
            
        # Making the action within the environment.
        observation, reward, done, _ = environment.step(action)
            
        # Storing the resulting data.
        observations.append(observation.astype('float32') / 255.)
        actions.append(action)
        rewards.append(reward)
        done_sequence.append(done)
        
    return (observations, actions, rewards, done_sequence)
        
# Generates an action based on the current timestep & randomness.
def get_action(step):
    if step <= 20:
        return [0, 1, 0]
    
    random_int = random.randint(0, 9)
    
    # Acceleration
    if random_int in [0, 1, 2, 3]:
        return [0, random.random(), 0]
    
    # Turn Left
    elif random_int in [4, 5]:
        return [-random.random(), 0, 0]
    
    # Turn Right
    elif random_int in [6, 7]:
        return [random.random(), 0, 0]
    
    # Break
    elif random_int in [8]:
        return [0, 0, random.random()]
    
    # Do Nothing
    return [0, 0, 0]

if __name__ == '__main__':
    generate_data()