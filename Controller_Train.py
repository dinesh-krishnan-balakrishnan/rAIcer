# -------------------------------- IMPORTS -------------------------------- #    

# Constants
from config import (
    BEST_CONTROLLER_WEIGHTS,
    OPTIMAL_CONTROLLER_WEIGHTS,
    TRAIN_STEPS,
    TRIALS_PER_WORKER, 
    MAX_STEPS,
    RENDER_MODE,
    MAX_INT,
    PRECISION
)

# Environment, Evolution Strategy, and Controller Imports
from car_racing import CarRacing
from CMA_ES import CMA_ES
from Controller import Controller

# Environment Management Imports
import os
import sys
import json

# Seed Generation Imports
import numpy as NP
from random import randint as Random

# Multiprocessing Import
from multiprocessing import Process, Pipe, cpu_count 
import gc

# ----------------------------- LEADER PROCESS ----------------------------- #    

def leader():
    # Starting Worker Processes
    COMMS, N_WORKERS = start_workers()
    POPULATION = N_WORKERS * TRIALS_PER_WORKER

    # Initializing the evolution strategy.
    ES = CMA_ES(Controller().N_PARAMS, POPULATION)
    best_reward = None
    
    while True:
        # -------------------- TRAINING
        for trial in range(TRAIN_STEPS):
            print(f'TRAINING ITERATION {trial}:')
            # Generating Seeds & Parameters
            seeds = [Random(0, MAX_INT) for _ in range(POPULATION)]
            parameters = ES.ask()
            
            # Running Trials & Updating the ES
            results = order_workers(COMMS, parameters, seeds, True)
            ES.tell(results)
       
        # -------------------- TESTING
        print('TESTING:')
        # Generating Seeds & Parameters
        seeds = [Random(0, MAX_INT) for _ in range(N_WORKERS)]
        parameters = [ES.best_parameters()] * N_WORKERS
        
        # Calculating Current Reward
        results = order_workers(COMMS, parameters, seeds, False)
        reward = sum(results) / len(results)
        
        # Determining Best Reward
        if best_reward == None or reward > best_reward:
            best_reward = reward
            save_weights(ES)
            
            
        print(f'BEST REWARD: {best_reward}')
        

def start_workers():
    # Storing worker process metadata for termination.
    global WORKERS
    WORKERS = []
    
    # Storing communication pipelines and determining # of workers.
    COMMS = []
    N_WORKERS = cpu_count() - 2

    for Rank in range(N_WORKERS):
        # Creating communications for a single process.
        leader_COMM, worker_COMM = Pipe()
        COMMS.append(leader_COMM)
        
        # Starting the process.
        worker_process = Process(target = worker, args = (worker_COMM, Rank))
        worker_process.start()
        WORKERS.append(worker_process)
        
    return COMMS, N_WORKERS

def order_workers(COMMS, parameters, seed, TEST_MODE):
    # Sending testing parameters to the workers.
    for N, COMM in enumerate(COMMS):
        if TEST_MODE:
            START = N * TRIALS_PER_WORKER
            END = (N + 1) * TRIALS_PER_WORKER
            COMM.send((parameters[START:END], seed[START:END]))
        else:
            COMM.send(([parameters[N]], [seed[N]]))
        
    # Retrieving worker parameters.
    results = []
    for COMM in COMMS:
        results += COMM.recv()
        
    return results

# Saves the best evaluated model weights.
def save_weights(ES):
    with open(BEST_CONTROLLER_WEIGHTS, 'w') as file:
        json.dump(ES.best_parameters(), file)
        
    with open(OPTIMAL_CONTROLLER_WEIGHTS, 'w') as file:
        json.dump(ES.optimal_parameters(), file)

# ----------------------------- WORKER PROCESS ----------------------------- #    

def worker(COMM, Rank):
    # Initializing a controller and environment.
    C = Controller()
    ENV = CarRacing(verbose = False)
    
    while True:
        # Retrieving Testing Metadata
        parameters, seed = COMM.recv()
        rewards = []
        
        # Running simulations for each set of parameters.
        for N in range(len(parameters)):
            reward = simulate(C, ENV, parameters[N], seed[N])
            
            # Storing and logging the reward.
            rewards.append(reward)
            print(f'WORKER {Rank}: {reward} (TRIAL {N})')
            sys.stdout.flush()
            
        # Returning the full result metadata.
        COMM.send(rewards)
            
def simulate(C, ENV, parameters, seed):
    # Setting parameters and environment seed.
    C.set_parameters(parameters)
    NP.random.seed(seed)
    ENV.seed(seed)
    
    # Resetting environment and Controller state.
    C.reset_RNN_state()
    observation = ENV.reset()
    reward_FULL = 0
    
    for step in range(MAX_STEPS):
        # Retrieving the Controller's action.
        action = C.get_action(observation)
        
        # Taking a step within the environment.
        observation, reward, done, _ = ENV.step(action)
        
        # Storing the reward data.
        reward_FULL += reward
        
        if RENDER_MODE: ENV.render()
        if done: break
        
    # Closing the environment & returning the resulting reward.
    ENV.close()
    gc.collect()
    return round(reward_FULL, PRECISION)
    
# -------------------------- LEADER INITIALIZATION ------------------------- #    
    
if __name__ == '__main__':  
    # Preventing the GPU from being used.
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    
    # Executing the leader.
    try: leader()
        
    # Catching the SIGKILL error & terminating processes gracefully.
    except KeyboardInterrupt:
        for worker_process in WORKERS:
            worker_process.terminate()
        
        print('---------------')
        print('ENDING TRAINING')