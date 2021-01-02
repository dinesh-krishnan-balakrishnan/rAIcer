# Constants
from config import (
    VAE_WEIGHTS, 
    RNN_WEIGHTS, 
    LATENT_FRAME_DIM, 
    ACTION_DIM
)

# Model Imports
from Conv_VAE import Conv_VAE
from MDN_RNN import MDN_RNN

# Other Imports
import json
import numpy as NP

class Controller():
    
# ----------------------------- INITIALIZATION ----------------------------- #    
    
    def __init__(self):
        # Initializing the trained VAE & RNN
        self.ConvVAE = Conv_VAE(); self.ConvVAE.load_weights(VAE_WEIGHTS);
        self.MDNRNN = MDN_RNN();   self.MDNRNN.load_weights(RNN_WEIGHTS);
        
        # Retrieving the models necessary for use by the Controller.
        self.Encoder = self.ConvVAE.Encoder.predict
        self.RNN = self.MDNRNN.RNN.predict
        self.reset_RNN_state()
        
        # Controller Parameters
        self.CONTROLLER_DIM = ((LATENT_FRAME_DIM + self.MDNRNN.LSTM_NODES), ACTION_DIM)
        self.N_PARAMS = NP.prod(self.CONTROLLER_DIM) + ACTION_DIM
        self.W = NP.zeros(self.CONTROLLER_DIM)
        self.B = NP.zeros(self.CONTROLLER_DIM[1])
        
# ------------------------ SETTING MODEL PARAMETERS ------------------------ #    
        
    # Loads the Controller's weights & bias from a file.
    def load_parameters(self, filename):
        with open(filename, 'r') as file:
            data = json.load(file)
            self.set_parameters(data)
        
    # Defines the Controller's weights & bias.
    def set_parameters(self, parameters):
        parameters = NP.array(parameters)
        self.B, W_unshaped = NP.split(parameters, [ACTION_DIM])
        self.W = W_unshaped.reshape(self.CONTROLLER_DIM)
        
    # Resetting the RNN state.
    def reset_RNN_state(self):
        # RNN State Storage
        self.hidden_state_CURRENT = NP.zeros((self.MDNRNN.LSTM_NODES,))
        self.cell_state_CURRENT = NP.zeros((self.MDNRNN.LSTM_NODES,))
        self.action_CURRENT = NP.zeros((ACTION_DIM,))
        
# --------------------------- CONTROLLER ACTION ---------------------------- #    
        
    # Generates the next action for the agent to take in the simulation.
    def get_action(self, observation):
        # Encoding the observation using the VAE.
        observation = observation.astype('float32') / 255.
        encoded_observation = self.Encoder(NP.array([observation]))[0]
        
        # Making a prediction of the future using the RNN.
        state_NEXT = self.RNN([
            NP.array([[NP.concatenate((encoded_observation, self.action_CURRENT))]]), 
            NP.array([self.hidden_state_CURRENT]),
            NP.array([self.cell_state_CURRENT])
        ])
        
        # Modifying the current state to match the future state.
        self.hidden_state_CURRENT = state_NEXT[0][0]
        self.cell_state_CURRENT = state_NEXT[1][0]
        
        # Determining an optimal action by passing the generated parameters 
        # into the Controller's dense network.
        _input = NP.concatenate((encoded_observation, self.hidden_state_CURRENT))
        dense = NP.matmul(_input, self.W) + self.B
        activation = NP.tanh(dense)
        activation[1] = (activation[1] + 1) / 2
        activation[2] = (activation[2] + 1) / 2
        
        return activation
        
        
