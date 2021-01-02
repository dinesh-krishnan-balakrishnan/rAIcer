# Constants
from config import LATENT_FRAME_DIM, ACTION_DIM

# Tensorflow Model Imports
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense

# Math Imports
import tensorflow as TF
import tensorflow.keras.backend as Math
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from math import pi as PI, sqrt 

class MDN_RNN():
    
# ----------------------------- USER FUNCTIONS ----------------------------- #  

    def train(self, _input, _output):
        # EarlyStopping() prevents the model from overfitting.
        callback = [EarlyStopping(
            monitor = 'val_loss',
            min_delta = 1e-4,
            patience = 5,
            verbose = 1
        )]
        
        # Training by passing the same data as input & output.
        self.Trainer.fit(
            _input,
            _output,
            epochs = self.EPOCHS,
            batch_size = self.BATCH_SIZE,
            validation_split = self.VALIDATION_SPLIT,
            callbacks = callback
        )
        
    # Loading weights from file.
    def load_weights(self, file_name):
        self.Trainer.load_weights(file_name)
        
    # Saving weights to file.
    def save_weights(self, file_name):
        self.Trainer.save_weights(file_name)
        
# ----------------------------- INITIALIZATION ----------------------------- #    
    
    def __init__(self):
        self._initialize_constants()
        self._initialize_models()
        
    def _initialize_constants(self):
        # Model Constants
        self.LSTM_NODES = 256
        self.N_DISTRIBUTIONS = 5
        
        # Training Constants
        self.LEARNING_RATE = 1e-4
        self.EPOCHS = 200
        self.BATCH_SIZE = 100 
        self.VALIDATION_SPLIT = 0.15
        
        
    def _initialize_models(self):
        # ------------------------ MODEL FUNCTIONS 
        
        # Extracting gaussian mixture coefficients from Dense layer output.
        def generate_gaussian_mixtures(prediction, RESHAPE_DIM):
            # Gathering distribution coefficients from the MDN Dense layer.
            weights, mean, log_deviation = TF.split(prediction, 3, axis = 1)
            
            # Reshaping coefficients.
            weights = Math.reshape(weights, RESHAPE_DIM)
            mean = Math.reshape(mean, RESHAPE_DIM)
            log_deviation = Math.reshape(log_deviation, RESHAPE_DIM)
            
            # Reformatting weights & deviation.
            weights = Math.exp(weights)
            weights = weights / Math.sum(weights, axis = 2, keepdims = True)
            deviation = Math.exp(log_deviation) + 1e-8
            
            return (weights, mean, deviation)
        
        # Determing log-norm deviation based on predicted gaussian mixtures.
        def log_normalization(actual, mixtures, RESHAPE_DIM):
            weights, mean, deviation = mixtures
            
            # Reshaping the expected output.
            actual = Math.tile(actual, (1, 1, self.N_DISTRIBUTIONS))
            actual = Math.reshape(actual, RESHAPE_DIM)

            # Calculating the log-normal proabibility density function.
            result = weights / (sqrt(2 * PI) * deviation)
            result *= Math.exp(-0.5 * Math.square((actual - mean) / deviation))
            return Math.sum(result, axis = 2)
        
        # Total Loss
        def loss(actual, prediction):
            # Reshape Constant
            RESHAPE_DIM = (-1, Math.shape(prediction)[1], self.N_DISTRIBUTIONS, LATENT_FRAME_DIM)
            
            # Calculating losses from log-norm deviation.
            mixtures = generate_gaussian_mixtures(prediction, RESHAPE_DIM)
            losses = log_normalization(actual, mixtures, RESHAPE_DIM)
            
            # Calculating total loss.
            result = Math.sum(-Math.log(losses + 1e-8), axis = 1)
            return Math.mean(result)
        
        # ------------------------ GENERAL LAYERS
        
        # Encoded frame & action input.
        _input = Input(shape = (None, (LATENT_FRAME_DIM + ACTION_DIM)))
        
        # Creating an LSTM layer that returns model state.
        LSTM_layer = LSTM(self.LSTM_NODES, 
            return_sequences = True, return_state = True)
        
        # ------------------------ TRAINING MODEL 
        
        # Predicting Gaussian mixtures.
        probabilities, _ , _ = LSTM_layer(_input)
        
        # Passing the result into a Mixture Density Network.
        MDN = Dense( 
            ACTION_DIM * LATENT_FRAME_DIM * self.N_DISTRIBUTIONS
        )(probabilities)
        
        # ----------------------- PREDICTION MODEL
                
        # Taking previous state as input.
        hidden_input = Input(shape = (self.LSTM_NODES,))
        cell_input =   Input(shape = (self.LSTM_NODES,))
        
        # Predicting next state.
        _, hidden_state, cell_state = LSTM_layer(
            _input, 
            initial_state = [hidden_input, cell_input]
        )
        
        # ----------------------- COMPILING MODELS
        
        # The 'Trainer' is used to determine optimal model weights.
        self.Trainer = Model(_input, MDN)
        
        # The 'RNN' is used to predict the next input state from a previous state.
        self.RNN = Model(
            [_input, hidden_input, cell_input], 
            [hidden_state, cell_state]
        )
        
        # Compilting 'Trainer' for training purposes.
        optimizer = Adam(learning_rate = 1e-3)
        self.Trainer.compile(
            optimizer = optimizer, 
            loss = loss, 
            metrics = [loss]
        )
        
if __name__ == '__main__':
    MDNRNN = MDN_RNN()
    print(MDNRNN.Trainer.summary())
    print(MDNRNN.RNN.summary())