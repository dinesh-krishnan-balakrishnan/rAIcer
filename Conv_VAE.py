# Constants
from config import FRAME_DIM, LATENT_FRAME_DIM

# Tensorflow Model Imports
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input,
    Conv2D, 
    Conv2DTranspose, 
    Flatten, 
    Reshape,
    Dense, 
    Lambda
)

# Math Imports
import tensorflow.keras.backend as Math
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

class Conv_VAE():
    
# ----------------------------- USER FUNCTIONS ----------------------------- #  

    def train(self, data):
        # EarlyStopping() prevents the model from overfitting.
        callback = [EarlyStopping(
            monitor = 'val_loss',
            min_delta = 1e-4,
            patience = 5,
            verbose = 1
        )]
        
        # Training by passing the same data as input & output.
        self.Trainer.fit(
            data,
            data,
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
        # Downsampling Convolutional Layer Constants
        self.C_FILTERS = [32, 64, 64, 128]
        self.C_KERNELS = [4, 4, 4, 4]
        self.C_STRIDES = [2, 2, 2, 2]
        self.C_ACTIVATIONS = ['relu', 'relu', 'relu', 'relu']
                
        # Upsampling Convolutional Layer Constants
        self.CT_FILTERS = [64, 64, 32, 3]
        self.CT_KERNELS = [5, 5, 6, 6]
        self.CT_STRIDES = [2, 2, 2, 2]
        self.CT_ACTIVATIONS = ['relu', 'relu', 'relu', 'sigmoid']
        
        # Output Conversion Constant
        self.DENSE_NODES = 1024
        
        # Training Constants
        self.LEARNING_RATE = 1e-4
        self.EPOCHS = 10
        self.BATCH_SIZE = 100
        self.VALIDATION_SPLIT = 0.15
 
    def _initialize_models(self):
        # ------------------------ MODEL FUNCTIONS  
        
        # Uses the mean & standard deviation to generate latent vector.
        def generate_latent_vector(vectors):
            mean = vectors[0]
            log_deviation = vectors[1]
            
            # Calculating the deviation from log-deviation.
            deviation = Math.exp(log_deviation / 2)
            
            # Generating a sampling distribution for the deviation.
            sample_shape = (Math.shape(deviation)[0], LATENT_FRAME_DIM)
            deviation_sampling = Math.random_normal(shape = sample_shape)
            
            return mean + deviation * deviation_sampling
        
        # L2 Loss
        def L2_loss(actual, prediction):            
            result = Math.sum(Math.square(actual - prediction), axis = 1)
            return Math.mean(result)
        
        # Kullback-Leibler Loss
        def KL_loss(actual, prediction):
            mean = Math.square(latent_mean)
            deviation = latent_deviation - Math.exp(latent_deviation)
            
            return -Math.sum((1 + deviation - mean), axis = -1)
        
        # Total Loss
        def loss(actual, prediction):
            return L2_loss(actual, prediction) + KL_loss(actual, prediction)
            
        # ------------------------- MODEL
        
        # Taking a frame of the Car Racing environment as input.
        _input = Input(shape = FRAME_DIM)
        
        # Downsampling
        conv_1 = Conv2D(self.C_FILTERS[0], kernel_size = self.C_KERNELS[0], 
                    strides = self.C_STRIDES[0], activation = self.C_ACTIVATIONS[0]
                 )(_input)
        conv_2 = Conv2D(self.C_FILTERS[1], kernel_size = self.C_KERNELS[1], 
                    strides = self.C_STRIDES[1], activation = self.C_ACTIVATIONS[1]
                 )(conv_1)
        conv_3 = Conv2D(self.C_FILTERS[2], kernel_size = self.C_KERNELS[2], 
                    strides = self.C_STRIDES[2], activation = self.C_ACTIVATIONS[2]
                 )(conv_2)
        conv_4 = Conv2D(self.C_FILTERS[3], kernel_size = self.C_KERNELS[3], 
                    strides = self.C_STRIDES[3], activation = self.C_ACTIVATIONS[3]
                 )(conv_3)
        
        # Altering data dimensions.
        latent_input = Flatten()(conv_4)
        
        # Calculating the latent vector.
        latent_mean =      Dense(LATENT_FRAME_DIM)(latent_input)
        latent_deviation = Dense(LATENT_FRAME_DIM)(latent_input)
        latent_frame =     Lambda(generate_latent_vector)([latent_mean, latent_deviation])
        
        # Increasing data parameters.
        increase_parameters = Dense(self.DENSE_NODES)(latent_frame)
        reshape_parameters =  Reshape((1, 1, self.DENSE_NODES))(increase_parameters)
        
        # Upsampling
        convT_1 = Conv2DTranspose(self.CT_FILTERS[0], kernel_size = self.CT_KERNELS[0],
                      strides = self.CT_STRIDES[0], activation = self.CT_ACTIVATIONS[0]
                  )(reshape_parameters)
        convT_2 = Conv2DTranspose(self.CT_FILTERS[1], kernel_size = self.CT_KERNELS[1],
                      strides = self.CT_STRIDES[1], activation = self.CT_ACTIVATIONS[1]
                  )(convT_1)
        convT_3 = Conv2DTranspose(self.CT_FILTERS[2], kernel_size = self.CT_KERNELS[2],
                      strides = self.CT_STRIDES[2], activation = self.CT_ACTIVATIONS[2]
                  )(convT_2)
        _output = Conv2DTranspose(self.CT_FILTERS[3], kernel_size = self.CT_KERNELS[3],
                      strides = self.CT_STRIDES[3], activation = self.CT_ACTIVATIONS[3]
                  )(convT_3)
        
        # ----------------------- COMPILING MODELS
        
        # The 'Encoder' is used to compress an input frame.
        self.Encoder = Model(_input, latent_frame)
        
        # The 'Trainer' is used to determine optimal model weights.
        self.Trainer = Model(_input, _output)
                
        # Compiling 'Trainer' for training purposes.
        optimizer = Adam(learning_rate = 1e-4)
        self.Trainer.compile(
            optimizer = optimizer, 
            loss = loss, 
            metrics = [L2_loss, KL_loss]
        )        
    
if __name__ == '__main__':
    ConvVAE = Conv_VAE()
    print(ConvVAE.Trainer.summary())