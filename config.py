import os

# ------------------------------- CONSTANTS ------------------------------- #

# Model Weight Storage Locations
WEIGHTS_DIR = os.path.join(os.getcwd(), 'Weights')
VAE_WEIGHTS = os.path.join(WEIGHTS_DIR, 'cnn_vae.h5')
RNN_WEIGHTS = os.path.join(WEIGHTS_DIR, 'mdn_rnn.h5')
BEST_CONTROLLER_WEIGHTS = os.path.join(WEIGHTS_DIR, 'controller_BEST.json')
OPTIMAL_CONTROLLER_WEIGHTS = os.path.join(WEIGHTS_DIR, 'controller_OPTIMAL.json')

# Model Training Data Storage Locations
VAE_TRAINING_DATA = os.path.join(os.getcwd(), 'Conv_VAE_Data')
RNN_TRAINING_DATA = os.path.join(os.getcwd(), 'MDN_RNN_Data')

# Data Sizes
FRAME_DIM = (64, 64, 3)
LATENT_FRAME_DIM = 32
ACTION_DIM = 3

# General Constants
RENDER_MODE = False
FINAL_SEED = 4
MAX_INT = 2**31 - 1

# ConvVAE & MDNRNN Training Constants
N_EPISODES = 4000
N_TIMESTEPS = 350
REFRESH_RATE = 10

# Controller Training Constants
TRAIN_STEPS = 10
TRIALS_PER_WORKER = 3
MAX_STEPS = 1000
PRECISION = 6

# ------------------------ INITIALIZING DIRECTORIES ----------------------- #

for directory in [WEIGHTS_DIR, VAE_TRAINING_DATA, RNN_TRAINING_DATA]:
    if not os.path.exists(directory):
        os.mkdir(directory)
