# r(AI)cer

**rAIcer** was created based on of Jürgen Schmidhuber's *Full World Model*. It also has such a name because I just wanted to name my AI (regardless of how bad the name is). 
The link to the model idea can be found here: https://worldmodels.github.io/.

----

## In Action

![Video of DeepRacer in action.](./Weights/controller_BEST_SIM/main.gif)

> *The above video demonstrates the final controller AI in action.*

It's very interesting to analyze AI traits that have surfaced during the model training process. The AI chooses to always steer toward a direction, using the resulting drag to smoothly navigate corners. Additionally, the AI has also learned how to correct its positioning when it leaves the track. Personally, I find it quite satisfying to watch.

----

## Models

<br>

### Variational Autoencoder

**Relevant Files:**
* car_racing.py
* Conv_VAE.py
* Conv_VAE_Generation.py
* Conv_VAE_Train.py

**Required Packages:**
* gym 
* Box2D
* tensorflow (or tensorflow-gpu)
* numpy

*car_racing.py* is the environment simulation file provided by OpenAI. *Conv_VAE_Generation.py* uses the simulation to gather randomly collected data, which involves the random action, observation, and completion state for each step. *Conv_VAE_Train.py* then uses the generated data to train the variational autoencoder model coded in *Conv_VAE.py*, reducing the image input to a set of 32 parameters.

----

### Mixture Density Network

**Relevant Files:**
* MDN_RNN.py
* MDN_RNN_Generation.py
* MDN_RNN_Train.py

**Required Packages:**
* tensorflow (or tensorflow-gpu)
* numpy
* matplotlib

*MDN_RNN.py* contains the mixture density model, which predicts how the next frame will look based on the current frame and action. *MDN_RNN_Generation.py* uses the trained variational autoencoder to encode the currently gathered data, which will serve as input to the mixture density model. *MDN_RNN_Train.py* then uses the encoded inputs to train the model. More specifically, for each timestep, each action and encoded frame are passed in as input to the model; the next frame is then passed in as the desired output.

----

### Controller

**Relevant Files:**
* Controller.<span></span>py
* Controller_Train.py
* CMA_ES.py

**Required Packages:**
* gym 
* Box2D
* cma
* tensorflow (or tensorflow-gpu)
* numpy

The mixture-density network consists of a hidden LSTM layer. The purpose of training the network was to train the LSTM's weights rather than the full model itself. The controller takes the encoded input from the variational autoencoder, the LSTM's hidden states, and the LSTM's cell states to predict the best possible action. Training involves using a genetic reinforcement learning algorithm, called *Coviance Matrix Adaptation*. The algorithm generates weights that are attached to a manually implemented DNN, then learns how effective the weights were through simulation.

----

### Other Files

**config<span>.</span>py:** Contains training and simulation constants. These values are the same throughout the training process, and can be modified as desired. Constants related to model architecture are found within the model files themselves. The file can also be executed to intialize the data and weight directories.

**Final_Simulation.py:** Once the three models have been trained, **Final_Simulation.py** can be run to test the final AI's performance. Additionally, videos of the simulations will be saved in *Weights* directory.

----

## Creating a Model from Scratch

*Conv_VAE_Data* and *MDN_RNN_Data* already contain samples generated from the simulation environment. Additionally, the *Weights* directory contains the trained model weights. To train the model from scratch, delete these folders and run *config.py* to reinitialize them.

----

## Packages

All required packages can be installed from *pip*. The **Required Packages** sections list the exact package that needs to be downloaded.  