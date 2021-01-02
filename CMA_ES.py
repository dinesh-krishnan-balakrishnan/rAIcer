from config import PRECISION

import cma
import numpy as NP

# Wrapper for the Covariance Matrix Adaptation evolution strategy.
class CMA_ES():

    def __init__(self, N_parameters, N_workers):
        # Learning Constants
        self.INIT_DEVIATION = 1e-1
        self.WEIGHT_DECAY = 1e-2
        
        # Initializing the evolution strategy.
        self.ES = cma.CMAEvolutionStrategy(
            N_parameters * [0],
            self.INIT_DEVIATION,
            {'popsize': N_workers}
        )
        
    # Provides a set of parameters for training.
    def ask(self):
        self.solutions = NP.array(self.ES.ask())
        self.solutions = NP.round_(self.solutions, decimals = PRECISION)
        return self.solutions
    
    # Updates the current ES model with training results.
    def tell(self, results):
        results = -NP.array(results)
        
        if self.WEIGHT_DECAY > 0:
            L2_decay = NP.mean(NP.square(self.solutions), axis = 1)
            L2_decay *= self.WEIGHT_DECAY
            results += L2_decay
            
        self.ES.tell(self.solutions, results.tolist())
        
    # Best Evaluated Parameters
    def best_parameters(self):
        parameters = NP.round_(self.ES.result[0], decimals = PRECISION)
        return parameters.tolist()
    
    # Predicted Optimal Parameters
    def optimal_parameters(self):
        parameters = NP.round_(self.ES.result[5], decimals = PRECISION)
        return parameters.tolist()