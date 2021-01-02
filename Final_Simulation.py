from config import (
    BEST_CONTROLLER_WEIGHTS,
    OPTIMAL_CONTROLLER_WEIGHTS,
    MAX_STEPS
)

from Controller import Controller
from car_racing import CarRacing
from gym.wrappers.monitor import Monitor

C = Controller()

for weights in [BEST_CONTROLLER_WEIGHTS, OPTIMAL_CONTROLLER_WEIGHTS]:
    ENV = Monitor(CarRacing(), f'{weights[:-5]}_SIM', force = True)

    try: C.load_parameters(weights)
    except: raise Exception('Train the Controller first.')

    done = False
    steps = 0

    observation = ENV.reset()
    reward_FULL = 0
    
    while not done and steps < MAX_STEPS:
        ENV.render()
        
        action = C.get_action(observation)
        observation, reward, done, _ = ENV.step(action)
        
        reward_FULL += reward
        steps += 1
        
    ENV.close()
    print(f'{weights} Reward: {reward_FULL}')