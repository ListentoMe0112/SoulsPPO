import numpy as np
import config.constant as constant
import logging

# ObservationSpace:Dict(
#       'phase': 1
#      'boss_animation': Discrete(33, start=-1), 
#      'boss_animation_duration': Box(0.0, 10.0, (1,), float32), 
#      'boss_hp': Box(0.0, 1037.0, (1,), float32), 
#      'boss_max_hp': Discrete(1, start=1037), 
#      'boss_pose': Box([110.     540.     -73.      -3.1416], [190.     640.     -55.       3.1416], (4,), float32), 
#      'camera_pose': Box([110. 540. -73.  -1.  -1.  -1.], [190. 640. -55.   1.   1.   1.], (6,), float32), 
#      'lock_on': Discrete(2), 'phase': Discrete(2, start=1), 
#      'player_animation': Discrete(51, start=-1), 
#      'player_animation_duration': Box(0.0, 10.0, (1,), float32), 
#      'player_hp': Box(0.0, 454.0, (1,), float32), 
#      'player_max_hp': Discrete(1, start=454), 
#      'player_max_sp': Discrete(1, start=95), 
#      'player_pose': Box([110.     540.     -73.      -3.1416], [190.     640.     -55.       3.1416], (4,), float32), 
#      'player_sp': Box(0.0, 95.0, (1,), float32)), 


# ActionSpace:
#       Discrete(20)

a = 1
b = True
c = np.arange(1)

# convert state to vector
def state_to_vector(s:dict):
    idx = 0
    obs = np.empty([constant.S_DIM,], dtype=float)
    for _, v in s.items():
        if type(v) == type(a) or type(v) == type(b):
            obs[idx] = float(v)
            idx+=1
        elif type(v) == type(c):
            for i in range(v.shape[0]):
                obs[idx] = v[i]
                idx += 1
        else:
            logging.error("Type not detected: %s", type(v))
    return obs
    
def action_to_legal(a: int):
    if a >= 0 and a <= 19:
        return np.int64(a)
    else:
        logging.error("Predict Action Is Not Legal")
        return np.int64(0)

