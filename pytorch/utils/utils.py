import constant.constant as constant
import numpy as np
import logging

a = 1
b = True
c = np.arange(1)

# ObservationSpace:Dict(
#      1 'phase': 1
#      2 'boss_animation': Discrete(33, start=-1), 
#      3 'boss_animation_duration': Box(0.0, 10.0, (1,), float32), 
#      4 'boss_hp': Box(0.0, 1037.0, (1,), float32), 
#      5 'boss_max_hp': Discrete(1, start=1037), 
#      6,7,8,9 'boss_pose': Box([110.     540.     -73.      -3.1416], [190.     640.     -55.       3.1416], (4,), float32), 
#      10, 11, 12, 13, 14, 15 'camera_pose': Box([110. 540. -73.  -1.  -1.  -1.], [190. 640. -55.   1.   1.   1.], (6,), float32), 
#      16 'lock_on': Discrete(2), 'phase': Discrete(2, start=1), 
#      17 'player_animation': Discrete(51, start=-1), 
#      18 'player_animation_duration': Box(0.0, 10.0, (1,), float32), 
#      19 'player_hp': Box(0.0, 454.0, (1,), float32), 
#      20 'player_max_hp': Discrete(1, start=454), 
#      21 'player_max_sp': Discrete(1, start=95), 
#      22,23,24,25 'player_pose': Box([110.     540.     -73.      -3.1416], [190.     640.     -55.       3.1416], (4,), float32), 
#      26 'player_sp': Box(0.0, 95.0, (1,), float32)), 


# ActionSpace:
#       Discrete(20)
def state_to_vector(s:dict):
    obs = np.empty([constant.OBS_DIM,], dtype=float)

    obs[0] = float(s['phase'])
    obs[1] = float(s['boss_animation'])
    obs[2] = float(s['boss_animation_duration'])
    obs[3] = float(s['boss_hp'])
    obs[4] = float(s['boss_max_hp'])
    obs[5] = float(s['boss_pose'][0])
    obs[6] = float(s['boss_pose'][1])
    obs[7] = float(s['boss_pose'][2])
    obs[8] = float(s['boss_pose'][3])
    obs[9] = float(s['camera_pose'][0])
    obs[10] = float(s['camera_pose'][1])
    obs[11] = float(s['camera_pose'][2])
    obs[12] = float(s['camera_pose'][3])
    obs[13] = float(s['camera_pose'][4])
    obs[14] = float(s['camera_pose'][5])
    obs[15] = float(s['lock_on'])
    obs[16] = float(s['player_animation'])
    obs[17] = float(s['player_animation_duration'])
    obs[18] = float(s['player_hp'])
    obs[19] = float(s['player_max_hp'])
    obs[20] = float(s['player_max_sp'])
    obs[21] = float(s['player_pose'][0])
    obs[22] = float(s['player_pose'][1])
    obs[23] = float(s['player_pose'][2])
    obs[24] = float(s['player_pose'][3])
    obs[25] = float(s['player_sp'])
    return obs