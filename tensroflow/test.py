import gymnasium
import soulsgym
import logging
from model import PPO
import numpy as np
import utils.utils as utils
import matplotlib.pyplot as plt
import sys

logger = logging.getLogger(__name__)
logger.setLevel(level=logging.DEBUG)

# StreamHandler
stream_handler = logging.StreamHandler(sys.stdout)
stream_handler.setLevel(level=logging.DEBUG)
logger.addHandler(stream_handler)

# FileHandler
file_handler = logging.FileHandler('output.log')
file_handler.setLevel(level=logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

EP_MAX = 1
BATCH = 32
GAMMA = 0.9

def test():
    Model = PPO.PPO()
    all_ep_r = []
    # in one episode
    for ep in range(EP_MAX): 
        ep_r = 0
        # logging.info("New Episode")
        # ori_obs, info = env.reset()

        # for lstm
        hc = np.zeros(shape=[1, 52], dtype=float)

        buffer_s, buffer_a, buffer_r, buffer_hc = [], [], [], []
        obs = np.zeros(shape=[1,26], dtype=float)

        for i in range(10):
            # dict to vector
            obs = np.reshape(obs, [1, 26])
            hc = np.reshape(hc, [1, 52])

            action, hc = Model.choose_action(obs, hc)
            next_obs, reward = obs, 1

            buffer_s.append(np.squeeze(obs, axis=0))
            buffer_hc.append(np.squeeze(hc, axis=0))
            buffer_a.append(action)
            buffer_r.append(reward)

            obs = next_obs
            ep_r += reward
        
            # calculate discounted reward after episode finished
        v_s_ = Model.get_v(obs, hc) # The value of last state, criticed by value network
        discounted_r = []
        for r in buffer_r[::-1]:
            v_s_ = r + GAMMA * v_s_
            discounted_r.append(v_s_)
        discounted_r.reverse()
        bs, ba, br, bhc = np.vstack(buffer_s), np.vstack(buffer_a), np.vstack(discounted_r).reshape([10,1]), np.vstack(buffer_hc)
        buffer_s, buffer_a, buffer_r, buffer_hc = [], [], [], []
        Model.update(bs, ba, br, bhc)

        logger.info("Ep: %s, |Ep_r: %s" , ep, ep_r)
        if ep == 0: 
            all_ep_r.append(ep_r)
        else: 
            all_ep_r.append(all_ep_r[-1]*0.9 + ep_r*0.1)

    plt.plot(np.arange(len(all_ep_r)), all_ep_r)
    plt.xlabel('Episode')
    plt.ylabel('Moving averaged episode reward')
    plt.show()

if __name__ == "__main__":
    # logger.info("Test")
    test()