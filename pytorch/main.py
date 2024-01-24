import gymnasium
import soulsgym
import logging
from model import PPO as PPO
import numpy as np
import utils.utils as utils
import matplotlib.pyplot as plt
import sys
import torch
from  constant import constant
logging.basicConfig(
    format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
    datefmt='%a, %d %b %Y %H:%M:%S',
    level=logging.DEBUG,
)

logger = logging.getLogger(__name__)
stream_handler = logging.StreamHandler(sys.stdout)
file_handler = logging.FileHandler('output.log')
logger.addHandler(stream_handler)
logger.addHandler(file_handler)

EP_MAX = 3000
BATCH = 32
GAMMA = 0.9


soulsgym.set_log_level(level=logging.DEBUG)
env = gymnasium.make("SoulsGymIudex-v0")

def train():
    if torch.cuda.is_available():
        torch.cuda.manual_seed(1993)
    else:
        torch.manual_seed(123)

    actor = PPO.PolicyNet().cuda()
    critic = PPO.ValueNet().cuda()
    
    Model = PPO.PPO(actor, critic)

    terminated = False
    all_ep_r = []
    # in one episode
    for ep in range(EP_MAX): 
        ep_r = 0
        ori_obs, info = env.reset()

        # for lstm
        hc = np.zeros(shape=[1, constant.HID_CELL_DIM], dtype=float)
        legal_action = np.zeros([1,constant.ACT_NUM])

        terminated = False
        buffer_s, buffer_a, buffer_r, buffer_hc, buffer_la, buffer_img = [], [], [], [], [], []

        while not terminated:
            # dict to vector
            obs = utils.state_to_vector(ori_obs)
            # 90 160, 3
            img = env.game.img
            img = np.reshape(img, [1, 90, 160, 3])
            obs = np.reshape(obs, [1,26])
            hc = np.reshape(hc, [1,constant.HID_CELL_DIM])
            legal_actions = env.current_valid_actions()
            for v in legal_actions:
                legal_action[0][v] = 1

            action, hc  = Model.get_action(hc, obs, img, legal_action, axis = 1)

            next_obs, reward, terminated, truncated, info = env.step(action)

            if ori_obs['boss_hp'][0] > next_obs['boss_hp'][0]:
                logging.info("Hit Boss")

            buffer_s.append(np.squeeze(obs, axis=0))
            buffer_hc.append(np.squeeze(hc, axis=0))
            buffer_a.append(action)
            buffer_r.append(reward)
            buffer_la.append(np.squeeze(legal_action, axis=0))
            buffer_img.append(np.squeeze(img, axis=0))

            ori_obs = next_obs
            ep_r += reward
        
        # calculate discounted reward after episode finished
        v_s_ = Model.critic(np.concatenate([hc, obs, img, legal_action], axis = 1)) # The value of last state, criticed by value network
        discounted_r = []
        for r in buffer_r[::-1]:
            v_s_ = r + GAMMA * v_s_
            discounted_r.append(v_s_)
        discounted_r.reverse()
        bs, ba, br, bhc, bla, bimg = np.vstack(buffer_s), np.vstack(buffer_a), np.array(discounted_r)[:, np.newaxis], np.vstack(buffer_hc), np.vstack(buffer_la), np.vstack(buffer_img)
        buffer_s, buffer_a, buffer_r, buffer_hc, buffer_la, buffer_img = [], [], [], [], [], []
        
        Model.update(bs, ba, br, bhc, bla, bimg, ep)

        logger.info("Ep: %s, |Ep_r: %s" , ep, ep_r)
        if ep == 0: 
            all_ep_r.append(ep_r)
        else: 
            all_ep_r.append(all_ep_r[-1]*0.9 + ep_r*0.1)

    plt.plot(np.arange(len(all_ep_r)), all_ep_r)
    plt.xlabel('Episode')
    plt.ylabel('Moving averaged episode reward')
    plt.show()
        
    env.close()


if __name__ == "__main__":
        train()