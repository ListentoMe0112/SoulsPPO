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

def test():
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

        terminated = False
        buffer_s, buffer_a, buffer_r, buffer_hc, buffer_la, buffer_img = [], [], [], [], [], []

        for i in range(10):
            # dict to vector
            obs = torch.randn([1, constant.OBS_DIM]).cuda()
            img = torch.randn([1, 90 * 160 * 3]).cuda()
            hc = torch.randn([1, constant.HID_CELL_DIM]).cuda()
            legal_action = torch.zeros([1,constant.ACT_NUM]).cuda()
            legal_action[0][1] = 1

            action, hc  = Model.get_action(hc, obs, img, legal_action)

            reward = 0

            buffer_s.append(np.squeeze(obs, axis=0))
            buffer_hc.append(np.squeeze(hc, axis=0))
            buffer_a.append(action)
            buffer_r.append(reward)
            buffer_la.append(np.squeeze(legal_action, axis=0))
            buffer_img.append(np.squeeze(img, axis=0))

            ep_r += reward
        
        # calculate discounted reward after episode finished
        v_s_ = Model.get_v(hc, obs, img, legal_action) # The value of last state, criticed by value network
        discounted_r = []
        for r in buffer_r[::-1]:
            v_s_ = r + GAMMA * v_s_
            discounted_r.append(v_s_)
        discounted_r.reverse()
        bs, ba, br, bhc, bla, bimg = torch.vstack(buffer_s), torch.FloatTensor(buffer_a).reshape([len(buffer_a), 1]), torch.FloatTensor(discounted_r).reshape([len(buffer_r), 1]), torch.vstack(buffer_hc), torch.vstack(buffer_la), torch.vstack(buffer_img)
        buffer_s, buffer_a, buffer_r, buffer_hc, buffer_la, buffer_img = [], [], [], [], [], []
        
        Model.update(bs, ba, br, bhc, bla, bimg)

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
        test()