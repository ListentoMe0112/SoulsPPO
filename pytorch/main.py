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

EP_MAX = 6000
BATCH = 32
GAMMA = 0.9


soulsgym.set_log_level(level=logging.DEBUG)
env = gymnasium.make("SoulsGymIudex-v0")

def train():
    torch.cuda.manual_seed(1993)

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
        hc = torch.randn([1, constant.HID_CELL_DIM]).cuda()


        terminated = False
        buffer_s, buffer_a, buffer_r, buffer_hc, buffer_la, buffer_img, buffer_old_v, buffer_old_dist = [], [], [], [], [], [], [], []
        
        while not terminated:
            # dict to vector
            legal_action = torch.from_numpy(np.zeros([1,constant.ACT_NUM])).type(torch.float32).cuda()
            obs = utils.state_to_vector(ori_obs)
            # 90 160, 3
            img = env.game.img
            img = torch.reshape(torch.from_numpy(img), [1, 90 * 160 *3]).type(torch.float32).cuda()
            obs = torch.reshape(torch.from_numpy(obs), [1,26]).type(torch.float32).cuda()
            hc = torch.reshape(hc, [1,constant.HID_CELL_DIM]).type(torch.float32).cuda()
            legal_actions = torch.from_numpy(np.array(env.current_valid_actions())).cuda()

            for v in legal_actions:
                legal_action[0][v] = 1.0
            legal_action.type(torch.float32)

            action, hc, old_dist, old_v = Model.inference(hc, obs, img, legal_action)

            next_obs, reward, terminated, truncated, info = env.step(action)

            buffer_s.append(np.squeeze(obs, axis=0))
            buffer_hc.append(np.squeeze(hc, axis=0))
            buffer_a.append(action)
            buffer_r.append(reward)
            buffer_la.append(np.squeeze(legal_action, axis=0))
            buffer_img.append(np.squeeze(img, axis=0))
            buffer_old_v.append(np.squeeze(old_v, axis=0))
            buffer_old_dist.append(np.squeeze(old_dist, axis = 0))

            ori_obs = next_obs
            ep_r += reward
        
        # calculate discounted reward after episode finished
        v_s_ = Model.get_v(hc, obs, img, legal_action) # The value of last state, criticed by value network
        discounted_r = []
        for r in buffer_r[::-1]:
            v_s_ = r + GAMMA * v_s_
            discounted_r.append(v_s_)
        discounted_r.reverse()
        bs, ba, br, bhc, bla, bimg = torch.vstack(buffer_s), torch.FloatTensor(buffer_a).reshape([len(buffer_a), 1]), torch.FloatTensor(discounted_r).reshape([len(buffer_r), 1]), torch.vstack(buffer_hc), torch.vstack(buffer_la), torch.vstack(buffer_img)
        bov = torch.FloatTensor(buffer_old_v).reshape([len(buffer_old_v), 1])
        bodist = torch.vstack(buffer_old_dist)
        
        # replay buffer to model
        # only get latest replay
        if len(buffer_s) > 4096:
            idx = len(buffer_s) - 4096
            buffer_s = buffer_s[idx:-1]
            buffer_a = buffer_a[idx:-1]
            buffer_hc = buffer_hc[idx:-1]
            buffer_la = buffer_la[idx:-1]
            buffer_img = buffer_img[idx:-1] 
        
        Model.update(bs, ba, br, bhc, bla, bimg, bov, bodist)

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