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

EP_MAX = 8192
BATCH = 32

torch.autograd.set_detect_anomaly(True)

soulsgym.set_log_level(level=logging.DEBUG)
env = gymnasium.make("SoulsGymIudex-v0")

def train():
    torch.cuda.manual_seed(1993)

    # actor = PPO.PolicyNet().cuda()
    # critic = PPO.ValueNet(actor).cuda()
    
    # Model = PPO.PPO(actor, critic)

    actor = PPO.PolicyNet().cuda()
    actor.load_state_dict(torch.load("checkpoint/actor.pth"))
    # actor.eval()
    critic = PPO.ValueNet(actor).cuda()
    critic.load_state_dict(torch.load("checkpoint/critic.pth"))
    # critic.eval()

    Model = PPO.PPO(actor, critic)

    buffer_s, buffer_a, buffer_hc, buffer_la, buffer_img, buffer_old_v, buffer_old_dist, buffer_adv, buffer_target_value= [], [], [], [], [], [], [], [], []
    
    terminated = False
    all_ep_r = []
    # in one episode
    for ep in range(EP_MAX): 
        ep_r = 0
        ori_obs, info = env.reset()
        # for lstm
        hc = torch.randn([1, constant.HID_CELL_DIM]).cuda()


        terminated = False
        ep_buffer_s, ep_buffer_a, ep_buffer_r, ep_buffer_hc, ep_buffer_la, ep_buffer_img, ep_buffer_old_v, ep_buffer_old_dist = [], [], [], [], [], [], [], []
        
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

            ep_buffer_s.append(np.squeeze(obs, axis=0))
            ep_buffer_hc.append(np.squeeze(hc, axis=0))
            ep_buffer_a.append(action)
            ep_buffer_r.append(reward)
            ep_buffer_la.append(np.squeeze(legal_action, axis=0))
            ep_buffer_img.append(np.squeeze(img, axis=0))
            ep_buffer_old_v.append(np.squeeze(old_v, axis=0))
            ep_buffer_old_dist.append(np.squeeze(old_dist, axis = 0))

            ori_obs = next_obs
            ep_r += reward
        
        # calculate discounted reward after episode finished
        # print("ebs: %s, eba: %s, ebhc: %s, ebla: %s, ebimg: %s, ebov: %s, ebodist: %s", 
        #         len(ep_buffer_s), len(ep_buffer_a), len(ep_buffer_hc), len(ep_buffer_la), len(ep_buffer_img), len(ep_buffer_old_v), len(ep_buffer_old_dist))
        
        lastValue = Model.get_v(hc, obs, img, legal_action).cuda()
        
        bs = torch.vstack(ep_buffer_s).cuda()
        bhc = torch.vstack(ep_buffer_hc).cuda()
        bimg = torch.vstack(ep_buffer_img).cuda()
        bla = torch.vstack(ep_buffer_la).cuda()
        ba = torch.FloatTensor(ep_buffer_a).reshape([len(ep_buffer_a), 1]).cuda()
        
        v_s_ = Model.get_v(bhc, bs, bimg, bla).cuda() # The value of last state, criticed by value network
        adv, target_value = Model.compute_generalized_advantage_estimator(ba, v_s_, lastValue, len(v_s_))

        bov = torch.FloatTensor(ep_buffer_old_v).reshape([len(ep_buffer_old_v), 1])
        bodist = torch.vstack(ep_buffer_old_dist)
        badv = adv
        btr = target_value
        
        buffer_s += bs
        buffer_a += ba
        buffer_hc += bhc
        buffer_la += bla
        buffer_img += bimg
        buffer_old_v += bov
        buffer_old_dist += bodist
        buffer_adv += badv
        buffer_target_value += btr
        
        # replay buffer to model
        # only get latest replay
        if len(buffer_s) > constant.BATCH_SIZE:
            idx = len(buffer_s) - constant.BATCH_SIZE
            buffer_s = buffer_s[idx:]
            buffer_a = buffer_a[idx:]
            buffer_hc = buffer_hc[idx:]
            buffer_la = buffer_la[idx:]
            buffer_img = buffer_img[idx:] 
            buffer_old_dist = buffer_old_dist[idx:]
            buffer_old_v = buffer_old_v[idx:]
            buffer_adv = buffer_adv[idx:]
            buffer_target_value = buffer_target_value[idx:]
            
        u_bs = torch.vstack(buffer_s).cuda()
        u_ba = torch.FloatTensor(buffer_a).reshape([len(buffer_a), 1]).cuda()
        u_bhc = torch.vstack(buffer_hc).cuda()
        u_bimg = torch.vstack(buffer_img).cuda()
        u_bov = torch.FloatTensor(buffer_old_v).reshape([len(buffer_old_v), 1]).cuda()
        u_bodist = torch.vstack(buffer_old_dist).cuda()
        u_badv = torch.FloatTensor(buffer_adv).reshape([len(buffer_adv), 1]).cuda()
        u_btv = torch.FloatTensor(buffer_target_value).reshape([len(buffer_target_value), 1]).cuda()
        u_bla = torch.vstack(buffer_la).cuda()
        
        # print("ubs: %s, u_ba: %s, ubhc: %s, u_bla: %s, u_bimg: %s, ubov: %s, ubodist: %s, ubadv: %s, ubtv: %s", 
                # u_bs.shape, u_ba.shape, u_bhc.shape, u_bla.shape, u_bimg.shape, u_bov.shape, u_bodist.shape, u_badv.shape, u_btv.shape)
        
        actor_loss, critic_loss = Model.update(u_bs, u_ba, u_bhc, u_bla, u_bimg, u_bov, u_bodist, u_badv, u_btv)
        
        logger.info("Ep: %s, |Ep_r: %s| Value(state): %s, actor_loss: %s, critic_loss: %s" , ep, ep_r, torch.mean(u_bov), actor_loss, critic_loss)
        if ep == 0: 
            all_ep_r.append(ep_r)
        else: 
            all_ep_r.append(all_ep_r[-1]*0.9 + ep_r*0.1)
            
        if ep > 0 and ep % constant.SAVE_EPOCH == 0:
            torch.save(actor.state_dict(), "checkpoint/actor.pth")
            torch.save(critic.state_dict(), 'checkpoint/critic.pth')
            

    plt.plot(np.arange(len(all_ep_r)), all_ep_r)
    plt.xlabel('Episode')
    plt.ylabel('Moving averaged episode reward')
    plt.show()
        
    env.close()

if __name__ == "__main__":
        train()