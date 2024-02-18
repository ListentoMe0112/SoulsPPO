from ple.games.flappybird import FlappyBird
from ple import PLE
import torch
import numpy as np
import time
from pygame.constants import K_w

import matplotlib.pyplot as plt
import logging
import sys
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


actor_learning_rate = 0.0002
value_learing_rate =  0.0001

actions = [None, K_w]
EP_MAX = 32768 * 2 * 2
HID_CELL_DIM = 32
ACT_NUM = 2
OBS_DIM = 8
IMG_DIM = 288 * 512 *3
BATCH_SIZE=128
UPDATE_NUM=1
SAVE_EPOCH=128


game = FlappyBird()
p = PLE(game, fps=30, display_screen=True, force_fps=True)
# agent = myAgentHere(allowed_actions=p.getActionSet())

def state_to_vector(ori_obs) -> np.ndarray:
    obs = np.empty([OBS_DIM,], dtype=float)
    obs[0] = ori_obs['player_y']
    obs[1] = ori_obs['player_vel']
    obs[2] = ori_obs["next_pipe_dist_to_player"]
    obs[3] = ori_obs["next_pipe_top_y"]
    obs[4] = ori_obs["next_pipe_bottom_y"]
    obs[5] = ori_obs["next_next_pipe_dist_to_player"]
    obs[6] = ori_obs["next_next_pipe_top_y"]
    obs[7] = ori_obs["next_next_pipe_bottom_y"]
    
    return obs

# output action [pi(s)]
class PolicyNet(torch.nn.Module):
    def __init__(self) -> None:
        super(PolicyNet, self).__init__()
        self.cnn_net = torch.nn.Sequential(
            # [None, 3, 288, 512] -> [None, 32, 71, 127]
            torch.nn.Conv2d(3, 32, kernel_size=8, stride=4), 
            # [None, 32, 71, 127] -> [None, 32, 35, 63]
            torch.nn.MaxPool2d(2, 2),
            torch.nn.ReLU(),
            
            #  [None, 32, 35, 63] -> [None, 64, 16, 30]
            torch.nn.Conv2d(32, 64, kernel_size=4, stride=2), 
            # [None, 64, 16, 30]-> [64, 8, 15]
            torch.nn.MaxPool2d(2, 2),
            torch.nn.ReLU(),

            # [None, 64, 8, 15] -> [None, 64, 3, 7]
            torch.nn.Conv2d(64, 64, kernel_size=3, stride=2), 
            # [None, 64, 3, 7] -> [None, 64, 2, 6]
            torch.nn.MaxPool2d(2, 1),
            torch.nn.Flatten(),
            torch.nn.Linear(2 * 6 * 64, 128), 
            torch.nn.Tanh(),
            torch.nn.Linear(128, 8), 
            torch.nn.ReLU(),
        )

        # batch, seq, feature
        self.lstm_net = torch.nn.LSTM(OBS_DIM * 2, 
                                      OBS_DIM * 2,
                                      batch_first=True)
        
        self.action_dist = torch.nn.Sequential(
            torch.nn.Sequential(torch.nn.Linear(OBS_DIM * 2, OBS_DIM)), torch.nn.ReLU(),
            torch.nn.Sequential(torch.nn.Linear(OBS_DIM, 2), torch.nn.Softmax(dim=1))
        )

    '''
        input [None, HID_CELL_DIM + OBS_DIM + IMG_DIM + ACT_NUM]
    '''
    def forward(self, input):
        hc = input[:, :HID_CELL_DIM].cuda()
        obs = input[:, HID_CELL_DIM : HID_CELL_DIM + OBS_DIM].cuda()
        img = input[:, HID_CELL_DIM + OBS_DIM : HID_CELL_DIM + OBS_DIM + IMG_DIM].cuda()

        # [None, 26]
        img = torch.reshape(img, [-1, 288, 512, 3])
        img = img.permute(0, 3, 1, 2).cuda()
        img_output = self.cnn_net(img).cuda()
        # [None, 26 + 26 + 20]
        feature = torch.cat([obs, img_output], dim = 1)
        # [None, 26 + 26 + 20]
        h, c = torch.split(hc, split_size_or_sections=int(HID_CELL_DIM/2), dim = 1)
        # [None, 1, 26 + 26 + 20]
        feature, h, c = torch.unsqueeze(feature, dim = 1), torch.unsqueeze(h, dim = 1), torch.unsqueeze(c, dim = 1)
        
        h = h.permute(1,0,2).cuda().contiguous()
        c = c.permute(1,0,2).cuda().contiguous()
        
        latent, (h, c) = self.lstm_net(feature, (h, c))
            
        # [None, 26 + 26 + 20]
        h = torch.squeeze(h, dim = 0)
        c = torch.squeeze(c, dim = 0)
        hc = torch.concat([h, c], dim = 1)
        latent = torch.squeeze(latent, dim = 0)
        
        action_dist = self.action_dist(h)
        
        return action_dist, hc

# output value v(s)
class ValueNet(torch.nn.Module):
    def __init__(self, pn : PolicyNet):
        super(ValueNet, self).__init__()
        self.cnn_net = pn.cnn_net
        self.lstm_net = pn.lstm_net
        
        self.value = torch.nn.Sequential(
            torch.nn.Sequential(torch.nn.Linear(OBS_DIM * 2, OBS_DIM)), torch.nn.ReLU(),
            torch.nn.Sequential(torch.nn.Linear(OBS_DIM , 1))
        )

    def forward(self, input):
        hc = input[:, :HID_CELL_DIM].cuda()
        obs = input[:, HID_CELL_DIM : HID_CELL_DIM + OBS_DIM].cuda()
        img = input[:, HID_CELL_DIM + OBS_DIM : HID_CELL_DIM + OBS_DIM + IMG_DIM].cuda()

        # [None, 26]
        img = torch.reshape(img, [-1, 288, 512, 3])
        img = img.permute(0, 3, 1, 2).cuda()
        img_output = self.cnn_net(img).cuda()
        # [None, 26 + 26 + 20]
        feature = torch.cat([obs, img_output], dim = 1)
        # [None, 26 + 26 + 20]
        h, c = torch.split(hc, split_size_or_sections=int(HID_CELL_DIM/2), dim = 1)
        # [None, 1, 26 + 26 + 20]
        feature, h, c = torch.unsqueeze(feature, dim = 1), torch.unsqueeze(h, dim = 1), torch.unsqueeze(c, dim = 1)
        
        h = h.permute(1,0,2).cuda().contiguous()
        c = c.permute(1,0,2).cuda().contiguous()

        
        latent, (h, c) = self.lstm_net(feature, (h, c))
            
        # [None, 26 + 26 + 20]
        h = torch.squeeze(h, dim = 0)
        c = torch.squeeze(c, dim = 0)
        latent = torch.squeeze(latent, dim = 0)
        hc = torch.concat([h, c], dim = 1)
        
        value = self.value(h)   
        
        return value
    
class PPO():
    def __init__(self, policy_net : PolicyNet, value_net : ValueNet):
        self.policy_net = policy_net
        self.value_net = value_net
        self.actor_optimizer = torch.optim.Adam(policy_net.parameters(), lr=actor_learning_rate)
        self.critic_optimizer = torch.optim.Adam(value_net.parameters(), lr=value_learing_rate)
        self.gamma = 0.95
        self.lam = 0.98
    
    def inference(self, hc, obs, img):
        with torch.no_grad():
            # [None, HID_CELL_DIM + OBS_DIM + IMG_DIM + ACT_NUM]
            inputs = torch.concatenate([hc, obs, img], axis= 1)
            # [None, 26 + 26 + 20]
            logits, hc = self.policy_net(inputs)
            # new_log = torch.log(torch.clamp(new_logits, 1e-5))
            log_p = torch.log(torch.clamp(logits, 1e-20))
            # [None, 1]
            action_dist = torch.distributions.Categorical(logits)
            action = action_dist.sample().item()
            
            v  = self.value_net(inputs)
            
            return action, hc, log_p, v
    
    def get_best_action(self, hc, obs, img, la):
        # [None, HID_CELL_DIM + OBS_DIM + IMG_DIM + ACT_NUM]
        inputs = torch.concatenate([hc, obs, img, la], axis= 1)
        # [None, 26 + 26 + 20]
        logits, hc = self.policy_net(inputs)
        action = torch.argmax(logits)
        # [None, 1]
        # action_dist = torch.distributions.Categorical(logits)
        # action = action_dist.maximum().item()
        
        # v  =self.value_net(inputs)
        
        return action, hc, logits
    
    def get_v(self, hc, obs, img):
        with torch.no_grad():
            inputs = torch.cat([hc, obs, img], dim= 1)
            value = self.value_net(inputs).detach()
            return value
    
    # def compute_advantage(self, br, bov):
    #     with torch.no_grad():
    #         # bv = self.value_net(torch.cat([bhc, bs, bimg, bla], dim = 1)).detach()
    #         # old_log_probs, _ = self.policy_net(torch.cat([bhc, bs, bimg, bla], dim=1))
    #         # old_log_probs = old_log_probs.detach()
    #         return br - bov
    
    def update(self, bs, ba, bhc, bimg, bov, old_log, badv, btv):
        # adv = self.compute_advantage(br, bov)
        new_logits, _ = self.policy_net(torch.cat([bhc, bs, bimg], dim = 1))
        new_log = torch.log(torch.clamp(new_logits, 1e-20))
        
        ratio = torch.exp(new_log - old_log)
        
        adv_action = torch.zeros([ratio.shape[0], ACT_NUM])
        for i in range(len(badv)):
            adv_action[i][int(ba[i][0])] = badv[i][0]

        ratio = ratio.cuda()
        adv_action = adv_action.cuda()
        surr1 = ratio * adv_action
        surr2 = torch.clamp(ratio, 1 - 0.2, 1+ 0.2) * adv_action

        new_value = self.value_net(torch.cat([bhc, bs, bimg], dim = 1))

        actor_loss = torch.mean(-torch.min(surr1, surr2)) - torch.mean(torch.distributions.Categorical(new_logits).entropy()) * 0.05
        critic_loss = torch.mean(torch.nn.functional.mse_loss(new_value, btv)) * 0.5
        
        self.actor_optimizer.zero_grad()
        self.critic_optimizer.zero_grad()
        actor_loss.backward()
        critic_loss.backward()
        self.actor_optimizer.step()
        self.critic_optimizer.step()
        return actor_loss, critic_loss

    def compute_generalized_advantage_estimator(self, mb_rewards, mb_values, lastValue, Steps):
        with torch.no_grad():
            mb_rewards = torch.asarray(mb_rewards, dtype=torch.float32)
            mb_values = torch.asarray(mb_values, dtype=torch.float32)
            mb_advs = torch.zeros_like(mb_rewards)
            mb_target_values = torch.zeros_like(mb_rewards)
            lastgaelam = 0
            nextnonterminal = 0
            for t in reversed(range(Steps)):  # 倒序实现，便于进行递归
                if t == Steps - 1:  # 如果是最后一步，要判断当前是否是终止状态，如果是，next_value就是0
                    nextnonterminal = 0
                    nextvalues = lastValue
                else:
                    nextnonterminal = 1.0 
                    nextvalues = mb_values[t + 1]
                mb_target_values[t] = mb_rewards[t] + self.gamma * nextvalues * nextnonterminal
                delta = mb_target_values[t] - mb_values[t]
                mb_advs[t] = lastgaelam = delta + self.gamma * self.lam * nextnonterminal * lastgaelam

            return mb_advs,  mb_advs + mb_values


def train():
    torch.cuda.manual_seed(1993)

    # actor = PolicyNet().cuda()
    # critic = ValueNet(actor).cuda()
    
    # Model = PPO(actor, critic)

    actor = PolicyNet().cuda()
    actor.load_state_dict(torch.load("actor.pth"))
    # actor.eval()
    critic = ValueNet(actor).cuda()
    critic.load_state_dict(torch.load("critic.pth"))
    # critic.eval()
    Model = PPO(actor, critic)

    p.init()

    buffer_s, buffer_a, buffer_hc, buffer_img, buffer_old_v, buffer_old_dist, buffer_adv, buffer_target_value= [], [], [], [], [], [], [], []
    
    terminated = False
    all_ep_r = []
    # in one episode
    for ep in range(EP_MAX): 
        ep_r = 0
        p.reset_game()
        # for lstm
        hc = torch.zeros([1, HID_CELL_DIM]).cuda()
        terminated = False
        ep_buffer_s, ep_buffer_a, ep_buffer_r, ep_buffer_hc, ep_buffer_img, ep_buffer_old_v, ep_buffer_old_dist = [], [], [], [], [], [], []
        
        while not terminated:
            ori_obs = p.getGameState()
            obs = state_to_vector(ori_obs)
            img = p.getScreenRGB()
            img = torch.reshape(torch.from_numpy(img), [1, IMG_DIM]).type(torch.float32).cuda()
            obs = torch.reshape(torch.from_numpy(obs), [1, OBS_DIM]).type(torch.float32).cuda()
            hc = torch.reshape(hc, [1,HID_CELL_DIM]).type(torch.float32).cuda()


            action, hc, old_dist, old_v = Model.inference(hc, obs, img)
            reward = p.act(actions[action])
            if reward > 0:
                reward *= 3
            # if reward < 0:
            #     print("action: {} reward: {}".format(action, reward))
            if action == 0:
                reward += 0.1
            else:
                reward -= 0.2
            terminated = p.game_over()

            ep_buffer_s.append(np.squeeze(obs, axis=0))
            ep_buffer_hc.append(np.squeeze(hc, axis=0))
            ep_buffer_a.append(action)
            ep_buffer_r.append(reward)
            ep_buffer_img.append(np.squeeze(img, axis=0))
            ep_buffer_old_v.append(np.squeeze(old_v, axis=0))
            ep_buffer_old_dist.append(np.squeeze(old_dist, axis = 0))
            ep_r += reward
    
        # calculate discounted reward after episode finished
        # print("ebs: %s, eba: %s, ebhc: %s, ebla: %s, ebimg: %s, ebov: %s, ebodist: %s", 
        #         len(ep_buffer_s), len(ep_buffer_a), len(ep_buffer_hc), len(ep_buffer_la), len(ep_buffer_img), len(ep_buffer_old_v), len(ep_buffer_old_dist))
        
        lastValue = Model.get_v(hc, obs, img).cuda()
        
        bs = torch.vstack(ep_buffer_s).cuda()
        bhc = torch.vstack(ep_buffer_hc).cuda()
        bimg = torch.vstack(ep_buffer_img).cuda()
        ba = torch.FloatTensor(ep_buffer_a).reshape([len(ep_buffer_a), 1]).cuda()
        br = torch.FloatTensor(ep_buffer_r).reshape([len(ep_buffer_r), 1]).cuda()

        v_s_ = Model.get_v(bhc, bs, bimg).cuda() # The value of last state, criticed by value network
        adv, target_value = Model.compute_generalized_advantage_estimator(br, v_s_, lastValue, len(v_s_))

        bov = torch.FloatTensor(ep_buffer_old_v).reshape([len(ep_buffer_old_v), 1])
        bodist = torch.vstack(ep_buffer_old_dist)
        badv = adv
        btr = target_value
        
        buffer_s += bs
        buffer_a += ba
        buffer_hc += bhc
        buffer_img += bimg
        buffer_old_v += bov
        buffer_old_dist += bodist
        buffer_adv += badv
        buffer_target_value += btr
        
        # replay buffer to model
        # only get latest replay
        if len(buffer_s) > BATCH_SIZE:
            idx = len(buffer_s) - BATCH_SIZE
            buffer_s = buffer_s[idx:]
            buffer_a = buffer_a[idx:]
            buffer_hc = buffer_hc[idx:]
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
        
        # print("ubs: %s, u_ba: %s, ubhc: %s, u_bla: %s, u_bimg: %s, ubov: %s, ubodist: %s, ubadv: %s, ubtv: %s", 
                # u_bs.shape, u_ba.shape, u_bhc.shape, u_bla.shape, u_bimg.shape, u_bov.shape, u_bodist.shape, u_badv.shape, u_btv.shape)

        for i in range(UPDATE_NUM):
            actor_loss, critic_loss = Model.update(u_bs, u_ba, u_bhc, u_bimg, u_bov, u_bodist, u_badv, u_btv)
            if i == UPDATE_NUM - 1 and ep % SAVE_EPOCH == 0:
                logger.info("Ep: %s, |Ep_r: %s| Value(state): %s, actor_loss: %s, critic_loss: %s" , ep, ep_r, torch.mean(u_bov), actor_loss, critic_loss)
            
        if ep > 0 and ep % SAVE_EPOCH == 0:
            torch.save(actor.state_dict(), "actor.pth")
            torch.save(critic.state_dict(), 'critic.pth')

        time.sleep(0.3)
            

    plt.plot(np.arange(len(all_ep_r)), all_ep_r)
    plt.xlabel('Episode')
    plt.ylabel('Moving averaged episode reward')
    plt.show()
    
if __name__ == "__main__":
    train()
