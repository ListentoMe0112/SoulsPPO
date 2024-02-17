import gym
import torch

import torch
import numpy as np
import math

import matplotlib.pyplot as plt

actor_learning_rate = 0.02
value_learing_rate = 0.02

# output action [pi(s)]
class PolicyNet(torch.nn.Module):
    def __init__(self) -> None:
        super(PolicyNet, self).__init__()

        # batch, seq, feature
        self.lstm_net = torch.nn.LSTM(4, 
                                      4,
                                      batch_first=True)
        
        self.action_dist = torch.nn.Sequential(
            torch.nn.Sequential(torch.nn.Linear(4,2)), torch.nn.Softmax(dim=1)
        )

    '''
        input [None, HID_CELL_DIM + OBS_DIM + IMG_DIM + ACT_NUM]
    '''
    def forward(self, inputs):
        hc = inputs[:, :8]
        obs = inputs[:, 8:]
        h, c = torch.split(hc, split_size_or_sections=int(8/2), dim = 1)
        # [None, 1, 26 + 26 + 20]
        obs, h, c = torch.unsqueeze(obs, dim = 1), torch.unsqueeze(h, dim = 1), torch.unsqueeze(c, dim = 1)
        
        h = h.permute(1,0,2).contiguous()
        c = c.permute(1,0,2).contiguous()

        
        latent, (h, c) = self.lstm_net(obs, (h, c))
            
        # [None, 26 + 26 + 20]
        h = torch.squeeze(h, dim = 1)
        c = torch.squeeze(c, dim = 1)
        latent = torch.squeeze(latent, dim = 1)
        hc = torch.concat([h, c], dim = 1)
        
        action_dist = self.action_dist(latent)
        
        legal_action_dist = action_dist
        return legal_action_dist, hc

# output value v(s)
class ValueNet(torch.nn.Module):
    def __init__(self, pn : PolicyNet):
        super(ValueNet, self).__init__()
                # batch, seq, feature
        self.lstm_net = pn.lstm_net
        
        self.value = torch.nn.Sequential(
            torch.nn.Sequential(torch.nn.Linear(4, 1))
        )

    def forward(self, input):
        hc = input[:, :8]
        obs = input[:, 8:]

        # [None, 26 + 26 + 20]
        # [None, 26 + 26 + 20]
        h, c = torch.split(hc, 4, dim = 1)
        # [None, 1, 26 + 26 + 20]
        obs, h, c = torch.unsqueeze(obs, dim = 1), torch.unsqueeze(h, dim = 1), torch.unsqueeze(c, dim = 1)
        
        h = h.permute(1,0,2).contiguous()
        c = c.permute(1,0,2).contiguous()
        
        _, (h, c) = self.lstm_net(obs, (h, c))
        latent = torch.squeeze(h, dim = 0)
        value = self.value(latent)         
        
        return value
    
class PPO():
    def __init__(self, policy_net : PolicyNet, value_net : ValueNet):
        self.policy_net = policy_net
        self.value_net = value_net
        self.actor_optimizer = torch.optim.Adam(policy_net.parameters(), lr=actor_learning_rate)
        self.critic_optimizer = torch.optim.Adam(value_net.parameters(), lr=value_learing_rate)
        self.gamma = 0.9
        self.lam = 0.5
    
    def inference(self, hc, obs):
        with torch.no_grad():
            # [None, HID_CELL_DIM + OBS_DIM + IMG_DIM + ACT_NUM]
            inputs = torch.concatenate([hc, obs], axis= 1)
            # [None, 26 + 26 + 20]
            logits, hc = self.policy_net(inputs)
            # new_log = torch.log(torch.clamp(new_logits, 1e-5))
            log_p = torch.log(torch.clamp(logits, 1e-5))
            # [None, 1]
            action_dist = torch.distributions.Categorical(logits)
            action = action_dist.sample().item()
            
            v  = self.value_net(inputs)
            
            return action, hc, log_p, v
    
    def get_best_action(self, hc, obs):
        # [None, HID_CELL_DIM + OBS_DIM + IMG_DIM + ACT_NUM]
        inputs = torch.concatenate([hc, obs], axis= 1)
        # [None, 26 + 26 + 20]
        logits, hc = self.policy_net(inputs)
        action = torch.argmax(logits)
        # [None, 1]
        # action_dist = torch.distributions.Categorical(logits)
        # action = action_dist.maximum().item()
        
        # v  =self.value_net(inputs)
        
        return action, hc, logits
    
    def get_v(self, hc, obs):
        with torch.no_grad():
            inputs = torch.concatenate([hc, obs], axis= 1)
            value = self.value_net(inputs)
            return value
    
    # def compute_advantage(self, br, bov):
    #     with torch.no_grad():
    #         # bv = self.value_net(torch.cat([bhc, bs, bimg, bla], dim = 1)).detach()
    #         # old_log_probs, _ = self.policy_net(torch.cat([bhc, bs, bimg, bla], dim=1))
    #         # old_log_probs = old_log_probs.detach()
    #         return br - bov
    
    def update(self, bs, ba, bhc, bov, old_log, badv, btv):
        # adv = self.compute_advantage(br, bov)
        new_logits, _ = self.policy_net(torch.cat([bhc, bs], dim = 1))
        new_log = torch.log(torch.clamp(new_logits, 1e-5))
        
        ratio = torch.exp(new_log - old_log)
        
        adv_action = torch.zeros([ratio.shape[0], 2])
        for i in range(len(badv)):
            adv_action[i][int(ba[i][0])] = badv[i][0]

        ratio = ratio
        adv_action = adv_action
        surr1 = ratio * adv_action
        surr2 = torch.clamp(ratio, 1 - 0.2, 1 + 0.2) * adv_action

        new_value = self.value_net(torch.cat([bhc, bs], dim = 1))

        # actor_loss = torch.mean(- torch.min(surr1, surr2)) - torch.mean(torch.distributions.Categorical(new_logits).entropy()) * 0.05
        actor_loss = torch.mean(- torch.min(surr1, surr2))
        critic_loss = torch.mean(torch.nn.functional.mse_loss(new_value, btv))
        
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
            return mb_advs, mb_advs + mb_values


env = gym.make('CartPole-v0')
env = env.unwrapped
def train():
    actor = PolicyNet()
    critic = ValueNet(actor)
    
    Model = PPO(actor, critic)

    buffer_s, buffer_a, buffer_r, buffer_hc, buffer_old_v, buffer_old_dist, buffer_adv, buffer_target_value= [], [], [], [], [], [], [], []

    buffer_ep_r = []

    for ep in range(2048):
        observation = env.reset()
        ep_r = 0
        done = False
        hc = torch.zeros([1,8], dtype=torch.float32)
        ep_buffer_s, ep_buffer_a, ep_buffer_r, ep_buffer_hc, ep_buffer_old_v, ep_buffer_old_dist = [], [], [], [], [], []
        while not done:
            env.render()
            observation = torch.from_numpy(observation)
            observation = observation.view(1,4)
            observation = torch.tensor(observation, dtype=torch.float32)
            action, hc, old_dist, old_v = Model.inference(hc, observation)
            observation_, reward, done, info = env.step(action)

            x, x_dot, theta, theta_dot = observation_

            # reward = math.exp((env.x_threshold - abs(x)) / env.x_threshold) - math.exp(1) / 2
            # reward += math.exp((env.theta_threshold_radians - abs(theta)) / env.theta_threshold_radians)- math.exp(1) / 2

            ep_buffer_s.append(np.squeeze(observation, axis=0))
            ep_buffer_hc.append(np.squeeze(hc, axis=0))
            ep_buffer_a.append(action)
            ep_buffer_r.append(reward)
            ep_buffer_old_v.append(np.squeeze(old_v, axis=0))
            ep_buffer_old_dist.append(np.squeeze(old_dist, axis = 0))

            observation = observation_
            ep_r += reward
        
        observation = torch.from_numpy(observation)
        observation = observation.view(1,4)
        observation = torch.tensor(observation, dtype=torch.float32)
        lastValue = Model.get_v(hc, observation)
        
        bs = torch.vstack(ep_buffer_s)
        bhc = torch.vstack(ep_buffer_hc)
        ba = torch.FloatTensor(ep_buffer_a).reshape([len(ep_buffer_a), 1])
        br = torch.FloatTensor(ep_buffer_r).reshape([len(ep_buffer_r), 1])
        
        v_s_ = Model.get_v(bhc, bs) # The value of last state, criticed by value network
        adv, target_value = Model.compute_generalized_advantage_estimator(br, v_s_, lastValue, len(v_s_))

        # print(target_value)

        bov = torch.FloatTensor(ep_buffer_old_v).reshape([len(ep_buffer_old_v), 1])
        bodist = torch.vstack(ep_buffer_old_dist)
        badv = adv
        btr = target_value
        
        buffer_s += bs
        buffer_a += ba
        buffer_r += br
        buffer_hc += bhc
        buffer_old_v += bov
        buffer_old_dist += bodist
        buffer_adv += badv
        buffer_target_value += btr
        
        # replay buffer to model
        # only get latest replay
        if len(buffer_s) > 128:
            idx = len(buffer_s) - 128
            buffer_s = buffer_s[idx:]
            buffer_a = buffer_a[idx:]
            buffer_hc = buffer_hc[idx:]
            buffer_r = buffer_r[idx:]
            buffer_old_dist = buffer_old_dist[idx:]
            buffer_old_v = buffer_old_v[idx:]
            buffer_adv = buffer_adv[idx:]
            buffer_target_value = buffer_target_value[idx:]
            
        u_bs = torch.vstack(buffer_s)
        u_ba = torch.FloatTensor(buffer_a).reshape([len(buffer_a), 1])
        u_bhc = torch.vstack(buffer_hc)
        u_bov = torch.FloatTensor(buffer_old_v).reshape([len(buffer_old_v), 1])
        u_bodist = torch.vstack(buffer_old_dist)
        u_badv = torch.FloatTensor(buffer_adv).reshape([len(buffer_adv), 1])
        u_btv = torch.FloatTensor(buffer_target_value).reshape([len(buffer_target_value), 1])
        
        # print("ubs: %s, u_ba: %s, ubhc: %s, u_bla: %s, u_bimg: %s, ubov: %s, ubodist: %s, ubadv: %s, ubtv: %s", 
                # u_bs.shape, u_ba.shape, u_bhc.shape, u_bla.shape, u_bimg.shape, u_bov.shape, u_bodist.shape, u_badv.shape, u_btv.shape)

        actor_loss, critic_loss = Model.update(u_bs, u_ba, u_bhc, u_bov, u_bodist, u_badv, u_btv)
        print("Ep: {}, |Ep_r: {}| Value(state): {}, actor_loss: {}, critic_loss: {}".format(ep, ep_r, torch.mean(u_bov), actor_loss, critic_loss))
        
        buffer_ep_r.append(ep_r)


    x = [i for i in range(2048)]
    plt.plot(x, buffer_ep_r)
    plt.savefig("test")
if __name__ == "__main__":
    train()