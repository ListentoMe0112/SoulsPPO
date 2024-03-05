import torch
import numpy as np
from constant import constant

actor_learning_rate = 0.002
value_learing_rate  = 0.002

# output action [pi(s)]
class PolicyNet(torch.nn.Module):
    def __init__(self) -> None:
        super(PolicyNet, self).__init__()
        self.cnn_net = torch.nn.Sequential(
            # [None, 3, 90, 160] -> [None, 32, 42, 77]
            torch.nn.Sequential(
                torch.nn.Conv2d(3, 32, kernel_size=8, stride=2), 
                # [None, 32, 42, 77] -> [None, 32, 21, 38]
                torch.nn.MaxPool2d(2,2),
                torch.nn.ReLU()),
            # [None, 32, 21, 38] -> [None, 64, 9, 18]
            torch.nn.Sequential(
                torch.nn.Conv2d(32, 64, kernel_size=4, stride=2),
                # [None, 64, 9, 18] -> [None, 64, 4, 9]
                torch.nn.MaxPool2d(2,2), 
                torch.nn.ReLU()),
            # [None, 64, 4, 9] -> [64, 2, 7]
            torch.nn.Sequential(
                torch.nn.Conv2d(64, 64, kernel_size=3, stride=1), 
                torch.nn.ReLU()),
            torch.nn.Flatten(),
            torch.nn.Sequential(
                torch.nn.Linear(2 * 7 * 64, 64), 
                torch.nn.Tanh()),
                torch.nn.Linear(64, constant.OBS_DIM),
                torch.nn.Tanh()
        )

        # batch, seq, feature
        self.lstm_net = torch.nn.LSTM(constant.OBS_DIM + constant.OBS_DIM + constant.ACT_NUM, 
                                      constant.OBS_DIM + constant.OBS_DIM + constant.ACT_NUM,
                                      batch_first=True)
        
        self.action_dist = torch.nn.Sequential(
            torch.nn.Sequential(torch.nn.Linear(constant.OBS_DIM + constant.OBS_DIM + constant.ACT_NUM, 20)), 
        )

    '''
        input [None, HID_CELL_DIM + OBS_DIM + IMG_DIM + ACT_NUM]
    '''
    def forward(self, input):
        hc = input[:, :constant.HID_CELL_DIM].cuda()
        obs = input[:, constant.HID_CELL_DIM : constant.HID_CELL_DIM + constant.OBS_DIM].cuda()
        img = input[:, constant.HID_CELL_DIM + constant.OBS_DIM : constant.HID_CELL_DIM + constant.OBS_DIM + constant.IMG_DIM].cuda()
        legal_action = input[:, constant.HID_CELL_DIM + constant.OBS_DIM + constant.IMG_DIM :].cuda()

        # [None, 26]
        img = torch.reshape(img, [-1, 90, 160, 3])
        img = img.permute(0, 3, 1, 2).cuda()
        img_output = self.cnn_net(img).cuda()
        # [None, 26 + 26 + 20]
        feature = torch.cat([obs, img_output, legal_action], dim = 1)
        # [None, 26 + 26 + 20]
        h, c = torch.split(hc, split_size_or_sections=int(constant.HID_CELL_DIM/2), dim = 1)
        # [None, 1, 26 + 26 + 20]
        feature, h, c = torch.unsqueeze(feature, dim = 1), torch.unsqueeze(h, dim = 1), torch.unsqueeze(c, dim = 1)
        
        h = h.permute(1,0,2).cuda().contiguous()
        c = c.permute(1,0,2).cuda().contiguous()
        
        latent, (h, c) = self.lstm_net(feature, (h, c))
            
        # [None, 26 + 26 + 20]
        h = torch.squeeze(h, dim = 1)
        c = torch.squeeze(c, dim = 1)
        latent = torch.squeeze(latent, dim = 1)
        hc = torch.concat([h, c], dim = 1)
        
        logists = self.action_dist(latent)
        values, indexes = torch.min(logists, dim=1, keepdim=True)
        logists = logists - values
        # Try Avoid sampling illegal action
        logists = logists + (legal_action - 1) * (32)
        action_dist = torch.softmax(logists, dim = 1)
        
        legal_action_dist = action_dist

        return legal_action_dist, hc

# output value v(s)
class ValueNet(torch.nn.Module):
    def __init__(self, pn : PolicyNet):
        super(ValueNet, self).__init__()
        self.cnn_net = pn.cnn_net
        self.lstm_net = pn.lstm_net
        
        self.value = torch.nn.Sequential(
            torch.nn.Sequential(torch.nn.Linear(constant.OBS_DIM + constant.OBS_DIM + constant.ACT_NUM, 16)),
            torch.nn.ReLU(),
            torch.nn.Linear(16, 1)
        )

    def forward(self, input):
        hc = input[:, :constant.HID_CELL_DIM].cuda()
        obs = input[:, constant.HID_CELL_DIM : constant.HID_CELL_DIM + constant.OBS_DIM].cuda()
        img = input[:, constant.HID_CELL_DIM + constant.OBS_DIM : constant.HID_CELL_DIM + constant.OBS_DIM + constant.IMG_DIM].cuda()
        legal_action = input[:, constant.HID_CELL_DIM + constant.OBS_DIM + constant.IMG_DIM :].cuda()

        # [None, 26]
        img = torch.reshape(img, [-1, 90, 160, 3])
        img = img.permute(0, 3, 1, 2).cuda()
        img_output = self.cnn_net(img).cuda()
        # [None, 26 + 26 + 20]
        feature = torch.cat([obs, img_output, legal_action], dim = 1)
        # [None, 26 + 26 + 20]
        h, c = torch.split(hc, split_size_or_sections=int(constant.HID_CELL_DIM/2), dim = 1)
        # [None, 1, 26 + 26 + 20]
        feature, h, c = torch.unsqueeze(feature, dim = 1), torch.unsqueeze(h, dim = 1), torch.unsqueeze(c, dim = 1)
        
        h = h.permute(1,0,2).cuda().contiguous()
        c = c.permute(1,0,2).cuda().contiguous()
        
        latent, (h, c) = self.lstm_net(feature, (h, c))
        
        # [None, 26 + 26 + 20]
        h = torch.squeeze(h, dim = 1)
        c = torch.squeeze(c, dim = 1)
        latent = torch.squeeze(latent, dim = 1)
        hc = torch.concat([h, c], dim = 1)
        
        value = self.value(latent)         
        
        return value
    
class PPO():
    def __init__(self, policy_net : PolicyNet, value_net : ValueNet):
        self.policy_net = policy_net
        self.value_net = value_net
        self.actor_optimizer = torch.optim.Adam(policy_net.parameters(), lr=actor_learning_rate)
        self.critic_optimizer = torch.optim.Adam(value_net.parameters(), lr=value_learing_rate)
        self.gamma = constant.GAMMA
        self.lam = constant.LAMBDA
    
    def inference(self, hc, obs, img, la):
        with torch.no_grad():
            # [None, HID_CELL_DIM + OBS_DIM + IMG_DIM + ACT_NUM]
            inputs = torch.concatenate([hc, obs, img, la], axis= 1)
            # [None, 26 + 26 + 20]
            p, hc = self.policy_net(inputs)
            log_p = torch.log(p)
            # new_log = torch.log(torch.clamp(new_logits, 1e-5))
            # log_p = torch.log(torch.clamp(logits, 1e-20))
            # [None, 1]
            action_dist = torch.distributions.Categorical(p)
            action = action_dist.sample().item()
            
            v  = self.value_net(inputs)
            
            return action, hc, log_p, v
    
    # def get_best_action(self, hc, obs, img, la):
    #     # [None, HID_CELL_DIM + OBS_DIM + IMG_DIM + ACT_NUM]
    #     inputs = torch.concatenate([hc, obs, img, la], axis= 1)
    #     # [None, 26 + 26 + 20]
    #     logits, hc = self.policy_net(inputs)
    #     action = torch.argmax(logits)
    #     # [None, 1]
    #     # action_dist = torch.distributions.Categorical(logits)
    #     # action = action_dist.maximum().item()
        
    #     # v  =self.value_net(inputs)
        
        # return action, hc, logits
    
    def get_v(self, hc, obs, img, la):
        with torch.no_grad():
            inputs = torch.concatenate([hc, obs, img, la], axis= 1)
            value = self.value_net(inputs).detach()
            return value
    
    # def compute_advantage(self, br, bov):
    #     with torch.no_grad():
    #         # bv = self.value_net(torch.cat([bhc, bs, bimg, bla], dim = 1)).detach()
    #         # old_log_probs, _ = self.policy_net(torch.cat([bhc, bs, bimg, bla], dim=1))
    #         # old_log_probs = old_log_probs.detach()
    #         return br - bov
    
    def update(self, bs, ba, bhc, bla, bimg, bov, old_log, badv, btv):
        # adv = self.compute_advantage(br, bov)
        p, _ = self.policy_net(torch.cat([bhc, bs, bimg, bla], dim = 1))
        new_log = torch.log(p)
        
        ratio = torch.exp(new_log - old_log)
        
        adv_action = torch.zeros([ratio.shape[0], constant.ACT_NUM])
        for i in range(len(badv)):
            adv_action[i][int(ba[i][0])] = badv[i][0]

        ratio = ratio.cuda()
        adv_action = adv_action.cuda()
        surr1 = ratio * adv_action
        surr2 = torch.clamp(ratio, 1 - 0.2, 1+ 0.2) * adv_action

        new_value = self.value_net(torch.cat([bhc, bs, bimg, bla], dim = 1))

        actor_loss = torch.mean(-torch.min(surr1, surr2)) - torch.mean(torch.distributions.Categorical(p).entropy()) * 0.05
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

            return mb_advs,  mb_target_values