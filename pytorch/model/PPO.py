import torch
import numpy as np
from constant import constant

actor_learning_rate = 0.001
value_learing_rate = 0.001

# output action [pi(s)]
class PolicyNet(torch.nn.Module):
    def __init__(self) -> None:
        super(PolicyNet, self).__init__()
        self.cnn_net = torch.nn.Sequential(
            # [None, 3, 90, 160] -> [None, 32, 21, 39]
            torch.nn.Sequential(torch.nn.Conv2d(3, 32, kernel_size=8, stride=4), torch.nn.ReLU()),
            # [None, 32, 21, 39] -> [None, 64, 9, 18]
            torch.nn.Sequential(torch.nn.Conv2d(32, 64, kernel_size=4, stride=2), torch.nn.ReLU()),
            # [None, 64, 9, 18] -> [64, 7, 16]
            torch.nn.Sequential(torch.nn.Conv2d(64, 64, kernel_size=3, stride=1), torch.nn.ReLU()),
            torch.nn.Flatten(),
            torch.nn.Sequential(torch.nn.Linear(16 * 7 * 64, 512), torch.nn.Tanh()),
            torch.nn.Sequential(torch.nn.Linear(512, 26), torch.nn.ReLU())
        )

        # batch, seq, feature
        self.lstm_net = torch.nn.LSTM(constant.OBS_DIM + constant.OBS_DIM + constant.ACT_NUM, 
                                      constant.OBS_DIM + constant.OBS_DIM + constant.ACT_NUM,
                                      batch_first=True)
        
        self.action_dist = torch.nn.Sequential(
            torch.nn.Sequential(torch.nn.Linear(constant.OBS_DIM + constant.OBS_DIM + constant.ACT_NUM, 20)), torch.nn.Softmax()
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
        
        action_dist = self.action_dist(latent)
        
        legal_action_dist = action_dist * legal_action
        
        return legal_action_dist, hc

# output value v(s)
class ValueNet(torch.nn.Module):
    def __init__(self):
        super(ValueNet, self).__init__()
        self.cnn_net = torch.nn.Sequential(
            # [None, 3, 90, 160] -> [None, 32, 21, 39]
            torch.nn.Sequential(torch.nn.Conv2d(3, 32, kernel_size=8, stride=4), torch.nn.ReLU()),
            # [None, 32, 21, 39] -> [None, 64, 9, 18]
            torch.nn.Sequential(torch.nn.Conv2d(32, 64, kernel_size=4, stride=2), torch.nn.ReLU()),
            # [None, 64, 9, 18] -> [64, 7, 16]
            torch.nn.Sequential(torch.nn.Conv2d(64, 64, kernel_size=3, stride=1), torch.nn.ReLU()),
            torch.nn.Flatten(),
            torch.nn.Sequential(torch.nn.Linear(16 * 7 * 64, 512), torch.nn.Tanh()),
            torch.nn.Sequential(torch.nn.Linear(512, 26), torch.nn.ReLU())
        )

        # batch, seq, feature
        self.lstm_net = torch.nn.LSTM(constant.OBS_DIM + constant.OBS_DIM + constant.ACT_NUM, 
                                      constant.OBS_DIM + constant.OBS_DIM + constant.ACT_NUM,
                                      batch_first=True)
        
        self.value = torch.nn.Sequential(
            torch.nn.Sequential(torch.nn.Linear(constant.OBS_DIM + constant.OBS_DIM + constant.ACT_NUM, 1))
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
    
    def get_action(self, hc, obs, img, la):
        # [None, HID_CELL_DIM + OBS_DIM + IMG_DIM + ACT_NUM]
        inputs = torch.concatenate([hc, obs, img, la], axis= 1)
        # [None, 26 + 26 + 20]
        logits, hc = self.policy_net(inputs)
        # [None, 1]
        action_dist = torch.distributions.Categorical(logits)
        action = action_dist.sample().item()
        
        return action, hc
    
    def get_v(self, hc, obs, img, la):
        with torch.no_grad():
            inputs = torch.concatenate([hc, obs, img, la], axis= 1)
            value = self.value_net(inputs).detach()
            return value
    
    def compute_advantage(self, bs, ba, br, bhc, bla, bimg):
        with torch.no_grad():
            bv = self.value_net(torch.cat([bhc, bs, bimg, bla], dim = 1)).detach()
            old_log_probs, _ = self.policy_net(torch.cat([bhc, bs, bimg, bla], dim=1))
            old_log_probs = old_log_probs.detach()
            return br.cuda() - bv.cuda(), bv, old_log_probs
    
    def update(self, bs, ba, br, bhc, bla, bimg):
        adv, _, old_log_probs = self.compute_advantage(bs, ba, br, bhc, bla, bimg)
        log_probs, _ = self.policy_net(torch.cat([bhc, bs, bimg, bla], dim = 1))
        ratio = torch.exp(log_probs - old_log_probs)
        adv_action = torch.zeros([ratio.shape[0], constant.ACT_NUM])
        for i in range(len(adv)):
            adv_action[i][int(ba[0][i])] = adv[0][i]

        ratio = ratio.cuda()
        adv_action = adv_action.cuda()
        surr1 = ratio * adv_action
        surr2 = torch.clamp(ratio, 1 - 0.2, 1+ 0.2) * adv_action
        actor_loss = torch.mean(-torch.min(surr1, surr2))
        critic_loss = torch.mean(torch.nn.functional.mse_loss(self.value_net(torch.cat([bhc, bs, bimg, bla], dim = 1)).cuda(), br.cuda()))
        self.actor_optimizer.zero_grad()
        self.critic_optimizer.zero_grad()
        actor_loss.backward(retain_graph=True)
        critic_loss.backward()
        self.actor_optimizer.step()
        self.critic_optimizer.step()