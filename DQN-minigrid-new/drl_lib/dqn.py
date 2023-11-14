import random
import torch.nn.utils as utils
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import replays
import time
class DQNAgent():
    def __init__(self, net, args, writer):
        self.net = net
        self.args = args
        self.optimizer = optim.Adam(self.net.parameters(), lr=args.lr)
        self.epsilon = args.epsilon_init
        self.writer = writer
        self.training_step = 0
        self.epsilon_delta = (args.epsilon_init - args.epsilon_min)/args.epsilon_decay_step
    def select_action(self, obs):
        """
        Selects an action based on the current observation
        """
        with torch.no_grad():
            q_values = self.net(obs.unsqueeze(0))
            if random.random() < self.epsilon:
                action = random.randrange(self.net.num_actions)
            else:
                action = q_values.argmax().item()
        self.epsilon = max(self.epsilon - self.epsilon_delta, self.args.epsilon_min)
        return action

    def update(self, replay_buffer, total_step):
        """
        Updates the Q-network based on a batch of experiences
        """
        obs_batch, action_batch, reward_batch, next_obs_batch, done_batch,weight,indices = replay_buffer.sample(total_step)

        obs_tensor = torch.from_numpy(obs_batch).float()
        action_tensor = torch.from_numpy(action_batch).unsqueeze(1)
        reward_tensor = torch.from_numpy(reward_batch).unsqueeze(1)
        next_obs_tensor = torch.from_numpy(next_obs_batch).float()
        done_tensor = torch.from_numpy(done_batch).unsqueeze(1)

        q_values = self.net(obs_tensor).gather(1, action_tensor)

        next_q_values = self.net(next_obs_tensor).detach().max(1)[0].unsqueeze(1)
        target_q_values = reward_tensor + self.args.gamma * next_q_values * (1 - done_tensor)

        if self.args.replay == "EPER":
            replay_buffer.get_q(indices, q_values, next_obs_tensor, action_tensor,reward_tensor, done_tensor, self.args.gamma, None,None,
                                target_q_values)
            # replay_buffer.check_stage_change(td_error_list)
        if self.args.replay == "proportional_PER" or self.args.replay == "PER":
            td_error = target_q_values - q_values
            td_error_list = []
            for t in td_error:
                td_error_list.append(abs(t[0].item()))
            # print(td_error_list)
            replay_buffer.priority_update(indices, td_error_list)
        weight = torch.FloatTensor(weight)
        loss = F.smooth_l1_loss(q_values, target_q_values,reduction='none')
        loss = torch.mean(weight*loss)
        # self.writer.add_scalar("loss", loss, global_step=self.training_step)

        self.optimizer.zero_grad()
        loss.backward()
        utils.clip_grad_norm_(list(self.net.parameters()), 0.5)
        self.optimizer.step()
        self.training_step += 1

    def save(self, directory, step):
        step = str(step)
        torch.save(self.net.state_dict(), '%s/%sk.pth' % (directory, step))
    def load(self, directory):
        self.net.load_state_dict(torch.load(directory))

# class DQN(nn.Module):
#     def __init__(self, input_shape, num_actions):
#         super(DQN, self).__init__()
#         self.input_shape = input_shape
#         self.num_actions = num_actions
#         print(input_shape[0]*input_shape[1]*input_shape[2])
#         self.fc_layer = nn.Sequential(
#             nn.Linear(input_shape[0]*input_shape[1]*input_shape[2], 128),
#             nn.ReLU(),
#             nn.Linear(128, 64),
#             nn.ReLU(),
#             nn.Linear(64, num_actions)
#         )
#
#     def forward(self, x):
#         x = x.reshape(x.shape[0], -1)
#         x = self.fc_layer(x)
#         return x
class DQN(nn.Module):
    def __init__(self,input_shape, num_actions):
        super(DQN, self).__init__()
        self.input_shape = input_shape
        self.num_actions = num_actions
        self.conv_layer = nn.Sequential(
            nn.Conv2d(3, 32, (2, 2)),
            nn.ReLU(),
            nn.Conv2d(32, 64, (2, 2)),
            nn.ReLU(),
            nn.Conv2d(64, 128, (2, 2)),
            nn.ReLU()
        )
        self.fc_layer = nn.Sequential(
            nn.Linear(self.get_conv_output_dim(), 16),
            nn.Tanh(),
            nn.Linear(16, num_actions)
        )
        print(self.get_conv_output_dim())

    def forward(self, x):
        x = self.conv_layer(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layer(x)
        return x

    def get_conv_output_dim(self):
        return self.conv_layer(torch.zeros(1, *self.input_shape)).view(1, -1).size(1)

class DQN_large(nn.Module):
    def __init__(self, input_shape, num_actions):
        super(DQN_large, self).__init__()
        self.input_shape = input_shape
        print(input_shape)
        self.num_actions = num_actions
        self.conv_layer = nn.Sequential(
            nn.Conv2d(3, 64, (2, 2)),
            nn.ReLU(),
            nn.MaxPool2d((2, 2),padding=1),
            nn.Conv2d(64, 128, (2, 2)),
            nn.ReLU(),
            nn.Conv2d(128, 256, (2, 2)),
            nn.ReLU(),

        )
        self.fc_layer = nn.Sequential(
            nn.Linear(self.get_conv_output_dim(), 256),
            nn.Tanh(),
            nn.Linear(256, 128),
            nn.Tanh(),
            nn.Linear(128, num_actions)
        )
        print(self.get_conv_output_dim())

    def forward(self, x):
        x = self.conv_layer(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layer(x)
        return x

    def get_conv_output_dim(self):
        return self.conv_layer(torch.zeros(1, *self.input_shape)).view(1, -1).size(1)

class DQN_very_large(nn.Module):
    def __init__(self, input_shape, num_actions):
        super(DQN_very_large, self).__init__()
        self.input_shape = input_shape
        print(input_shape)
        self.num_actions = num_actions
        self.conv_layer = nn.Sequential(
            nn.Conv2d(3, 64, (2, 2)),
            nn.ReLU(),
            nn.MaxPool2d((2, 2),padding=1),
            nn.Conv2d(64, 256, (2, 2)),
            nn.ReLU(),
            nn.Conv2d(256, 512, (2, 2)),
            nn.ReLU(),

        )
        self.fc_layer = nn.Sequential(
            nn.Linear(self.get_conv_output_dim(), 512),
            nn.Tanh(),
            nn.Linear(512, 256),
            nn.Tanh(),
            nn.Linear(256, num_actions)
        )
        print(self.get_conv_output_dim())

    def forward(self, x):
        x = self.conv_layer(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layer(x)
        return x

    def get_conv_output_dim(self):
        return self.conv_layer(torch.zeros(1, *self.input_shape)).view(1, -1).size(1)