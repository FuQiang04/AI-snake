import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os

# 定义模型
class Linear_QNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x
    # 保存模型
    def save(self, file_name='model.pth'):
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)

        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)

# 定义训练器
class QTrainer:
    def __init__(self, model, lr, gamma):
        self.lr = lr
        self.gamma = gamma
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()

    def train_step(self, state, action, reward, next_state, done):
        state = torch.tensor(np.array(state), dtype=torch.float)
        next_state = torch.tensor(np.array(next_state), dtype=torch.float)
        action = torch.tensor(np.array(action), dtype=torch.long)
        reward = torch.tensor(np.array(reward), dtype=torch.float)

        # 将一维张量转换为二维张量，以便在后续的计算中保持张量的形状一致
        if len(state.shape) == 1:
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            done = (done, )

        # 用模型预测当前状态的Q值
        pred = self.model(state)
        target = pred.clone()
        # 更新Q值
        for idx in range(len(done)):
            # 如果游戏结束，Q值等于奖励值
            Q_new = reward[idx]
            # 如果游戏没有结束，Q值等于奖励值加上下一个状态的最大Q值
            if not done[idx]: # 学习率α=1时的Bellman方程：Q(s, a) = r + gamma * maxQ(s', a')
                Q_new = reward[idx] + self.gamma * torch.max(self.model(next_state[idx]))
            # 更新Q值表
            target[idx][torch.argmax(action[idx]).item()] = Q_new
        # 梯度清零
        self.optimizer.zero_grad()
        # 计算损失
        loss = self.criterion(target, pred)
        # 反向传播
        loss.backward()
        # 更新模型参数
        self.optimizer.step()



