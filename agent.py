import torch
import random
import numpy as np
from collections import deque
from game import SnakeGameAI, Direction, Point
from model import Linear_QNet, QTrainer

MAX_MEMORY = 100_000 # 设置最大记忆长度
BATCH_SIZE = 1000 # 设置批量大小
LR = 0.001 # 设置学习率

# 定义Agent类
class Agent:

    def __init__(self):
        # 初始化参数
        self.n_games = 0 # 设置初始游戏次数为0
        self.epsilon = 0 # 初始化 epsilon 参数
        self.gamma = 0.9 # 初始化 gamma 参数
        self.memory = deque(maxlen=MAX_MEMORY) # 创建一个双端队列，并设置其最大长度为MAX_MEMORY
        self.model = Linear_QNet(11, 256, 3) # 创建一个模型
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma) # 创建一个训练器

    # 获取当前的状态
    def get_state(self, game):
        # 获取蛇头的位置
        head = game.snake[0]
        # 蛇头的左边一块坐标
        point_l = Point(head.x - 20, head.y)
        # 蛇头的右边一块坐标
        point_r = Point(head.x + 20, head.y)
        # 蛇头的上边一块坐标
        point_u = Point(head.x, head.y - 20)
        # 蛇头的下边一块坐标
        point_d = Point(head.x, head.y + 20)

        # 判断当前蛇头的方向
        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        # state是一个11维的向量，定义当前蛇的状态
        state = [
            # 直行会碰撞
            (dir_r and game.is_collision(point_r)) or 
            (dir_l and game.is_collision(point_l)) or 
            (dir_u and game.is_collision(point_u)) or 
            (dir_d and game.is_collision(point_d)),

            # 右转会碰撞
            (dir_u and game.is_collision(point_r)) or 
            (dir_d and game.is_collision(point_l)) or 
            (dir_l and game.is_collision(point_u)) or 
            (dir_r and game.is_collision(point_d)),

            # 左转会碰撞
            (dir_d and game.is_collision(point_r)) or 
            (dir_u and game.is_collision(point_l)) or 
            (dir_r and game.is_collision(point_u)) or 
            (dir_l and game.is_collision(point_d)),
            
            # 移动方向
            dir_l,
            dir_r,
            dir_u,
            dir_d,
            
            # 食物相对于蛇头的位置
            game.food.x < game.head.x,  # 食物在蛇头左边
            game.food.x > game.head.x,  # 食物在蛇头右边
            game.food.y < game.head.y,  # 食物在蛇头上边
            game.food.y > game.head.y  # 食物在蛇头下边
            ]

        return np.array(state, dtype=int)

    # 将当前的状态、动作、奖励、下一个状态和是否结束的信息存储到self.memory中
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    # 从 self.memory 中随机抽取一批样本进行训练
    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE: # 如果self.memory中的样本数大于BATCH_SIZE
            mini_sample = random.sample(self.memory, BATCH_SIZE) # 从self.memory中随机抽取BATCH_SIZE个样本
        else:
            mini_sample = self.memory
        # 遍历mini_sample中的每一个样本，对模型进行批量训练
        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)

    # 使用当前的状态、动作、奖励、下一个状态和是否结束的信息进行单步训练
    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    # 根据当前状态获取下一步动作
    def get_action(self, state):
        # epsilon-greedy策略，以epsilon的概率随机选择动作，以1-epsilon的概率选择模型预测的最优动作，平衡探索和利用
        self.epsilon = 80 - self.n_games
        final_move = [0,0,0]
        # 如果随机数小于epsilon，随机选择一个动作，否则选择模型预测的最优动作
        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item() # 获取预测的最优动作
            final_move[move] = 1

        return final_move
