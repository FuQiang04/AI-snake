import torch
import numpy as np
from game import SnakeGameAI, Direction, Point
from model import Linear_QNet

class Agent:
    def __init__(self):
        self.model = Linear_QNet(11, 256, 3) # 创建一个模型
        self.model.load_state_dict(torch.load('model/best_model.pth')) # 加载模型参数
        self.model.eval() # 设置模型为评估模式
        self.gama = 0.9 # 初始化 gamma 参数
        self.epsilon = 0  # 在这里我们不需要epsilon，因为我们不需要探索

    # 获取动作
    def get_action(self, state):
        state0 = torch.tensor(state, dtype=torch.float)
        prediction = self.model(state0)
        move = torch.argmax(prediction).item()
        final_move = [0, 0, 0]
        final_move[move] = 1
        return final_move

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

# 运行AI贪吃蛇
def play():
    agent = Agent()
    game = SnakeGameAI()
    game_number = 0
    while True:
        state = agent.get_state(game)
        final_move = agent.get_action(state)
        reward, is_done, score = game.play_step(final_move)
        if is_done:
            game.reset()
            game_number += 1
            print('Game:',game_number,'Score:',score)

if __name__ == '__main__':
    play()