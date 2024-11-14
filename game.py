import pygame
import random
from enum import Enum
from collections import namedtuple
import numpy as np
# 初始化游戏
pygame.init()
# 定义字体大小
font = pygame.font.Font(None, 25)
# 定义方向
class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4
# 建一个名为 Point 的命名元组，它有两个属性：x 和 y，用以表示一个点的坐标。
Point = namedtuple('Point', 'x, y')

# 定义颜色
WHITE = (255, 255, 255)
RED = (200,0,0)
BLUE1 = (0, 0, 255)
BLUE2 = (0, 100, 255)
BLACK = (0,0,0)
# 定义方块大小和速度
BLOCK_SIZE = 20
SPEED = 60
# 定义游戏类
class SnakeGameAI:

    def __init__(self, w=640, h=480):
        # 定义宽高
        self.w = w
        self.h = h
        # 初始化显示窗口
        self.display = pygame.display.set_mode((self.w, self.h))
        # 设置标题
        pygame.display.set_caption('AI Snake')
        self.clock = pygame.time.Clock()
        # 重启游戏
        self.reset()


    def reset(self):
        # 初始化方向向右
        self.direction = Direction.RIGHT
        # 初始化蛇头位置
        self.head = Point(self.w/2, self.h/2)
        # 初始化蛇身，包括蛇头和两节身体，最开始蛇有三节身体
        self.snake = [self.head,
                      Point(self.head.x-BLOCK_SIZE, self.head.y),
                      Point(self.head.x-(2*BLOCK_SIZE), self.head.y)]

        self.score = 0
        self.food = None
        # 随机生成食物
        self._place_food()
        # 初始化帧数
        self.frame_iteration = 0


    def _place_food(self):
        # 随机生成食物的位置
        x = random.randint(0, (self.w-BLOCK_SIZE )//BLOCK_SIZE )*BLOCK_SIZE
        y = random.randint(0, (self.h-BLOCK_SIZE )//BLOCK_SIZE )*BLOCK_SIZE
        self.food = Point(x, y)
        # 如果食物生成在蛇身上，重新生成
        if self.food in self.snake:
            self._place_food()


    def play_step(self, action):
        # 增加帧数
        self.frame_iteration += 1
        # 获取用户输入
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
        
        # 根据动作移动蛇头
        self._move(action)
        self.snake.insert(0, self.head)

        reward = 0
        game_over = False
        # 判断游戏是否结束（发生碰撞或者帧数超过100*蛇身长度），帧数即运动次数，防止蛇在一个地方一直打转
        if self.is_collision() or self.frame_iteration > 100*len(self.snake):
            game_over = True
            reward = -10
            return reward, game_over, self.score

        if self.head == self.food: # 吃到食物
            self.score += 1 # 分数加1
            reward = 10 # 奖励10分
            self._place_food() # 重新生成食物
        else: # 没吃到食物，蛇尾去掉
            self.snake.pop()
        
        # 更新界面
        self._update_ui()
        # 设置帧率
        self.clock.tick(SPEED)
        # 返回每一步之后的奖励，游戏是否结束，分数
        return reward, game_over, self.score

    # 判断是否碰撞
    def is_collision(self, pt=None):
        if pt is None:
            pt = self.head
        # 撞到边界
        if pt.x > self.w - BLOCK_SIZE or pt.x < 0 or pt.y > self.h - BLOCK_SIZE or pt.y < 0:
            return True
        # 撞到蛇身
        if pt in self.snake[1:]:
            return True

        return False

    # 更新界面
    def _update_ui(self):
        # 设置背景颜色为黑色
        self.display.fill(BLACK)
        # 画蛇，外层为蓝色，内层为深蓝色
        for pt in self.snake:
            pygame.draw.rect(self.display, BLUE1, pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))
            pygame.draw.rect(self.display, BLUE2, pygame.Rect(pt.x+4, pt.y+4, 12, 12))
        # 画食物，颜色为红色
        pygame.draw.rect(self.display, RED, pygame.Rect(self.food.x, self.food.y, BLOCK_SIZE, BLOCK_SIZE))
        # 标记分数
        text = font.render("Score: " + str(self.score), True, WHITE)
        self.display.blit(text, [0, 0])
        pygame.display.flip()


    def _move(self, action):
        # action列表:[straight, right, left]

        clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        idx = clock_wise.index(self.direction)

        if np.array_equal(action, [1, 0, 0]): # 直走
            new_dir = clock_wise[idx] # 方向不改变
        elif np.array_equal(action, [0, 1, 0]):# 右转
            next_idx = (idx + 1) % 4
            new_dir = clock_wise[next_idx] # 根据顺时针方向改变方向
        else: # 左转
            next_idx = (idx - 1) % 4
            new_dir = clock_wise[next_idx] # 根据逆时针方向改变方向
        # 更新方向
        self.direction = new_dir
        # 更新蛇头位置
        x = self.head.x
        y = self.head.y
        if self.direction == Direction.RIGHT:
            x += BLOCK_SIZE
        elif self.direction == Direction.LEFT:
            x -= BLOCK_SIZE
        elif self.direction == Direction.DOWN:
            y += BLOCK_SIZE
        elif self.direction == Direction.UP:
            y -= BLOCK_SIZE

        self.head = Point(x, y)