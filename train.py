from matplotlib import pyplot as plt
from game import SnakeGameAI
from agent import Agent

#  绘制分数曲线
def plot_scores(scores):
    plt.clf()
    plt.figure(figsize=(10, 5))
    plt.plot(scores, label='Score Curve')
    plt.xlabel('Game Number')
    plt.ylabel('Score')
    plt.title('Score Curve')
    plt.legend()
    plt.show()

# 训练AI贪吃蛇
def train():
    # 记录每次游戏的分数
    scores=[]
    # 初始化历史最优分数为0
    record = 0
    # 创建Agent对象
    agent = Agent()
    # 创建SnakeGameAI对象
    game = SnakeGameAI()
    while True:
        # 获取当前状态
        state_old = agent.get_state(game)

        # 获取动作
        final_move = agent.get_action(state_old)

        # 执行动作，获取奖励、是否结束、分数
        reward, done, score = game.play_step(final_move)

        # 获取下一个状态
        state_new = agent.get_state(game)

        # 短期记忆训练
        agent.train_short_memory(state_old, final_move, reward, state_new, done)

        # 记录当前状态、动作、奖励、下一个状态和是否结束的信息
        agent.remember(state_old, final_move, reward, state_new, done)

        # 如果游戏结束，重启游戏，训练长期记忆，保存最佳模型
        if done:
            # 重启游戏
            game.reset()
            # 游戏次数加1
            agent.n_games += 1
            # 训练长期记忆
            agent.train_long_memory()

            # 如果当前分数大于历史最优分数，保存模型
            if score > record:
                record = score
                agent.model.save()

            print('Game', agent.n_games, 'Score', score, 'Highest Record:', record)
            # 记录分数
            scores.append(score)
            # 每50次游戏绘制分数曲线
            if agent.n_games % 50 == 0:
                plot_scores(scores)
if __name__ == '__main__':
    train()
