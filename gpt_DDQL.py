#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Flappy Bird Double DQN 2025 现代修复版

特点：
- TensorFlow 1.x 风格（tf.compat.v1）
- 使用 Double DQN（主网选动作 + 目标网估值）
- 使用 target network 提升稳定性
- 简单 reward shaping：存活+微弱奖励，过管子+1，死亡-1
"""

from __future__ import print_function
import tensorflow as tf
tf.compat.v1.disable_eager_execution()

import cv2
import sys
import random
import numpy as np
from collections import deque

sys.path.append("game/")
import wrapped_flappy_bird as game

# ==================== 超参数 ====================
GAME = 'bird'
ACTIONS = 2                 # 动作数：不跳 / 跳
GAMMA = 0.99                # 折扣因子
OBSERVE = 1000              # 纯观察步数（只收集经验，不训练）
EXPLORE = 300000            # epsilon 从 INITIAL 衰减到 FINAL 所花的步数
INITIAL_EPSILON = 0.5
FINAL_EPSILON = 0.05
REPLAY_MEMORY = 100000      # 经验池容量
BATCH = 32
FRAME_PER_ACTION = 4
LEARNING_RATE = 2.5e-4      # 稍微调高一点，Adam 比较稳
TARGET_UPDATE_FREQ = 1000   # target 网络同步频率
SAVE_INTERVAL = 20000       # 保存模型步数间隔

# ==================== 网络权重辅助函数 ====================
def weight_variable(shape):
    initial = tf.compat.v1.truncated_normal(shape, stddev=0.01)
    return tf.compat.v1.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.01, shape=shape)
    return tf.compat.v1.Variable(initial)

def conv2d(x, W, stride):
    return tf.nn.conv2d(x, W, strides=[1, stride, stride, 1], padding="SAME")

def max_pool_2x2(x):
    return tf.nn.max_pool(
        x,
        ksize=[1, 2, 2, 1],
        strides=[1, 2, 2, 1],
        padding="SAME"
    )

# ==================== 网络结构（主网络 / 目标网络共用） ====================
def createNetwork(name='q_network'):
    with tf.compat.v1.variable_scope(name):
        # 卷积层
        W_conv1 = weight_variable([8, 8, 4, 32])
        b_conv1 = bias_variable([32])

        W_conv2 = weight_variable([4, 4, 32, 64])
        b_conv2 = bias_variable([64])

        W_conv3 = weight_variable([3, 3, 64, 64])
        b_conv3 = bias_variable([64])

        # 全连接层
        W_fc1 = weight_variable([1600, 512])
        b_fc1 = bias_variable([512])

        W_fc2 = weight_variable([512, ACTIONS])
        b_fc2 = bias_variable([ACTIONS])

        # 输入：80x80，4 帧堆叠
        s = tf.compat.v1.placeholder("float", [None, 80, 80, 4])

        h_conv1 = tf.nn.relu(conv2d(s, W_conv1, 4) + b_conv1)
        h_pool1 = max_pool_2x2(h_conv1)

        h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2, 2) + b_conv2)
        h_conv3 = tf.nn.relu(conv2d(h_conv2, W_conv3, 1) + b_conv3)

        h_conv3_flat = tf.reshape(h_conv3, [-1, 1600])
        h_fc1 = tf.nn.relu(tf.matmul(h_conv3_flat, W_fc1) + b_fc1)

        # 输出：每个动作的 Q 值
        readout = tf.matmul(h_fc1, W_fc2) + b_fc2

    return s, readout, h_fc1

# ==================== 主训练函数（Double DQN） ====================
def trainNetwork(s, readout, h_fc1, sess):
    # ----- 1. 创建目标网络 -----
    s_target, readout_target, h_fc1_target = createNetwork(name='target_network')

    # 动作 one-hot & Q 目标值
    a = tf.compat.v1.placeholder("float", [None, ACTIONS])
    y = tf.compat.v1.placeholder("float", [None])

    # 选中当前动作对应的 Q(s,a)
    readout_action = tf.reduce_sum(tf.multiply(readout, a), axis=1)

    # 使用 Huber loss 稍微稳一点（可选），也可以继续用 MSE
    # cost = tf.reduce_mean(tf.square(y - readout_action))
    cost = tf.compat.v1.losses.huber_loss(y, readout_action)

    optimizer = tf.compat.v1.train.AdamOptimizer(LEARNING_RATE)
    train_step = optimizer.minimize(cost)

    # ----- 2. 构建 target network 参数同步操作 -----
    # 现在我们用 variable_scope，所以可以直接按 scope 拿变量（更安全）
    main_vars   = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES,
                                              scope='q_network')
    target_vars = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES,
                                              scope='target_network')

    copy_target_op = [target_vars[i].assign(main_vars[i]) for i in range(len(main_vars))]

    # ----- 3. 初始化游戏环境和经验池 -----
    game_state = game.GameState()
    D = deque(maxlen=REPLAY_MEMORY)

    # 获得初始状态：先执行一次“do nothing”
    do_nothing = np.zeros(ACTIONS)
    do_nothing[0] = 1
    x_t, r_0, terminal = game_state.frame_step(do_nothing)

    x_t = cv2.cvtColor(cv2.resize(x_t, (80, 80)), cv2.COLOR_BGR2GRAY)
    _, x_t = cv2.threshold(x_t, 128, 255, cv2.THRESH_BINARY)  # 阈值用 128，更稳定
    s_t = np.stack((x_t, x_t, x_t, x_t), axis=2)              # 堆叠成 4 帧

    # Saver
    saver = tf.compat.v1.train.Saver()

    # # ==== 加载已有模型继续训练（可选）====
    # checkpoint_path = "saved_networks/bird-dqn-60000"
    # saver.restore(sess, checkpoint_path)
    # print("成功从 60000 步的模型继续训练！")

    # ----- 4. 初始化变量并同步一次 target 网络 -----
    sess.run(tf.compat.v1.global_variables_initializer())
    sess.run(copy_target_op)
    print("开始训练！小鸟要起飞了🚀（Double DQN + target network 已初始化）")

    epsilon = INITIAL_EPSILON
    t = 0

    # ==================== 训练主循环 ====================
    while True:
        # 1) 使用 ε-greedy 选择动作（用主网络）
        readout_t = sess.run(readout, feed_dict={s: [s_t]})[0]
        a_t = np.zeros([ACTIONS])
        action_index = 0

        if t % FRAME_PER_ACTION == 0:
            if random.random() <= epsilon:
                # 降低随机跳跃概率（例如 10% 随机跳，90% 随机不跳）
                jump_random_prob = 0.10
                if random.random() < jump_random_prob:
                    action_index = 1   # 随机跳
                else:
                    action_index = 0   # 随机不跳
                a_t[action_index] = 1
            else:
                action_index = np.argmax(readout_t)
                a_t[action_index] = 1
        else:
            # 不在动作帧就不跳
            a_t[0] = 1

        # ε 线性退火
        if t > OBSERVE and epsilon > FINAL_EPSILON:
            epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / float(EXPLORE)
            epsilon = max(FINAL_EPSILON, epsilon)

        # 2) 执行动作，获得新帧、原始 reward、是否死亡
        x_t1_colored, r_t, terminal = game_state.frame_step(a_t)

        # 奖励整形
        if terminal:
            r_t = -1.0              # 死亡惩罚
        elif r_t == 1:              # 只有过管子才是 1
            r_t = 1.0
        else:
            r_t = 0.0005            # 活着

        # 3) 预处理下一帧
        x_t1 = cv2.cvtColor(cv2.resize(x_t1_colored, (80, 80)), cv2.COLOR_BGR2GRAY)
        _, x_t1 = cv2.threshold(x_t1, 128, 255, cv2.THRESH_BINARY)
        x_t1 = np.reshape(x_t1, (80, 80, 1))
        s_t1 = np.append(x_t1, s_t[:, :, :3], axis=2)   # 新帧加到最前面

        # 4) 存入经验池
        D.append((s_t, a_t, r_t, s_t1, terminal))
        s_t = s_t1
        t += 1

        # 5) 从经验池采样训练（超过 OBSERVE 才开始）
        if t > OBSERVE:
            minibatch = random.sample(D, BATCH)

            s_j_batch   = [d[0] for d in minibatch]
            a_batch     = [d[1] for d in minibatch]
            r_batch     = [d[2] for d in minibatch]
            s_j1_batch  = [d[3] for d in minibatch]
            terminal_batch = [d[4] for d in minibatch]

            # ===== Double DQN 核心部分 =====
            # ① 用主网络在 s' 上选动作（argmax）
            q_next_main = sess.run(readout, feed_dict={s: s_j1_batch})
            # ② 用目标网络在 s' 上评估这些动作的 Q 值
            q_next_target = sess.run(readout_target, feed_dict={s_target: s_j1_batch})

            y_batch = []
            for i in range(len(minibatch)):
                if terminal_batch[i]:
                    # 终止状态：没有未来回报
                    y_batch.append(r_batch[i])
                else:
                    # 主网络选 a_max
                    a_max = np.argmax(q_next_main[i])
                    # 目标网络给出该动作价值
                    target_q = q_next_target[i][a_max]
                    y_batch.append(r_batch[i] + GAMMA * target_q)

            # 梯度更新一步（只更新主网络参数）
            sess.run(train_step, feed_dict={
                y:  y_batch,
                a:  a_batch,
                s:  s_j_batch
            })

        # 6) 定期同步 target 网络（复制主网络参数）
        if t % TARGET_UPDATE_FREQ == 0:
            sess.run(copy_target_op)
            print("Target network updated at step", t)

        # 7) 定期保存模型
        if t % SAVE_INTERVAL == 0:
            saver.save(sess, 'saved_networks/' + GAME + '-double-dqn', global_step=t)
            print("第 {} 步 - 模型已保存！当前 ε = {:.3f}".format(t, epsilon))

        # 8) 打印训练状态
        if t <= OBSERVE:
            state = "observe"
        elif t <= OBSERVE + EXPLORE:
            state = "explore"
        else:
            state = "train"

        print("Step {} | {} | ε = {:.3f} | Action = {} | Reward = {:.4f}".format(
            t, state, epsilon, action_index, r_t
        ))

# ==================== 入口函数 ====================
def playGame():
    sess = tf.compat.v1.InteractiveSession()
    # 主网络用专门的 scope，方便和 target 网络区分
    s, readout, h_fc1 = createNetwork(name='q_network')
    trainNetwork(s, readout, h_fc1, sess)

if __name__ == "__main__":
    playGame()