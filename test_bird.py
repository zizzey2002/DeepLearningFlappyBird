#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tensorflow as tf
tf.compat.v1.disable_eager_execution()

import cv2
import numpy as np
import sys
sys.path.append("game/")
import wrapped_flappy_bird as game


def preprocess(img):
    img = cv2.cvtColor(cv2.resize(img, (80, 80)), cv2.COLOR_BGR2GRAY)
    _, img = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY)
    return img


def playGame():

    game_state = game.GameState()
    sess = tf.compat.v1.InteractiveSession()

    # ====== 1. 加载训练好的 meta graph ======
    saver = tf.compat.v1.train.import_meta_graph("saved_networks/bird-dqn-220000.meta")

    # ====== 2. 恢复参数 ======
    saver.restore(sess, "saved_networks/bird-dqn-220000")
    print("模型已成功加载！开始测试 🚀")

    graph = tf.compat.v1.get_default_graph()

    # ====== 3. 获取训练时的输入和输出节点 ======
    # 输入 placeholder（名字一般就是 Placeholder）
    s = graph.get_tensor_by_name("Placeholder:0")

    # 输出 Q 值（训练图里通常是 MatMul_1 或 add_3）
    # 如果不确定，也可打印所有节点名
    readout = graph.get_tensor_by_name("MatMul_1:0")

    # ====== 4. 初始化游戏状态 ======
    do_nothing = np.zeros(2)
    do_nothing[0] = 1
    x_t, _, _ = game_state.frame_step(do_nothing)
    x_t = preprocess(x_t)
    s_t = np.stack((x_t, x_t, x_t, x_t), axis=2)

    while True:
        # 让模型预测动作 Q 值
        q = sess.run(readout, feed_dict={s: [s_t]})[0]
        action = np.argmax(q)

        a_t = np.zeros(2)
        a_t[action] = 1

        # 执行动作
        x_t1, r_t, terminal = game_state.frame_step(a_t)
        x_t1 = preprocess(x_t1).reshape(80, 80, 1)

        # 更新状态序列
        s_t = np.append(x_t1, s_t[:, :, :3], axis=2)

        print("Action:", action, "Reward:", r_t, "Q:", q)

        if terminal:
            print("Game Over!")
            break


if __name__ == "__main__":
    playGame()