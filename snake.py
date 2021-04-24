import numpy as np
import matplotlib.pyplot as plt
import random


rewards = {"food": 200, "hit": -1000, "step": -1}
mov_dir = [(0, 1), (1, 0), (0, -1), (-1, 0)]


class SnakeEnv:
    def __init__(self, start_loc, start_dir, width=60, height=60):
        self.s_body = [start_loc]

        # 0: Up, 1: Right, 2: Down, 3: Left
        self.cur_dir = start_dir
        self.state = np.zeros((width, height), int)
        self.food = [0, 0]

        self.width = width
        self.height = height

        # pixel types: empty = 0, snake body = 1, snake head = 2, food = 3
        self.state[start_loc[0], start_loc[1]] = 2

    def step(self, action):
        s_body = self.s_body
        cur_dir = self.cur_dir
        food = self.food
        reward = 0
        done = False

        if cur_dir == (action + 2) % 4:
            action = cur_dir
        x, y = [s_body[-1][0] + mov_dir[action][0],
                s_body[-1][1] + mov_dir[action][1]]
        if x < 0 or y < 0 or x >= self.width or y >= self.height:
            reward = rewards["hit"]
            done = True
        if len(s_body) > 1:
            s_body = s_body[1:] + [[x, y]]
            if s_body[-1] in s_body[:-2]:
                reward = rewards["hit"]
                done = True
        else:
            s_body[0] = [x, y]
        if x == food[0] and y == food[1]:
            print("Food collected")
            new_point = [s_body[0][0] - mov_dir[action][0], s_body[0][1] - mov_dir[action][1]]
            s_body = [new_point] + s_body
            self.reset_food()
            reward = rewards["food"]
        if not done:
            self.update_state()
        return self.state, reward, done

    def render(self, rate):
        s_body = self.s_body
        food = self.food
        plt.cla()
        plt.xlim(0, self.width)
        plt.ylim(0, self.height)

        # plot body
        for x, y in s_body:
            plt.scatter(x, y, c='#000000')

        # plot food
        plt.scatter(food[0], food[1], c='#00ff00')

        plt.pause(rate)
        return True

    def reset_food(self):
        self.food = (random.randint(0, self.width - 1), random.randint(0, self.height - 1))

    def reset_snake(self):
        self.s_body = [[random.randint(0, self.width - 1), random.randint(0, self.height - 1)]]

    def update_state(self):
        self.state = np.zeros((self.width, self.height))
        s_body = self.s_body
        for i in s_body[:-1]:
            self.state[i[0], i[1]] = 1
        self.state[s_body[-1][0], s_body[-1][1]] = 2
        self.state[self.food[0], self.food[1]] = 3

    def reset(self):
        self.reset_food()
        self.reset_snake()
        self.update_state()
        return self.state
