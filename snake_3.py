import numpy as np
import matplotlib.pyplot as plt
import random


rewards = {"food": 500, "hit": -500, "step": -1, "near": 3}
mov_dir = [(0, 1), (1, 0), (0, -1), (-1, 0)]
p_types = {"empty": 0, "body": 0.33, "head": 0.67, "food": 1}


def get_distance(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


class SnakeEnv:
    def __init__(self, start_loc, start_dir, width=60, height=60):
        self.s_body = [start_loc]
        self.state_size = (width, height)
        # 0: Up, 1: Right, 2: Down, 3: Left
        self.cur_dir = start_dir
        self.state = np.zeros(self.state_size, float)
        self.food = [0, 0]

        self.width = width
        self.height = height

        self.state[start_loc[0], start_loc[1]] = p_types["head"]

    def step(self, action):
        s_body = self.s_body
        cur_dir = self.cur_dir
        food = self.food
        reward = 0
        done = False
        prev_dist = get_distance(s_body[-1], food)

        x, y = [s_body[-1][0] + mov_dir[action][0],
                s_body[-1][1] + mov_dir[action][1]]
        if x < 0 or y < 0 or x >= self.width or y >= self.height:
            reward = rewards["hit"]
            done = True
        elif x == food[0] and y == food[1]:
            # print("Food collected")
            self.reset_food()
            reward = rewards["food"]
            self.s_body += [[x, y]]
        else:
            self.s_body = s_body[1:] + [[x, y]]
            if self.s_body[-1] in s_body[:-2]:
                reward = rewards["hit"]
                done = True
        if not done:
            self.update_state()
        new_dist = get_distance(s_body[-1], food)
        if prev_dist > new_dist:
            reward += rewards["near"]
        else:
            reward -= rewards["near"]
        return self.state, reward, done

    def render(self, rate):
        s_body = self.s_body
        food = self.food
        plt.cla()
        plt.xlim(0, self.width)
        plt.ylim(0, self.height)

        # plot head
        plt.scatter(s_body[-1][0], s_body[-1][1], c='#0000ff')

        # plot body
        for x, y in s_body[:-1]:
            plt.scatter(x, y, c='#000000')

        # plot food
        plt.scatter(food[0], food[1], c='#00ff00')

        plt.pause(rate)
        return True

    def reset_food(self):
        self.food = (random.randint(0, self.width - 1), random.randint(0, self.height - 1))

    def reset_snake(self):
        x, y = random.randint(1, self.width - 1), random.randint(1, self.height - 1)
        self.s_body = [[x, y - 1], [x, y]]

    def reset(self):
        self.reset_food()
        self.reset_snake()
        self.update_state()
        self.step(0)
        return self.state

    def update_state(self):
        self.state = np.zeros(self.state_size)
        s_body = self.s_body
        for i in s_body[:-1]:
            self.state[i[0], i[1]] = p_types["body"]
        self.state[s_body[-1][0], s_body[-1][1]] = p_types["head"]
        self.state[self.food[0], self.food[1]] = p_types["food"]

    def get_cur_dir(self):
        return self.cur_dir
