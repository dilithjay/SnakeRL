import numpy as np
import matplotlib.pyplot as plt
import random


rewards = {"food": 500, "hit": -500, "step": -1, "near": 3}
mov_dir = [(0, 1), (1, 0), (0, -1), (-1, 0)]
p_types = {"empty": 0, "body": 0.33, "head": 0.67, "food": 1}


def get_distance(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


class SnakeEnv:
    def __init__(self, width=60, height=60):
        self.s_body = None
        self.state_size = (width, height)
        self.cur_dir = None
        self.state = np.zeros(self.state_size, float)
        self.food = (0, 0)
        self.invalid_dir_count = 0
        self.count = 0

        self.width = width
        self.height = height

    def step(self, action):
        reward = 0
        done = False
        prev_dist = get_distance(self.s_body[-1], self.food)

        if self.cur_dir != action and self.cur_dir % 2 == action % 2:
            reward -= 10
            action = self.cur_dir
        else:
            self.cur_dir = action
        x, y = [self.s_body[-1][0] + mov_dir[action][0],
                self.s_body[-1][1] + mov_dir[action][1]]
        if x < 0 or y < 0 or x >= self.width or y >= self.height:
            reward = rewards["hit"]
            done = True
        elif x == self.food[0] and y == self.food[1]:
            self.reset_food()
            reward = rewards["food"]
            self.count += 1
            self.s_body += [[x, y]]
        else:
            self.s_body = self.s_body[1:] + [[x, y]]
            if self.s_body[-1] in self.s_body[:-1]:
                reward = rewards["hit"]
                done = True
        if not done:
            self.update_state()
        new_dist = get_distance(self.s_body[-1], self.food)
        if prev_dist > new_dist:
            reward += rewards["near"]
        else:
            reward -= rewards["near"] + 7
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
        x, y = random.randint(1, self.width - 2), random.randint(1, self.height - 2)
        d = random.randint(0, 3)
        self.cur_dir = d
        self.s_body = [[x - mov_dir[d][0], y - mov_dir[d][1]], [x, y]]

    def reset(self):
        self.count = 0
        self.reset_food()
        self.reset_snake()
        self.update_state()
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

    def get_food_count(self):
        return self.count
