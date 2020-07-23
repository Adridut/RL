# Same environment each time
# Q-Learning with table

import random
from math import sqrt

num_episodes = 50
length = 4
width = 4
maxSteps = length * width
exitNumber = 1
holeNumber = 0
wallNumber = 0
# objective = random.randint(1, length * width)
objective = 8

from datetime import date, datetime

from time import sleep
from IPython.display import clear_output
import matplotlib.pyplot as plt

day = date.today().strftime("%d%m%Y")
time = datetime.now().strftime("%H%M%S")
file_name = 'QT' + day + time




class Game:
    ACTION_UP = 0
    ACTION_LEFT = 1
    ACTION_DOWN = 2
    ACTION_RIGHT = 3


    ACTIONS = [ACTION_DOWN, ACTION_LEFT, ACTION_RIGHT, ACTION_UP]

    ACTION_NAMES = ["UP", "LEFT", "DOWN", "RIGHT"]

    MOVEMENTS = {
        ACTION_UP: (1, 0),
        ACTION_RIGHT: (0, 1),
        ACTION_LEFT: (0, -1),
        ACTION_DOWN: (-1, 0)
    }

    num_actions = len(ACTIONS)

    def __init__(self, n, m, wrong_action_p=0, alea=False):
        self.n = n
        self.m = m
        self.wrong_action_p = wrong_action_p
        self.alea = alea
        self.generate_game()

    def _position_to_id(self, x, y):
        """Donne l'identifiant de la position entre 0 et 15"""
        return x + y * self.n

    def _id_to_position(self, id):
        """Réciproque de la fonction précédente"""
        return (id % self.n, id // self.n)

    def generate_game(self):
        cases = [(x, y) for x in range(self.n) for y in range(self.m)]
        hole = []
        i = 0
        while i < holeNumber:
            hole.append(list(random.choice(cases)))
            hole[i] = tuple(hole[i])
            cases.remove(hole[i])
            i += 1
        # hole = random.choice(cases)
        # cases.remove(hole)
        start = random.choice(cases)
        cases.remove(start)
        end = []
        i = 0
        while i < exitNumber:
            end.append(list(random.choice(cases)))
            end[i] = tuple(end[i])
            cases.remove(end[i])
            i += 1

        # end = random.choice(cases)
        # cases.remove(end)
        block = []
        i = 0
        while i < wallNumber:
            block.append(list(random.choice(cases)))
            block[i] = tuple(block[i])
            cases.remove(block[i])
            i += 1
        # block = random.choice(cases)
        # cases.remove(block)

        key = (0,0)
        keyOpt = False
        distanceSE = abs((abs(start[0] - end[0][0]) + abs(start[1] - end[0][1])))
        # while distanceSE > objective:
        #     end.append(list(random.choice(cases)))
        #     end[i] = tuple(end[i])
        #     cases.remove(end[i])
        #     distanceSE = abs((abs(start[0] - end[0][0]) + abs(start[1] - end[0][1])))

        for e in cases:
            distance1 = abs((abs(e[0] - end[0][0]) + abs(e[1] - end[0][1])) + (abs(start[0] - e[0]) + abs(start[1] - e[1])) - objective)
            distance2 = abs((abs(key[0] - end[0][0]) + abs(key[1] - end[0][1])) + (abs(start[0] - key[0]) + abs(start[1] - key[1])) - objective)
            if distance1 < distance2:
                key = e
                if distance1 <= 2 or objective <= distanceSE:
                    keyOpt = True

        self.hasKey = False
        self.key = key
        self.position = start
        self.end = end
        self.hole = hole
        self.block = block


        self.counter = 0

        if not self.alea:
            self.start = start
        return self._get_state()

    def reset(self):
        if not self.alea:
            self.position = self.start
            self.counter = 0
            self.hasKey = False
            return self._get_state()
        else:
            return self.generate_game()

    def _get_grille(self, x, y):
        grille = [
            [0] * self.n for i in range(self.m)
        ]
        grille[x][y] = 1
        return grille

    def _get_state(self):
        if self.alea:
            return [self._get_grille(x, y) for (x, y) in
                    [self.position, self.end, self.block, self.hole, self.key, self.hasKey]]
        return self._position_to_id(*self.position)

    def move(self, action):
        """
        takes an action parameter
        :param action : the id of an action
        :return ((state_id, end, hole, block), reward, is_final, actions)
        """

        self.counter += 1

        if action not in self.ACTIONS:
            raise Exception("Invalid action")

        # random actions sometimes (2 times over 10 default)
        choice = random.random()
        if choice < self.wrong_action_p:
            action = (action + 1) % 4
        elif choice < 2 * self.wrong_action_p:
            action = (action - 1) % 4

        d_x, d_y = self.MOVEMENTS[action]
        x, y = self.position
        new_x, new_y = x + d_x, y + d_y

        oldDistancePK = abs(self.key[0] - x) + abs(self.key[1] - y)
        oldDistancePE = abs(self.end[0][0] - x) + abs(self.end[0][1] - y)
        newDistancePK = abs(self.key[0] - new_x) + abs(self.key[1] - new_y)
        newDistancePE = abs(self.end[0][0] - new_x) + abs(self.end[0][1] - new_y)

        if oldDistancePE <= newDistancePE:
            r1 = -2
        else:
            r1 = 1
        if oldDistancePK <= newDistancePK:
            r2 = -2
        else:
            r2 = 1

        if self.hasKey:
            k = -2
            e = 1
            r = r1
        else:
            k = 1
            e = -2
            r = r2


        if (new_x, new_y) in self.block:
            return self._get_state(), r, False, self.ACTIONS
        elif (new_x, new_y) in self.hole:
            self.position = new_x, new_y
            return self._get_state(), -50, True, None
        elif (new_x, new_y) == self.key:
            self.position = new_x, new_y
            self.hasKey = True
            return self._get_state(), k, False, self.ACTIONS
        elif (new_x, new_y) in self.end:
            self.position = new_x, new_y
            return self._get_state(), e, True, self.ACTIONS
        elif new_x >= self.n or new_y >= self.m or new_x < 0 or new_y < 0:
            return self._get_state(), r, False, self.ACTIONS
        elif self.counter > maxSteps:
            self.position = new_x, new_y
            return self._get_state(), r, True, self.ACTIONS
        else:
            self.position = new_x, new_y
            return self._get_state(), r, False, self.ACTIONS

    def print(self):
        str = ""
        for i in range(self.n - 1, -1, -1):
            for j in range(self.m):
                if (i, j) == self.position:
                    str += "x"
                elif (i, j) in self.block:
                    str += "¤"
                elif (i, j) in self.hole:
                    str += "o"
                elif (i, j) in self.end:
                    str += "@"
                elif (i, j) == self.key:
                    str += "+"
                else:
                    str += "."
            str += "\n"
        return str


## q learning with table
import numpy as np

def train():
    states_n = width * length
    actions_n = 4
    Q = np.zeros([states_n, actions_n])

    # Set learning parameters
    lr = .85
    y = .99
    n = 0
    maxN = 5

    lower_bound = []
    upper_bound = []
    delta_lower_bound = []
    delta_upper_bound = []
    while n < maxN:
        n += 1
        cumul_reward_list = []
        delta = []
        actions_list = []
        states_list = []
        game = Game(length, width, 0)  # 0 chance to go left or right instead of asked direction
        for i in range(num_episodes):
            actions = []
            s = game.reset()
            states = [s]
            cumul_reward = 0
            moves = 0
            d = False
            while True:
                # on choisit une action aléatoire avec une certaine probabilité, qui décroit
                Q2 = Q[s, :] + np.random.randn(1, actions_n) * (1. / (i + 1))
                a = np.argmax(Q2)
                s1, reward, d, _ = game.move(a)
                Q[s, a] = Q[s, a] + lr * (reward + y * np.max(Q[s1, :]) - Q[s, a])  # Fonction de mise à jour de la Q-table
                cumul_reward += reward
                s = s1
                actions.append(a)
                states.append(s)
                moves += 1
                if d == True:
                    break
            states_list.append(states)
            actions_list.append(actions)
            cumul_reward_list.append(cumul_reward)
            delta.append(abs(moves - objective))

        if n == 1:
            upper_bound = cumul_reward_list.copy()
            lower_bound = cumul_reward_list.copy()
            delta_lower_bound = delta.copy()
            delta_upper_bound = delta.copy()
        else:
            i = 0
            while i < len(cumul_reward_list):
                if cumul_reward_list[i] > upper_bound[i]:
                    upper_bound[i] = cumul_reward_list[i]
                if cumul_reward_list[i] < lower_bound[i]:
                    lower_bound[i] = cumul_reward_list[i]
                if delta[i] > delta_upper_bound[i]:
                    delta_upper_bound[i] = delta[i]
                if delta[i] < delta_lower_bound[i]:
                    delta_lower_bound[i] = delta[i]
                i += 1

    game.reset()
    game.print()
    graph(upper_bound, lower_bound, delta_upper_bound, delta_lower_bound)
    d = False
    g = game
    s = g.reset()
    print(g.print())
    print("reward : ", 0)
    print("score : ", 0)
    score = 0
    sleep(2)
    moves = 0
    board = []
    while not d:
        moves += 1
        g._get_state()
        a = np.argmax(Q[s, :])
        s, r, d, _ = g.move(a)
        score += r
        delta = abs(objective - moves)
        clear_output(wait=True)
        print(g.print())
        board.append(g.print())
        print("reward : ", r)
        print("score : ", score)
        print("moves : ", moves)
        print("delta : ", delta)
        print(Game.ACTION_NAMES[a])
        sleep(0.5)

    saveResult(s, num_episodes, delta, objective, moves, plt, board)


# print("Score over time: " + str(sum(cumul_reward_list[-100:]) / 100.0))



import pandas as pd


def saveResult(score, numberOfEpisodes, delta, objective, moves, plt, board):

    description = input('Description: ')

    result = {'score': [score], 'numberOfEpisodes': [numberOfEpisodes],
              'delta': [delta], 'objective': [objective], 'moves': [moves],
              'day': [day], 'time': [time], 'description': [description]}
    df = pd.DataFrame(data=result)
    print(df)
    df.to_csv(file_name + '.csv', encoding='utf-8', index=False)
    boardResult = {'board': board}
    df = pd.DataFrame(data=boardResult)
    df.to_csv(file_name + 'board' +  '.csv', encoding='utf-8', index=False)


# t = Trainer(filepath="model-1496937952")

def graph(upper_bound, lower_bound, delta_upper_bound, delta_lower_bound):
    middle_list = []
    delta_middle_list = []
    i = 0
    while i < len(upper_bound):
        middle_list.append((upper_bound[i] + lower_bound[i]) / 2)
        delta_middle_list.append((delta_upper_bound[i] + delta_lower_bound[i]) / 2)
        i += 1
    x = range(0,num_episodes)

    i = 0
    scorePerf, deltaPerf, scorePerf_lower_bound, scorePerf_upper_bound,\
    deltaPerf_lower_bound, deltaPerf_upper_bound = 0, 0, 0, 0, 0, 0
    while i < 10:
        i += 1
        scorePerf += middle_list[len(middle_list) - i]
        deltaPerf += delta_middle_list[len(delta_middle_list) - i]
        scorePerf_lower_bound += lower_bound[len(lower_bound) - i]
        scorePerf_upper_bound += upper_bound[len(upper_bound) - i]
        deltaPerf_lower_bound += delta_lower_bound[len(delta_lower_bound) - i]
        deltaPerf_upper_bound += delta_upper_bound[len(delta_upper_bound) - i]
    scorePerf = scorePerf / 10
    deltaPerf = deltaPerf / 10
    scorePerf_lower_bound = scorePerf_lower_bound / 10
    scorePerf_upper_bound = scorePerf_upper_bound / 10
    deltaPerf_lower_bound = deltaPerf_lower_bound / 10
    deltaPerf_upper_bound = deltaPerf_upper_bound / 10

    fig, ax1 = plt.subplots()

    ax2 = ax1.twinx()
    ax1.bar(-1, scorePerf, color='b')
    ax1.set_ylabel('Score', color='b')
    ax1.tick_params('y', colors='b')
    ax2.bar(1, deltaPerf, color='g')
    ax2.set_ylabel('Delta', color='g')
    ax2.tick_params('y', colors='g')
    plt.title("Perfomance Histogram")
    plt.show()

    fig, ax1 = plt.subplots()
    # ax1.plot(cumul_reward_list[:num_episodes], color='b')
    ax1.plot(middle_list[:num_episodes], color='b')
    ax1.fill_between(x, upper_bound, lower_bound, alpha=0.1, color='b')
    ax1.set_ylabel('Score', color='b')
    ax1.tick_params('y', colors='b')
    ax2 = ax1.twinx()
    # ax2.plot(delta, color='g')
    ax2.plot(delta_middle_list[:num_episodes], color='g')
    ax2.fill_between(x, delta_upper_bound, delta_lower_bound, alpha=0.1, color='g')
    ax2.set_ylabel('Delta', color='g')
    ax2.tick_params('y', colors='g')
    plt.title("Score, and Delta over training")
    ax1.set_xlabel("Episodes")
    plt.savefig(file_name + '.png')
    plt.figure()
    plt.show()

train()





