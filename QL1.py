# Same environment each time
# Q-Learning with table

import random
from math import sqrt

length = 8
width = 8
maxSteps = length * width
exitNumber = 2
holeNumber = 2
wallNumber = 1



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
                    [self.position, self.end, self.block, self.hole]]
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

        if self.counter >= maxSteps:
            r = -40
        else:
            r = 1

        if (new_x, new_y) in self.block:
            return self._get_state(), r, False, self.ACTIONS
        elif (new_x, new_y) in self.hole:
            self.position = new_x, new_y
            return self._get_state(), -30, True, None
        elif (new_x, new_y) in self.end:
            self.position = new_x, new_y
            return self._get_state(), r, True, self.ACTIONS
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
                else:
                    str += "."
            str += "\n"
        print(str)


## q learning with table
import numpy as np

states_n = width * length
actions_n = 4
Q = np.zeros([states_n, actions_n])

# Set learning parameters
lr = .85
y = .99
num_episodes = 200
cumul_reward_list = []
actions_list = []
states_list = []
game = Game(length, width, 0)  # 0 chance to go left or right instead of asked direction
for i in range(num_episodes):
    actions = []
    s = game.reset()
    states = [s]
    cumul_reward = 0
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
        if d == True:
            break
    states_list.append(states)
    actions_list.append(actions)
    cumul_reward_list.append(cumul_reward)

print("Score over time: " + str(sum(cumul_reward_list[-100:]) / 100.0))

game.reset()
game.print()

import matplotlib.pyplot as plt

plt.plot(cumul_reward_list[:num_episodes])
plt.ylabel('Cumulative reward')
plt.xlabel('Episodes')
plt.show()

# t = Trainer(filepath="model-1496937952")

from time import sleep
from IPython.display import clear_output

d = False
g = game
s = g.reset()
g.print()
print("reward : ", 0)
print("score : ", 0)
score = 0
sleep(2)
moves = 0
while not d:
    moves += 1
    g._get_state()
    a = np.argmax(Q[s, :])
    s, r, d, _ = g.move(a)
    score += r
    clear_output(wait=True)
    g.print()
    print("reward : ", r)
    print("score : ", score)
    print("moves : ", moves)
    print(Game.ACTION_NAMES[a])
    sleep(0.5)
