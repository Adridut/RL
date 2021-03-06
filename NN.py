# Same environment each time
# Neural network used

import random

flatten = lambda l: [item for sublist in l for item in sublist]

length = 8
width = 8
maxSteps = length * width
exitNumber = 1
holeNumber = 0
wallNumber = 0
objective = 15
numberOfEpisodes = 10

from datetime import date, datetime

day = date.today().strftime("%d%m%Y")
time = datetime.now().strftime("%H%M%S")
file_name = 'SNN' + day + time


class Game:
    ACTION_UP = 0
    ACTION_LEFT = 1
    ACTION_DOWN = 2
    ACTION_RIGHT = 3

    ACTIONS = [ACTION_DOWN, ACTION_LEFT, ACTION_RIGHT, ACTION_UP]

    ACTION_NAMES = ["UP", "LEFT ", "DOWN ", "RIGHT"]

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
        for e in cases:
            distance1 = abs((abs(e[0] - end[0][0]) + abs(e[1] - end[0][1])) + (
                        abs(start[0] - e[0]) + abs(start[1] - e[1])) - objective)
            distance2 = abs((abs(key[0] - end[0][0]) + abs(key[1] - end[0][1])) + (
                        abs(start[0] - key[0]) + abs(start[1] - key[1])) - objective)
            if distance1 < distance2:
                key = e

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
        x, y = self.position
        if self.alea:
            return [self._get_grille(x, y) for (x, y) in
                    [self.position, self.end, self.block, self.hole, self.key, self.hasKey]]
        return flatten(self._get_grille(x, y))

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

        delta = abs(objective - self.counter)

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
                elif (i, j) == self.key:
                    str += "+"
                elif (i, j) in self.end:
                    str += "@"
                else:
                    str += "."
            str += "\n"
        return str


import numpy as np
from keras.models import Sequential, load_model
from keras.layers.core import Dense, Activation
from keras.optimizers import Adam, sgd
import random
import time
import os

from collections import deque


class Trainer:
    def __init__(self, learning_rate, epsilon_decay):
        self.state_size = length * width
        self.action_size = 4
        self.gamma = 0.9
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = epsilon_decay
        self.learning_rate = learning_rate

        name = None

        self.name = name

        if name is not None and os.path.isfile("model-" + name):
            model = load_model("model-" + name)
        else:
            model = Sequential()
            #try with 25, state+8?
            model.add(Dense(self.state_size, input_shape=(self.state_size,), activation='relu'))
            model.add(Dense(self.state_size, activation="relu"))
            model.add(Dense(self.action_size, activation='linear'))
            model.compile(loss='mse', optimizer=sgd(lr=self.learning_rate))

        self.model = model

    def get_best_action(self, state, rand=True):

        self.epsilon *= self.epsilon_decay

        if rand and np.random.rand() <= self.epsilon:
            # The agent acts randomly
            return random.randrange(self.action_size)

        # Predict the reward value based on the given state
        act_values = self.model.predict(np.array([state]))

        # Pick the action based on the predicted reward
        action = np.argmax(act_values[0])
        return action

    def train(self, state, action, reward, next_state, done):
        target = self.model.predict(np.array([state]))[0]
        if done:
            target[action] = reward
        else:
            target[action] = reward + self.gamma * np.max(self.model.predict(np.array([next_state])))

        inputs = np.array([state])
        outputs = np.array([target])

        return self.model.fit(inputs, outputs, epochs=1, verbose=0, batch_size=1)

    def save(self):
        if self.name:
            self.model.save("model-" + self.name, overwrite=True)
        else:
            self.model.save("model-" + str(time.time()))

import time

def train(episodes, trainer, game):
    scores = []
    epsilons = []
    delta = []
    losses = [0]
    for e in range(episodes):
        state = game.reset()
        score = 0  # score in current game
        done = False
        steps = 0  # steps in current game
        while not done:
            steps += 1
            action = trainer.get_best_action(state)
            next_state, reward, done, _ = game.move(action)
            score += reward
            trainer.train(state, action, reward, next_state, done)
            #print(state.index(1), Game.ACTION_NAMES[action], reward, next_state.index(1), "DONE" if done else "")
            state = next_state
            if done:
                epsilons.append(trainer.epsilon)
                scores.append(score)
                delta.append(abs(steps - objective))
                break
            if steps > maxSteps:
                trainer.train(state, action, -10, state, True) # we end the game
                epsilons.append(trainer.epsilon)
                scores.append(score)
                delta.append(abs(steps - objective))
                break
        if e % 100 == 0: # print log every 100 episode
            print("episode: {}/{}, moves: {}, score: {}, delta: {}"
                  .format(e, episodes, steps, score, abs(steps - objective)))
            print(f"epsilon : {trainer.epsilon}")
    #trainer.save()
    return scores, epsilons, delta

g = Game(length, width, 0, alea=False)
g.print()
g._get_state()

import pandas as pd

def saveResult(score, numberOfEpisodes, delta, objective, moves, board):
    description = input('Description: ')
    result = {'score': [score], 'numberOfEpisodes': [numberOfEpisodes],
              'delta': [delta], 'objective': [objective], 'moves': [moves],
              'day': [day], 'time': [time], 'description': [description]}
    df = pd.DataFrame(data=result)
    print(df)
    df.to_csv(file_name + '.csv', encoding='utf-8', index=False)
    boardResult = {'board': board}
    df = pd.DataFrame(data=boardResult)
    df.to_csv(file_name + 'board' + '.csv', encoding='utf-8', index=False)

trainer = Trainer(learning_rate=0.01, epsilon_decay=0.99992)
#0.01, 0.9999
score, epsilons, delta = train(numberOfEpisodes, trainer, g)
#2000


import matplotlib.pyplot as plt

score = np.array(score)
score_c = np.convolve(score, np.full((10,), 1/10), mode="same")
d = np.convolve(delta, np.full((10,), 1/10), mode="same")


fig, ax1 = plt.subplots()
ax1.plot(score_c, color='b')
ax2 = ax1.twinx()
ax3 = ax1.twinx()
ax3.plot(d, color='g')
ax3.set_ylabel('Delta', color='g')
ax3.tick_params('y', colors='g')
ax3.spines["right"].set_position(("axes", 0))
ax2.plot(epsilons, color='r')
ax1.set_ylabel('Score', color='b')
ax1.tick_params('y', colors='b')
ax2.set_ylabel('Epsilon', color='r')
ax2.tick_params('y', colors='r')
plt.title("Score,Epsilon and Delta over training")
ax1.set_xlabel("Episodes")
plt.savefig(file_name + '.png')
plt.figure()
plt.show()


from IPython import display
import time

state = g.reset()
state = g._get_state()
done = False
print(g.print())
moves = 0
s = 0
board = []
while not done and moves < maxSteps:
    moves +=1
    time.sleep(1)
    print(trainer.model.predict(np.array([g._get_state()])))
    action = trainer.get_best_action(g._get_state(), rand=False)
    next_state, reward, done, _ = g.move(action)
    s += reward
    delta = abs(objective - moves)
    print(g.print())
    board.append(g.print())
    print("reward : ", reward)
    print("score : ", s)
    print("moves : ", moves)
    print("delta : ", delta)
    print(Game.ACTION_NAMES[action])

saveResult(s, numberOfEpisodes, delta, objective, moves, board)



state = flatten(g._get_grille(0, 1))
# print(state)
trainer.model.predict(np.array([state]))

