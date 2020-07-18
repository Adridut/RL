import random

# Random environment each time
# Neural network used

# TODO Make it work for any numbers of exits, walls and holes
# TODO Make epsilon proportional to number of episodes
# TODO merge the 3 algo in 1
# TODO Save results in the same csv file for each algo
# TODO Make it possible for the user to choose the goal -
# - and the start and end position and the end of the training

flatten = lambda l: [item for sublist in l for item in sublist]
length = 4
width = 4
maxSteps = (width * length) - 1
exitNumber = 1
holeNumber = 0
wallNumber = 0
numberOfEpisodes = 200000
elementInState = 4

from datetime import date, datetime


class Game:
    ACTION_UP = 0
    ACTION_LEFT = 1
    ACTION_DOWN = 2
    ACTION_RIGHT = 3
    ACTION_STAY = 4

    ACTIONS = [ACTION_DOWN, ACTION_LEFT, ACTION_RIGHT, ACTION_UP, ACTION_STAY]

    ACTION_NAMES = ["UP", "LEFT ", "DOWN ", "RIGHT", "STAY"]

    MOVEMENTS = {
        ACTION_UP: (1, 0),
        ACTION_RIGHT: (0, 1),
        ACTION_LEFT: (0, -1),
        ACTION_DOWN: (-1, 0),
        ACTION_STAY: (0, 0)
    }

    num_actions = len(ACTIONS)

    # wrong = 0.1
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

        self.position = start
        self.end = end
        self.hole = hole
        self.block = block
        self.delta = (0, 0)
        distanceStartEnd = abs(start[0] - end[0][0]) + abs(start[1] - end[0][1])
        self.goal = random.randint(distanceStartEnd, maxSteps)
        i = 0
        a = 0
        b = 0
        while i < self.goal:
            i += 1
            a += 1
            if a == length:
                a = 0
                b += 1

        self.goalT = (a, b)

        self.counter = 0
        self.deltaI = 0
        self.deltaJ = 0

        if not self.alea:
            self.start = start
        return self._get_state()

    def reset(self):
        if not self.alea:
            self.position = self.start
            self.counter = 0
            self.deltaI = 0
            self.deltaJ = 0
            self.delta = (0, 0)
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
            return np.reshape([self._get_grille(x, y) for [(x, y)] in
                               [[self.position], self.end, [self.delta], [self.goalT]]],
                              (1, width * length * elementInState))
        return flatten(self._get_grille(x, y))

    def get_random_action(self):
        return random.choice(self.ACTIONS)

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
            action = (action + 1) % 5
        elif choice < 2 * self.wrong_action_p:
            action = (action - 1) % 5

        d_x, d_y = self.MOVEMENTS[action]
        x, y = self.position
        new_x, new_y = x + d_x, y + d_y

        if self.counter == self.goal:
            self.delta = (width - 1, length - 1)
        else:
            self.delta = (self.deltaJ, self.deltaI)

        self.deltaI += 1

        if self.deltaI >= length:
            self.deltaI = 0
            self.deltaJ += 1

        if self.counter <= self.goal:
            r = 10 / self.goal
        else:
            r = -1

        if (new_x, new_y) in self.block:
            return self._get_state(), r, False, self.goal, self.ACTIONS
        elif (new_x, new_y) in self.hole:
            self.position = new_x, new_y
            return self._get_state(), -30, True, self.goal, None
        elif (new_x, new_y) in self.end:
            self.position = new_x, new_y
            return self._get_state(), r, True, self.goal, self.ACTIONS
        elif new_x >= self.n or new_y >= self.m or new_x < 0 or new_y < 0:
            return self._get_state(), r, False, self.goal, self.ACTIONS
        elif self.counter > maxSteps:
            self.position = new_x, new_y
            return self._get_state(), r, True, self.goal, self.ACTIONS
        else:
            self.position = new_x, new_y
            return self._get_state(), r, False, self.goal, self.ACTIONS

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
        return str


# defining the neural network
import numpy as np
from keras.models import Sequential, load_model
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import RMSprop, Adam, sgd
from keras.layers.advanced_activations import LeakyReLU
import random
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from collections import deque


class Trainer:
    def __init__(self, name=None, learning_rate=0.001, epsilon_decay=0.9999, batch_size=30, memory_size=3000):
        self.action_size = 5
        self.state_size = width * length * elementInState
        self.gamma = 0.9
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = epsilon_decay
        self.learning_rate = learning_rate
        self.memory = deque(maxlen=memory_size)
        self.batch_size = batch_size

        self.name = name
        if name is not None and os.path.isfile("model-" + name):
            model = load_model("model-" + name)
        else:
            model = Sequential()
            model.add(Dense(self.state_size, input_dim=self.state_size, activation='relu'))
            model.add(Dense(self.state_size * 2, activation='relu'))
            model.add(Dense(self.state_size * 2, activation='relu'))
            model.add(Dense(self.action_size, activation='linear'))
            model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))

        self.model = model

    def decay_epsilon(self):
        self.epsilon *= self.epsilon_decay

    def get_best_action(self, state, rand=True):

        if rand and np.random.rand() <= self.epsilon:
            # The agent acts randomly
            return random.randrange(self.action_size)

        # Predict the reward value based on the given state
        act_values = self.model.predict(np.array(state))

        # Pick the action based on the predicted reward
        action = np.argmax(act_values[0])
        return action

    def remember(self, state, action, reward, next_state, done):
        self.memory.append([state, action, reward, next_state, done])

    def replay(self, batch_size):
        batch_size = min(batch_size, len(self.memory))

        minibatch = random.sample(self.memory, batch_size)

        inputs = np.zeros((batch_size, self.state_size))
        outputs = np.zeros((batch_size, self.action_size))

        for i, (state, action, reward, next_state, done) in enumerate(minibatch):
            target = self.model.predict(state)[0]
            if done:
                target[action] = reward
            else:
                target[action] = reward + self.gamma * np.max(self.model.predict(next_state))

            inputs[i] = state
            outputs[i] = target

        return self.model.fit(inputs, outputs, epochs=1, verbose=0, batch_size=batch_size)

    def save(self, id=None, overwrite=False):
        name = 'model'
        if self.name:
            name += '-' + self.name
        else:
            name += '-' + str(time.time())
        if id:
            name += '-' + id
        self.model.save(name, overwrite=overwrite)


def smooth(vector, width=30):
    return np.convolve(vector, [1 / width] * width, mode='valid')


import time
from IPython.core.debugger import set_trace


def train(episodes, trainer, wrong_action_p, alea, collecting=False, snapshot=5000):
    batch_size = 32
    lower_bound = []
    upper_bound = []
    delta_lower_bound = []
    delta_upper_bound = []
    n = 0
    while n < 5:
        trainer.__init__()
        trainer = Trainer(learning_rate=0.001, epsilon_decay=(0.999995))
        n += 1
        g = Game(length, width, wrong_action_p, alea=alea)
        counter = 1
        scores = []
        global_counter = 0
        losses = [0]
        epsilons = []
        delta = []

        # we start with a sequence to collect information, without learning
        if collecting:
            collecting_steps = numberOfEpisodes / 3
            print("Collecting game without learning")
            steps = 0
            while steps < collecting_steps:
                state = g.reset()
                done = False
                while not done:
                    steps += 1
                    action = g.get_random_action()
                    next_state, reward, done, _ = g.move(action)
                    trainer.remember(state, action, reward, next_state, done)
                    state = next_state

        print("Starting training")
        global_counter = 0
        for e in range(episodes + 1):
            state = g.generate_game()
            state = np.reshape(state, [1, width * length * elementInState])
            score = 0
            done = False
            steps = 0
            while not done:
                steps += 1
                global_counter += 1
                action = trainer.get_best_action(state)
                trainer.decay_epsilon()
                next_state, reward, done, goal, _ = g.move(action)
                next_state = np.reshape(next_state, [1, width * length * elementInState])
                score += reward
                trainer.remember(state, action, reward, next_state, done)
                state = next_state
                if global_counter % 100 == 0:
                    l = trainer.replay(batch_size)
                    losses.append(l.history['loss'][0])
                if done:
                    epsilons.append(trainer.epsilon)
                    scores.append(score)
                    delta.append(abs(steps - goal))
                    break
                if steps > maxSteps:
                    epsilons.append(trainer.epsilon)
                    scores.append(score)
                    delta.append(abs(steps - goal))
                    break
            if e % 200 == 0:
                print("episode: {}/{}, moves: {}, score: {}, delta: {}, goal: {}, epsilon: {}, loss: {}"
                      .format(e, episodes, steps, score, abs(steps - goal), goal, trainer.epsilon, losses[-1]))
            # if e > 0 and e % snapshot == 0:
            #     trainer.save(id='iteration-%s' % e)

        sc = smooth(scores, width=round(numberOfEpisodes / 70) + 1)
        d = smooth(delta, width=round(numberOfEpisodes / 70) + 1)

        if n == 1:
            upper_bound = sc.copy()
            lower_bound = sc.copy()
            delta_upper_bound = d.copy()
            delta_lower_bound = d.copy()
        else:
            i = 0
            while i < len(sc):
                if sc[i] > upper_bound[i]:
                    upper_bound[i] = sc[i]
                if sc[i] < lower_bound[i]:
                    lower_bound[i] = sc[i]
                if d[i] > delta_upper_bound[i]:
                    delta_upper_bound[i] = d[i]
                if d[i] < delta_lower_bound[i]:
                    delta_lower_bound[i] = d[i]
                i += 1
        if n == 5:
            return scores, losses, epsilons, delta, upper_bound, lower_bound, delta_upper_bound, delta_lower_bound


import pandas as pd


def saveResult(score, numberOfEpisodes, moves, board, goal, delta):
    description = input('Description: ')
    result = {'score': [score], 'numberOfEpisodes': [numberOfEpisodes], 'moves': [moves], 'goal': [goal],
              'delta': [delta], 'day': [day], 'time': [hour], 'description': [description]}
    df = pd.DataFrame(data=result)
    print(df)
    df.to_csv(file_name + '.csv', encoding='utf-8', index=False)
    boardResult = {'board': board}
    df = pd.DataFrame(data=boardResult)
    df.to_csv(file_name + 'board' + '.csv', encoding='utf-8', index=False)


trainer = Trainer(learning_rate=0.001, epsilon_decay=(0.999995))
# 0.999995
scores, losses, epsilons, delta, upper_bound, lower_bound, delta_upper_bound, delta_lower_bound = train(numberOfEpisodes, trainer, 0, True, snapshot=2500)
# 35000

import matplotlib.pyplot as plt

sc = smooth(scores, width=round(numberOfEpisodes / 70) + 1)
d = smooth(delta, width=round(numberOfEpisodes / 70) + 1)
ub = smooth(upper_bound, width=round(numberOfEpisodes / 70) + 1)
lb = smooth(lower_bound, width=round(numberOfEpisodes / 70) + 1)
dub = smooth(delta_upper_bound, width=round(numberOfEpisodes / 70) + 1)
dlb = smooth(delta_lower_bound, width=round(numberOfEpisodes / 70) + 1)

middle_list = []
delta_middle_list = []
i = 0
while i < len(ub):
    delta_middle_list.append((dub[i] + dlb[i]) / 2)
    middle_list.append((ub[i] + lb[i]) / 2)
    i += 1
# x = range(0,numberOfEpisodes)
x = range(0, len(ub))

# score = np.array(scores)
# score_c = np.convolve(score, np.full((10,), 1/10), mode="same")

day = date.today().strftime("%d%m%Y")
hour = datetime.now().strftime("%H%M%S")
file_name = 'DNN' + day + hour

fig, ax1 = plt.subplots()
# ax1.plot(sc, color='b')
ax1.plot(middle_list, color='b')
ax1.fill_between(x, ub, lb, alpha=0.1, color='b')
ax2 = ax1.twinx()
ax2.plot(epsilons, color='r')
ax3 = ax1.twinx()
# ax3.plot(d, color='g')
ax3.plot(delta_middle_list, color='g')
ax3.fill_between(x, dub, dlb, alpha=0.1, color='g')
ax3.set_ylabel('Delta', color='g')
ax3.tick_params('y', colors='g')
ax3.spines["right"].set_position(("axes", 0))
ax1.set_ylabel('Score', color='b')
ax1.tick_params('y', colors='b')
ax2.set_ylabel('Epsilon', color='r')
ax2.tick_params('y', colors='r')
plt.title("Score, Epsilon and Delta over training")
ax1.set_xlabel("Episodes")
# ax4 = ax1.twinx()
# ax4.plot(sc2, color='y')
plt.savefig(file_name + '.png')
plt.figure()
plt.show()

from IPython import display
import time

# 0.1
g = Game(length, width, 0, alea=True)

state = g.reset()
state = g._get_state()
print("state")
print("  ")
print(g.print())
done = False
time.sleep(5)
moves = 0
s = 0
board = []
while not done and moves < maxSteps:
    moves += 1
    time.sleep(1)
    display.clear_output(wait=True)
    print(trainer.model.predict(np.array(g._get_state())))
    action = trainer.get_best_action(g._get_state(), rand=False)
    print(Game.ACTION_NAMES[action])
    next_state, reward, done, goal, _ = g.move(action)
    print(g.print())
    board.append(g.print())
    s += reward
    delta = abs(goal - moves)
    print('Reward', reward)
    print('Score', s)
    print('Moves', moves)
    print("Delta : ", delta)
    print("Goal: ", goal)

saveResult(s, numberOfEpisodes, moves, board, goal, delta)
