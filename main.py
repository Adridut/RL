import random

# Random environment each time
# Neural network used

#TODO Save result in cvs files
#TODO Make it work for any numbers of exits, walls and holes
#TODO Make epsilon proportional to number of episodes

flatten = lambda l: [item for sublist in l for item in sublist]
length = 8
width = 8
maxSteps = length * width
exitNumber = 1
holeNumber = 0
wallNumber = 0
objective = 15
numberOfEpisodes = 1000



class Game:
    ACTION_UP = 0
    ACTION_LEFT = 1
    ACTION_DOWN = 2
    ACTION_RIGHT = 3

    ACTIONS = [ACTION_DOWN, ACTION_LEFT, ACTION_RIGHT, ACTION_UP]

    ACTION_NAMES = ["UP   ", "LEFT ", "DOWN ", "RIGHT"]

    MOVEMENTS = {
        ACTION_UP: (1, 0),
        ACTION_RIGHT: (0, 1),
        ACTION_LEFT: (0, -1),
        ACTION_DOWN: (-1, 0)
    }

    num_actions = len(ACTIONS)

    #wrong = 0.1
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
        x, y = self.position
        if self.alea:
            return np.reshape([self._get_grille(x, y) for [(x, y)] in
                               [[self.position], self.end]], (1, width * length * 2))
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
            action = (action + 1) % 4
        elif choice < 2 * self.wrong_action_p:
            action = (action - 1) % 4

        d_x, d_y = self.MOVEMENTS[action]
        x, y = self.position
        new_x, new_y = x + d_x, y + d_y

        r = 0

        if self.counter <= objective:
            r = -1
        else:
            r = -1

        if (new_x, new_y) in self.block:
            return self._get_state(), r, False, self.ACTIONS
        elif (new_x, new_y) in self.hole:
            self.position = new_x, new_y
            return self._get_state(), -30, True, None
        elif (new_x, new_y) in self.end:
            self.position = new_x, new_y
            return self._get_state(), 10 , True, self.ACTIONS
        elif new_x >= self.n or new_y >= self.m or new_x < 0 or new_y < 0:
            return self._get_state(), r , False, self.ACTIONS
        elif self.counter > maxSteps:
            self.position = new_x, new_y
            return self._get_state(), r , True, self.ACTIONS
        else:
            self.position = new_x, new_y
            return self._get_state(), r , False, self.ACTIONS

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
        self.state_size = width * length * 2
        self.action_size = 4
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
            model.add(Dense(50, input_dim=self.state_size, activation='relu'))
            model.add(Dense(30, activation='relu'))
            model.add(Dense(30, activation='relu'))
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
    return np.convolve(vector, [1/width]*width, mode='valid')

import time
from IPython.core.debugger import set_trace

def train(episodes, trainer, wrong_action_p, alea, collecting=False, snapshot=5000):
    batch_size = 32
    g = Game(length, width, wrong_action_p, alea=alea)
    counter = 1
    scores = []
    global_counter = 0
    losses = [0]
    epsilons = []
    delta = []

    # we start with a sequence to collect information, without learning
    if collecting:
        collecting_steps = 10000
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
    for e in range(episodes+1):
        state = g.generate_game()
        state = np.reshape(state, [1, width * length * 2])
        score = 0
        done = False
        steps = 0
        while not done:
            steps += 1
            global_counter += 1
            action = trainer.get_best_action(state)
            trainer.decay_epsilon()
            next_state, reward, done, _ = g.move(action)
            next_state = np.reshape(next_state, [1, width * length * 2])
            score += reward
            trainer.remember(state, action, reward, next_state, done)
            state = next_state
            if global_counter % 100 == 0:
                l = trainer.replay(batch_size)
                losses.append(l.history['loss'][0])
            if done:
                epsilons.append(trainer.epsilon)
                scores.append(score)
                delta.append(abs(steps - objective))
                break
            if steps > maxSteps:
                epsilons.append(trainer.epsilon)
                scores.append(score)
                delta.append(abs(steps - objective))
                break
        if e % 200 == 0:
            print("episode: {}/{}, moves: {}, score: {}, epsilon: {}, loss: {}, delta: {}"
                  .format(e, episodes, steps, score, trainer.epsilon, losses[-1], abs(steps - objective)))
        if e > 0 and e % snapshot == 0:
            trainer.save(id='iteration-%s' % e)
    return scores, losses, epsilons, delta

trainer = Trainer(learning_rate=0.001, epsilon_decay=(0.9995))
#0.999995

scores, losses, epsilons, delta = train(numberOfEpisodes, trainer, 0, True, snapshot=2500)
#35000

import matplotlib.pyplot as plt
sc = smooth(scores, width=round(numberOfEpisodes/70) + 1)

# score = np.array(scores)
# score_c = np.convolve(score, np.full((10,), 1/10), mode="same")



fig, ax1 = plt.subplots()
ax1.plot(sc)
ax2 = ax1.twinx()
ax2.plot(epsilons, color='r')
# ax3 = ax1.twinx()
# ax3.plot(delta, color='g')
# ax3.set_ylabel('Delta', color='g')
# ax3.tick_params('y', colors='g')
# ax3.spines["right"].set_position(("axes", 1.3))
ax1.set_ylabel('Score')
ax2.set_ylabel('Epsilon', color='r')
ax2.tick_params('y', colors='r')
plt.title("Score, and Epsilon over training")
ax1.set_xlabel("Episodes")
plt.figure()
plt.show()


from IPython import display
import time
#0.1
g = Game(length, width, 0, alea=True)

state = g.reset()
state = g._get_state()
print("state")
print("  ")
g.print()
done = False
time.sleep(5)
moves = 0
s = 0
while not done:
    moves += 1
    time.sleep(1)
    display.clear_output(wait=True)
    print(trainer.model.predict(np.array(g._get_state())))
    action = trainer.get_best_action(g._get_state(), rand=False)
    print(Game.ACTION_NAMES[action])
    next_state, reward, done, _ = g.move(action)
    g.print()
    s += reward
    print('Reward', reward)
    print('Score', s)
    print('Moves', moves)
    print("delta : ", abs(objective - moves))

