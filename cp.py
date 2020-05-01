"""
Classic cart-pole system implemented by Rich Sutton et al.
Copied from http://incompleteideas.net/sutton/book/code/pole.c
permalink: https://perma.cc/C9ZM-652R
"""

import math
import gym
import tf as tf
from gym import spaces, logger
from gym.utils import seeding
import numpy as np
import random


class CartPoleEnv(gym.Env):
    """
    Description:
        A pole is attached by an un-actuated joint to a cart, which moves along a frictionless track. The pendulum
        starts upright, and the goal is to prevent it from falling over by increasing and reducing the cart's velocity.
    Source:
        This environment corresponds to the version of the cart-pole problem described by Barto, Sutton, and Anderson
    Observation:
        Type: Box(4)
        Num	Observation               Min             Max
        0	Cart Position             -4.8            4.8
        1	Cart Velocity             -Inf            Inf
        2	Pole Angle                -24 deg         24 deg
        3	Pole Velocity At Tip      -Inf            Inf

    Actions:
        Type: Discrete(2)
        Num	Action
        0	Push cart to the left
        1	Push cart to the right

        Note: The amount the velocity that is reduced or increased is not fixed; it depends on the angle the pole is
        pointing. This is because the center of gravity of the pole increases the amount of energy needed to move the
        cart underneath it
    Reward:
        Reward is 1 for every step taken, including the termination step
    Starting State:
        All observations are assigned a uniform random value in [-0.05..0.05]
    Episode Termination:
        Pole Angle is more than 12 degrees
        Cart Position is more than 2.4 (center of the cart reaches the edge of the display)
        Episode length is greater than 200
        Solved Requirements
        Considered solved when the average reward is greater than or equal to 195.0 over 100 consecutive trials.
    """

    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 50
    }

    def __init__(self):
        self.gravity = 9.8
        self.masscart = 1.0
        self.masspole = 0.1
        self.total_mass = (self.masspole + self.masscart)
        self.length = 0.5  # actually half the pole's length
        self.polemass_length = (self.masspole * self.length)
        self.force_mag = 10.0
        self.tau = 0.02  # seconds between state updates
        self.kinematics_integrator = 'euler'
        # self.goal = 0.3
        self.counter = 0

        # Angle at which to fail the episode
        self.theta_threshold_radians = 12 * 2 * math.pi / 360
        self.x_threshold = 2.4

        # Angle limit set to 2 * theta_threshold_radians so failing observation is still within bounds
        high = np.array([self.x_threshold * 2,
                         np.finfo(np.float32).max,
                         self.theta_threshold_radians * 2,
                         np.finfo(np.float32).max],
                        dtype=np.float32)

        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)

        self.seed()
        self.viewer = None
        self.state = None

        self.steps_beyond_done = None

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        self.counter += 1
        assert self.action_space.contains(action), "%r (%s) invalid" % (action, type(action))
        state = self.state
        x, x_dot, theta, theta_dot = state
        force = self.force_mag if action == 1 else -self.force_mag
        costheta = math.cos(theta)
        sintheta = math.sin(theta)
        temp = (force + self.polemass_length * theta_dot * theta_dot * sintheta) / self.total_mass
        thetaacc = (self.gravity * sintheta - costheta * temp) / (
                self.length * (4.0 / 3.0 - self.masspole * costheta * costheta / self.total_mass))
        xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass
        if self.kinematics_integrator == 'euler':
            x = x + self.tau * x_dot
            x_dot = x_dot + self.tau * xacc
            theta = theta + self.tau * theta_dot
            theta_dot = theta_dot + self.tau * thetaacc
        else:  # semi-implicit euler
            x_dot = x_dot + self.tau * xacc
            x = x + self.tau * x_dot
            theta_dot = theta_dot + self.tau * thetaacc
            theta = theta + self.tau * theta_dot
        self.state = (x, x_dot, theta, theta_dot) #self.goal
        done = x < -self.x_threshold \
               or x > self.x_threshold \
               or theta < -self.theta_threshold_radians \
               or theta > self.theta_threshold_radians
        done = bool(done)

        if not done:
            reward = 1.0
            # if self.counter < self.goal:
            #     reward = 1.0
            # else:
            #     reward = -1.0
        elif self.steps_beyond_done is None:
            # Pole just fell!
            self.steps_beyond_done = 0
            # reward = 1.0
            reward = 0.0
        else:
            if self.steps_beyond_done == 0:
                logger.warn(
                    "You are calling 'step()' even though this environment has already returned done = True. You should always call 'reset()' once you receive 'done = True' -- any further steps are undefined behavior.")
            self.steps_beyond_done += 1
            reward = 0.0

        return np.array(self.state), reward, done, {}

    def reset(self):
        self.state = self.np_random.uniform(low=-0.05, high=0.05, size=(4,)) #5
        self.steps_beyond_done = None
        return np.array(self.state)

    def render(self, mode='human'):
        screen_width = 600
        screen_height = 400

        world_width = self.x_threshold * 2
        scale = screen_width / world_width
        carty = 100  # TOP OF CART
        polewidth = 10.0
        polelen = scale * (2 * self.length)
        cartwidth = 50.0
        cartheight = 30.0

        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height)
            l, r, t, b = -cartwidth / 2, cartwidth / 2, cartheight / 2, -cartheight / 2
            axleoffset = cartheight / 4.0
            cart = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            self.carttrans = rendering.Transform()
            cart.add_attr(self.carttrans)
            self.viewer.add_geom(cart)
            l, r, t, b = -polewidth / 2, polewidth / 2, polelen - polewidth / 2, -polewidth / 2
            pole = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            pole.set_color(.8, .6, .4)
            self.poletrans = rendering.Transform(translation=(0, axleoffset))
            pole.add_attr(self.poletrans)
            pole.add_attr(self.carttrans)
            self.viewer.add_geom(pole)
            self.axle = rendering.make_circle(polewidth / 2)
            self.axle.add_attr(self.poletrans)
            self.axle.add_attr(self.carttrans)
            self.axle.set_color(.5, .5, .8)
            self.viewer.add_geom(self.axle)
            self.track = rendering.Line((0, carty), (screen_width, carty))
            self.track.set_color(0, 0, 0)
            self.viewer.add_geom(self.track)

            self._pole_geom = pole

        if self.state is None: return None

        # Edit the pole polygon vertex
        pole = self._pole_geom
        l, r, t, b = -polewidth / 2, polewidth / 2, polelen - polewidth / 2, -polewidth / 2
        pole.v = [(l, b), (l, t), (r, t), (r, b)]

        x = self.state
        cartx = x[0] * scale + screen_width / 2.0  # MIDDLE OF CART
        self.carttrans.set_translation(cartx, carty)
        self.poletrans.set_rotation(-x[2])

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None


import sys
import gym
import pylab
import random
import numpy as np
from collections import deque
from keras.layers import Dense
from keras.optimizers import Adam
from keras.models import Sequential

EPISODES = 300


# DQN Agent for the Cartpole
# it uses Neural Network to approximate q function
# and replay memory & target q network
class DQNAgent:
    def __init__(self, state_size, action_size):
        # if you want to see Cartpole learning, then change to True
        self.render = False
        self.load_model = False

        # get size of state and action
        self.state_size = state_size
        self.action_size = action_size

        # These are hyper parameters for the DQN
        self.discount_factor = 0.99
        self.learning_rate = 0.001
        self.epsilon = 1.0
        self.epsilon_decay = 0.999
        self.epsilon_min = 0.01
        self.batch_size = 64
        self.train_start = 1000
        # create replay memory using deque
        self.memory = deque(maxlen=2000)

        # create main model and target model
        self.model = self.build_model()
        self.target_model = self.build_model()

        # initialize target model
        self.update_target_model()

        if self.load_model:
            self.model.load_weights("./save_model/cartpole_dqn.h5")

    # approximate Q function using Neural Network
    # state is input and Q Value of each action is output of network
    def build_model(self):
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation='relu',
                        kernel_initializer='he_uniform'))
        model.add(Dense(24, activation='relu',
                        kernel_initializer='he_uniform'))
        model.add(Dense(self.action_size, activation='linear',
                        kernel_initializer='he_uniform'))
        model.summary()
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        return model

    # after some time interval update the target model to be same with model
    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    # get action from model using epsilon-greedy policy
    def get_action(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        else:
            q_value = self.model.predict(state)
            return np.argmax(q_value[0])

    # save sample <s,a,r,s'> to the replay memory
    def append_sample(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    # pick samples randomly from replay memory (with batch_size)
    def train_model(self):
        if len(self.memory) < self.train_start:
            return
        batch_size = min(self.batch_size, len(self.memory))
        mini_batch = random.sample(self.memory, batch_size)

        update_input = np.zeros((batch_size, self.state_size))
        update_target = np.zeros((batch_size, self.state_size))
        action, reward, done = [], [], []

        for i in range(self.batch_size):
            update_input[i] = mini_batch[i][0]
            action.append(mini_batch[i][1])
            reward.append(mini_batch[i][2])
            update_target[i] = mini_batch[i][3]
            done.append(mini_batch[i][4])

        target = self.model.predict(update_input)
        target_val = self.target_model.predict(update_target)

        for i in range(self.batch_size):
            # Q Learning: get maximum Q value at s' from target model
            if done[i]:
                target[i][action[i]] = reward[i]
            else:
                target[i][action[i]] = reward[i] + self.discount_factor * (
                    np.amax(target_val[i]))

        # and do the model fit!
        self.model.fit(update_input, target, batch_size=self.batch_size,
                       epochs=1, verbose=0)


if __name__ == "__main__":
    # In case of CartPole-v1, maximum length of episode is 500
    env = CartPoleEnv()
    # get size of state and action from environment
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    agent = DQNAgent(state_size, action_size)

    scores, episodes = [], []
    e = 0

    while e < EPISODES:
        done = False
        score = 0
        state = env.reset()
        state = np.reshape(state, [1, state_size])

        while not done:


            if e == EPISODES - 1:
                env.render()

            # get action for the current state and go one step in environment
            action = agent.get_action(state)
            next_state, reward, done, info = env.step(action)
            next_state = np.reshape(next_state, [1, state_size])
            # if an action make the episode end, then gives penalty of -100
            reward = reward if not done or score == 499 else -100

            # save the sample <s, a, r, s'> to the replay memory
            agent.append_sample(state, action, reward, next_state, done)
            # every time step do the training
            agent.train_model()
            score += reward
            state = next_state

            if done:
                # every episode update the target model to be same with model
                agent.update_target_model()

                # every episode, plot the play time
                score = score if score == 500 else score + 100
                scores.append(score)
                episodes.append(e)
                pylab.plot(episodes, scores, 'b')
                pylab.savefig("cartpole_dqn.png")
                print("episode:", e, "  score:", score, "  memory length:",
                      len(agent.memory), "  epsilon:", agent.epsilon)

                e += 1

                if np.mean(scores[-min(10, len(scores)):]) > 490:
                    print('***OUI***')
                    e = EPISODES - 1
                    # sys.exit()

            # if the mean of scores of last 10 episode is bigger than 490
                # stop training



        # save the model
        # if e % 50 == 0:
        #     agent.model.save_weights("./save_model/cartpole_dqn.h5")

################################################ Genetic Algo

# import numpy as np
# import math
# from matplotlib import pyplot as plt
# from random import randint
# from statistics import median, mean
#
# # env = gym.make('CartPole-v0')
# env = CartPoleEnv()
# maxTime = 10
# goal = 0.3
#
# ind = env.observation_space.shape[0]
# adim = env.action_space.n  # discrete
#
# # adim = env.action_space.shape[0] # continues
#
#
# award_set = []
# test_run = 150
# best_gen = []
#
#
# def softmax(x):
#     x = np.exp(x) / np.sum(np.exp(x))
#     return x
#
#
# def lreLu(x):
#     alpha = 0.2
#     return tf.nn.relu(x) - alpha * tf.nn.relu(-x)
#
#
# def sigmoid(x):
#     return 1 / (1 + np.exp(-x))
#
#
# def reLu(x):
#     return np.maximum(0, x)
#
#
# # Function generate initial set of weights and bias
# def intial_gen(test_run):
#     input_weight = []
#     input_bias = []
#
#     hidden_weight = []
#     out_weight = []
#
#     g = []
#
#     in_node = 5  # 4,2 combination good
#     hid_node = 2
#
#     for i in range(test_run):
#         # in_w = np.random.rand(ind, in_node)
#         in_w = np.random.rand(5, in_node)
#         input_weight.append(in_w)
#
#         in_b = np.random.rand((in_node))
#         input_bias.append(in_b)
#
#         hid_w = np.random.rand(in_node, hid_node)
#         hidden_weight.append(hid_w)
#
#         out_w = np.random.rand(hid_node, adim)
#         out_weight.append(out_w)
#
#         g.append(goal)
#
#     generation = [input_weight, input_bias, hidden_weight, out_weight, g]
#     return generation
#
#
# # creat a neural network
# def nn(obs, in_w, in_b, hid_w, out_w, g):
#     # obs = np.reshape(obs,(1,4))
#     # obs = np.array(obs).reshape(1,len(obs))
#
#     # hid_layer = np.dot(Ain,hid_w)
#     # Ahid = sigmoid(np.dot(Ain,hid_w))
#     # obs = obs/max(np.max(obs),1)
#
#     obs = obs / max(np.max(np.linalg.norm(obs)), 1)
#
#     Ain = reLu(np.dot(obs, in_w) + in_b.T)
#
#     Ahid = reLu(np.dot(Ain, hid_w))
#     lhid = np.dot(Ahid, out_w)
#
#
#     out_put = reLu(lhid)
#     out_put = softmax(out_put)
#     out_put = out_put.argsort().reshape(1, adim)
#     act = out_put[0][0]  # index of discrete action
#
#     # Continues actions
#     # out_put = 2*np.tanh(np.dot(Ahid,out_w))
#     # act = out_put.reshape(adim) # Vector of continues actions
#
#     return act
#
#
# def run_env(env, in_w, in_b, hid_w, out_w, g):
#     obs = env.reset()
#     award = 0
#     for t in range(maxTime):
#         # env.render() #this slows the process
#         action = nn(obs, in_w, in_b, hid_w, out_w, g)
#         obs, reward, done, info = env.step(action)
#         award += reward
#         if done:
#             break
#     return award
#
#
# # Run environment randomly
# def rand_run(env, test_run):
#     award_set = []
#     generations = intial_gen(test_run)
#
#     for episode in range(test_run):  # run env 10 time
#         in_w = generations[0][episode]
#         in_b = generations[1][episode]
#         hid_w = generations[2][episode]
#         out_w = generations[3][episode]
#         g = generations[4][episode]
#         award = run_env(env, in_w, in_b, hid_w, out_w, g)
#         award_set = np.append(award_set, award)
#     gen_award = [generations, award_set]
#     return gen_award
#
#
# def mutation(new_dna):
#     j = np.random.randint(0, len(new_dna))
#     if (0 < j < 10):  # controlling rate of amount mutation
#         for ix in range(j):
#             n = np.random.randint(0, len(new_dna))  # random postion for mutation
#             new_dna[n] = new_dna[n] + np.random.rand()
#
#     mut_dna = new_dna
#
#     return mut_dna
#
#
# def crossover(Dna_list):
#     newDNA_list = []
#     newDNA_list.append(Dna_list[0])
#     newDNA_list.append(Dna_list[1])
#
#     for l in range(10):  # generation after crassover
#         j = np.random.randint(0, len(Dna_list[0]))
#         new_dna = np.append(Dna_list[0][:j], Dna_list[1][j:])
#
#         mut_dna = mutation(new_dna)
#         newDNA_list.append(mut_dna)
#
#     return newDNA_list
#
#
# # Generate new set of weigts and bias from the best previous weights and bias
#
# def reproduce(award_set, generations):
#     good_award_idx = award_set.argsort()[-2:][::-1]  # here only best 2 are selected
#     good_generation = []
#     DNA_list = []
#
#     new_input_weight = []
#     new_input_bias = []
#
#     new_hidden_weight = []
#
#     new_output_weight = []
#
#     new_g = []
#
#     new_award_set = []
#
#     # Extraction of all weight info into a single sequence
#     for index in good_award_idx:
#         w1 = generations[0][index]
#         dna_in_w = w1.reshape(w1.shape[1], -1)
#
#         b1 = generations[1][index]
#         dna_b1 = np.append(dna_in_w, b1)
#
#         w2 = generations[2][index]
#         dna_whid = w2.reshape(w2.shape[1], -1)
#         dna_w2 = np.append(dna_b1, dna_whid)
#
#         wh = generations[3][index]
#         dna = np.append(dna_w2, wh)
#
#         DNA_list.append(dna)  # make 2 dna for good gerneration
#
#     newDNA_list = crossover(DNA_list)
#
#     for newdna in newDNA_list:  # collection of weights from dna info
#
#         newdna_in_w1 = np.array(newdna[:generations[0][0].size])
#         new_in_w = np.reshape(newdna_in_w1, (-1, generations[0][0].shape[1]))
#         new_input_weight.append(new_in_w)
#
#         new_in_b = np.array([newdna[newdna_in_w1.size:newdna_in_w1.size + generations[1][0].size]]).T  # bias
#         new_input_bias.append(new_in_b)
#
#         sh = newdna_in_w1.size + new_in_b.size
#         newdna_in_w2 = np.array([newdna[sh:sh + generations[2][0].size]])
#         new_hid_w = np.reshape(newdna_in_w2, (-1, generations[2][0].shape[1]))
#         new_hidden_weight.append(new_hid_w)
#
#         sl = newdna_in_w1.size + new_in_b.size + newdna_in_w2.size
#         new_out_w = np.array([newdna[sl:]]).T
#         new_out_w = np.reshape(new_out_w, (-1, generations[3][0].shape[1]))
#         new_output_weight.append(new_out_w)
#
#         new_g.append(goal)
#
#         new_award = run_env(env, new_in_w, new_in_b, new_hid_w, new_out_w, new_g)  # bias
#         new_award_set = np.append(new_award_set, new_award)
#
#
#     new_generation = [new_input_weight, new_input_bias, new_hidden_weight, new_output_weight, new_g]
#
#     return new_generation, new_award_set
#
#
# def evolution(env, test_run, n_of_generations):
#     gen_award = rand_run(env, test_run)
#
#     current_gens = gen_award[0]
#     current_award_set = gen_award[1]
#     best_gen = []
#     A = []
#     iw = 0
#     ib = 0
#     hw = 0
#     ow = 0
#     g = 0
#     for n in range(n_of_generations):
#         new_generation, new_award_set = reproduce(current_award_set, current_gens)
#         current_gens = new_generation
#         current_award_set = new_award_set
#         avg = np.average(current_award_set)
#         #avg > 4500
#         if avg == goal:
#             best_gen = np.array([current_gens[0][0], current_gens[1][0], current_gens[2][0], current_gens[3][0]], current_gens[4][0])
#             iw = current_gens[0][0]
#             ib = current_gens[1][0]
#             hw = current_gens[2][0]
#             ow = current_gens[3][0]
#             g = current_gens[4][0]
#         a = np.amax(current_award_set)
#         print("generation: {}, score: {}".format(n + 1, a))
#         A = np.append(A, a)
#     Best_award = np.amax(A)
#
#     plt.plot(A)
#     plt.xlabel('generations')
#     plt.ylabel('score')
#     plt.grid()
#
#     print('Average accepted score:', mean(A))
#     print('Median score for accepted scores:', median(A))
#     return plt.show(), iw, ib, hw, ow, g
#
#
# n_of_generations = 100
# _, iw, ib, hw, ow, g = evolution(env, test_run, n_of_generations)
#
# in_w = iw
# in_b = ib
# hid_w = hw
# out_w = ow
#
#
#
# def test_run_env(env, in_w, in_b, hid_w, out_w, g):
#     obs = env.reset()
#     award = 0
#     for t in range(maxTime):
#         env.render()  # thia slows the process
#         action = nn(obs, in_w, in_b, hid_w, out_w, g)
#         obs, reward, done, info = env.step(action)
#         award += reward
#
#         print("time: {}, fitness: {}".format(t, award))
#         if done:
#             break
#     return award
#
#
# print(test_run_env(env, in_w, in_b, hid_w, out_w, g))
# env.close()

