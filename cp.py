"""
Classic cart-pole system implemented by Rich Sutton et al.
Copied from http://incompleteideas.net/sutton/book/code/pole.c
permalink: https://perma.cc/C9ZM-652R
"""

#TODO try with max = 25 and 200
#TODO try with layers size of 96x2 (192)
#TODO try to add counter in state
#TODO try to delete max and min range in state

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
        # self.goal = 0.003
        self.goal = (random.randint(10, 200))/10000
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
        self.delta = (abs(self.counter - (self.goal * 10000))) / 10000
        assert self.action_space.contains(action), "%r (%s) invalid" % (action, type(action))
        state = self.state
        x, x_dot, theta, theta_dot, _, _, _ = state
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
        self.state = (x, x_dot, theta, theta_dot, self.goal, self.delta, self.counter / 10000)
        done = x < -self.x_threshold \
               or x > self.x_threshold \
               or theta < -self.theta_threshold_radians \
               or theta > self.theta_threshold_radians
        done = bool(done)



        # if not done:

        if self.steps_beyond_done is None:
            # Pole just fell!
            self.steps_beyond_done = 0
            # if self.counter < self.goal * 10000:
            #     reward = -1.0
            # else:
            #     reward = 1.0
            reward = 0.0
        else:
            if self.steps_beyond_done == 0:
                logger.warn(
                    "You are calling 'step()' even though this environment has already returned done = True. You should always call 'reset()' once you receive 'done = True' -- any further steps are undefined behavior.")
            self.steps_beyond_done += 1
            reward = 0.0

        if self.counter >= 200:
            done = True

        if self.counter <= self.goal * 10000:
            reward = 1
        else:
            reward = -1.0

        return np.array(self.state), reward, done,  {}, self.counter, (self.goal * 10000), self.delta * 10000

    def reset(self):
        self.state = self.np_random.uniform(low=-0.05, high=0.05, size=(7,)) #4
        self.steps_beyond_done = None
        self.counter = 0
        self.goal = (random.randint(10, 200))/10000
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

EPISODES = 7000



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
        self.epsilon_decay = 0.9999 #0.999
        self.epsilon_min = 0.01
        self.batch_size = 96
        self.train_start = 1000
        # create replay memory using deque
        self.memory = deque(maxlen=20000000) #2000

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

        model.add(Dense(96, input_dim=self.state_size, activation='relu',
                        kernel_initializer='he_uniform'))
        model.add(Dense(96, activation='relu',
                        kernel_initializer='he_uniform'))
        model.add(Dense(96, activation='relu',
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
    # state_size = env.observation_space.shape[0]
    state_size = 7
    action_size = env.action_space.n



    def smooth(vector, width=30):
        return np.convolve(vector, [1 / width] * width, mode='valid')

    n = 0
    lower_bound = []
    upper_bound = []
    delta_lower_bound = []
    delta_upper_bound = []
    while n < 5:
        agent = DQNAgent(state_size, action_size)
        n += 1
        scores, episodes, deltas, epsilons = [], [], [], []
        e = 0

        while e < EPISODES:
            done = False
            score = 0
            state = env.reset()
            state = np.reshape(state, [1, state_size])

            while not done:


                # if e == EPISODES - 1:
                #     env.render()

                # get action for the current state and go one step in environment
                action = agent.get_action(state)
                next_state, reward, done, info, counter, goal, delta = env.step(action)
                next_state = np.reshape(next_state, [1, state_size])
                # if an action make the episode end, then gives penalty of -100
                # reward = reward if not done or score == 499 else -100

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
                    # score = score if score == 500 else score + 100

                    scores.append(score)
                    episodes.append(e)
                    deltas.append(delta)
                    epsilons.append(agent.epsilon)
                    # pylab.savefig("cartpole_dqn.png")
                    if e % 100 == 0:
                        print("episode:", e, "  score:", score, "  memory length:",
                               len(agent.memory), "  epsilon:", agent.epsilon, "  Counter:", counter, "  Goal:", goal, "  Delta:", delta)

                    e += 1


                    if e >= EPISODES:

                        if n == 1:
                            upper_bound = scores.copy()
                            lower_bound = scores.copy()
                            delta_upper_bound = deltas.copy()
                            delta_lower_bound = deltas.copy()
                        else:
                            i = 0
                            while i < len(scores):
                                if scores[i] > upper_bound[i]:
                                    upper_bound[i] = scores[i]
                                if scores[i] < lower_bound[i]:
                                    lower_bound[i] = scores[i]
                                if deltas[i] > delta_upper_bound[i]:
                                    delta_upper_bound[i] = deltas[i]
                                if deltas[i] < delta_lower_bound[i]:
                                    delta_lower_bound[i] = deltas[i]
                                i += 1

                        if n == 5:
                            sc = smooth(scores, width=round(EPISODES / 70) + 1)
                            d = smooth(deltas, width=round(EPISODES / 70) + 1)
                            ub = smooth(upper_bound, width=round(EPISODES / 70) + 1)
                            lb = smooth(lower_bound, width=round(EPISODES / 70) + 1)
                            dub = smooth(delta_upper_bound, width=round(EPISODES / 70) + 1)
                            dlb = smooth(delta_lower_bound, width=round(EPISODES / 70) + 1)

                            middle_list = []
                            delta_middle_list = []
                            i = 0
                            while i < len(ub):
                                delta_middle_list.append((dub[i] + dlb[i]) / 2)
                                middle_list.append((ub[i] + lb[i]) / 2)
                                i += 1
                            # x = range(0,numberOfEpisodes)
                            x = range(0, len(ub))

                            fig, ax1 = pylab.subplots()
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
                            pylab.title("Score, Epsilon and Delta over training")
                            ax1.set_xlabel("Episodes")
                            fileName = input('Name: ')
                            pylab.savefig(fileName + ".png")
                            break

                # if np.mean(scores[-min(10, len(scores)):]) >= w:
                #     e = EPISODES - 1
                    # sys.exit()

            # if the mean of scores of last 10 episode is bigger than 4
                # stop training



        # save the model
        # if e % 50 == 0:
        #     agent.model.save_weights("./save_model/cartpole_dqn.h5")

