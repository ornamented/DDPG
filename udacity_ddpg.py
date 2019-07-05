import random
from collections import namedtuple, deque
from keras import layers, models, optimizers
from keras import backend as K
import csv
from task import Task
import numpy as np
import copy

import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

import os
import shutil

EPISODES = 100000
TIMESTEPS = 1000
TEST = 10
LR_ACTOR = 0.0001
LR_CRITIC = 0.0001
GAMMA = 0.9 # 0.99
TAU = 0.01 # 0.01
OU_MU = 0.0
OU_THETA = 0.075 #0.15
OU_SIGMA = 0.1 #0.2

INIT_POSE = [0., 0., 10., 0., 0., 0.]
TARGET_POS = [0., 0., 50.]
INIT_VELOCITIES = [0., 0., 0.]

class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, mu, theta, sigma):
        """Initialize parameters and noise process."""
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(len(x))
        self.state = x + dx
        return self.state

class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, buffer_size, batch_size):
        """Initialize a ReplayBuffer object.
        Params
        ======
            buffer_size: maximum size of buffer
            batch_size: size of each training batch
        """
        self.memory = deque(maxlen=buffer_size)  # internal memory (deque)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])

    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self, batch_size=64):
        """Randomly sample a batch of experiences from memory."""
        return random.sample(self.memory, k=self.batch_size)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)


class Actor:
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, action_low, action_high):
        """Initialize parameters and build model.

        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            action_low (array): Min value of each action dimension
            action_high (array): Max value of each action dimension
        """
        self.state_size = state_size
        self.action_size = action_size
        self.action_low = action_low
        self.action_high = action_high
        self.action_range = self.action_high - self.action_low

        # Initialize any other variables here

        self.build_model()

    def build_model(self):
        """Build an actor (policy) network that maps states -> actions."""
        # Define input layer (states)
        states = layers.Input(shape=(self.state_size,), name='states')

        # Add hidden layers
        net = layers.Dense(units=400, activation='relu')(states)
        net = layers.BatchNormalization()(net)
        net = layers.Activation('relu')(net)
        net = layers.Dense(units=300, activation='relu')(net)
        # net = layers.Dense(units=32, activation='relu')(net)

        # Try different layer sizes, activations, add batch normalization, regularizers, etc.

        # Add final output layer with sigmoid activation
        raw_actions = layers.Dense(units=self.action_size, activation='sigmoid',
            name='raw_actions')(net)

        # Scale [0, 1] output for each action dimension to proper range
        actions = layers.Lambda(lambda x: (x * self.action_range) + self.action_low,
            name='actions')(raw_actions)

        # Create Keras model
        self.model = models.Model(inputs=states, outputs=actions)

        # Define loss function using action value (Q value) gradients
        action_gradients = layers.Input(shape=(self.action_size,))
        loss = K.mean(-action_gradients * actions)

        # Incorporate any additional losses here (e.g. from regularizers)

        # Define optimizer and training function
        optimizer = optimizers.Adam(lr=LR_ACTOR)
        updates_op = optimizer.get_updates(params=self.model.trainable_weights, loss=loss)
        self.train_fn = K.function(
            inputs=[self.model.input, action_gradients, K.learning_phase()],
            outputs=[],
            updates=updates_op)


class Critic:
    """Critic (Value) Model."""

    def __init__(self, state_size, action_size):
        """Initialize parameters and build model.

        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
        """
        self.state_size = state_size
        self.action_size = action_size

        # Initialize any other variables here

        self.build_model()

    def build_model(self):
        """Build a critic (value) network that maps (state, action) pairs -> Q-values."""
        # Define input layers
        states = layers.Input(shape=(self.state_size,), name='states')
        actions = layers.Input(shape=(self.action_size,), name='actions')

        # Add hidden layer(s) for state pathway
        net_states = layers.Dense(units=400, activation='relu')(states)
        net_states = layers.BatchNormalization()(net_states)
        net_states = layers.Activation('relu')(net_states)
        net_states = layers.Dense(units=300, activation='relu')(net_states)
        net_states = layers.BatchNormalization()(net_states)
        net_states = layers.Activation('relu')(net_states)
        net_states = layers.Dropout(0.2)(net_states)


        # Add hidden layer(s) for action pathway
        net_actions = layers.Dense(units=300, activation='relu')(actions)
        # net_actions = layers.Dense(units=300, activation='relu')(net_actions)
        net_actions = layers.BatchNormalization()(net_actions)
        net_actions = layers.Activation('relu')(net_actions)
        net_actions = layers.Dropout(0.2)(net_actions)

        # Try different layer sizes, activations, add batch normalization, regularizers, etc.

        # Combine state and action pathways
        net = layers.Add()([net_states, net_actions])
        net = layers.Activation('relu')(net)

        # Add more layers to the combined network if needed

        # Add final output layer to prduce action values (Q values)
        Q_values = layers.Dense(units=1, name='q_values')(net)

        # Create Keras model
        self.model = models.Model(inputs=[states, actions], outputs=Q_values)

        # Define optimizer and compile model for training with built-in loss function
        optimizer = optimizers.Adam(lr=LR_CRITIC)
        self.model.compile(optimizer=optimizer, loss='mse')

        # Compute action gradients (derivative of Q values w.r.t. to actions)
        action_gradients = K.gradients(Q_values, actions)

        # Define an additional function to fetch action gradients (to be used by actor model)
        self.get_action_gradients = K.function(
            inputs=[*self.model.input, K.learning_phase()],
            outputs=action_gradients)

class DDPG():
    """Reinforcement Learning agent that learns using DDPG."""
    def __init__(self, task):
        self.task = task
        self.state_size = task.state_size
        print('state_size:',self.state_size)
        self.action_size = task.action_size
        self.action_low = task.action_low
        self.action_high = task.action_high

        # Actor (Policy) Model
        self.actor_local = Actor(self.state_size, self.action_size, self.action_low, self.action_high)
        self.actor_target = Actor(self.state_size, self.action_size, self.action_low, self.action_high)

        # Critic (Value) Model
        self.critic_local = Critic(self.state_size, self.action_size)
        self.critic_target = Critic(self.state_size, self.action_size)

        # Initialize target model parameters with local model parameters
        self.critic_target.model.set_weights(self.critic_local.model.get_weights())
        self.actor_target.model.set_weights(self.actor_local.model.get_weights())

        # Noise process
        self.exploration_mu = OU_MU #0
        self.exploration_theta = OU_THETA #0.015
        self.exploration_sigma = OU_SIGMA #0.02
        self.noise = OUNoise(self.action_size, self.exploration_mu, self.exploration_theta, self.exploration_sigma)

        # Replay memory
        self.buffer_size = 100000
        self.batch_size = 64
        self.memory = ReplayBuffer(self.buffer_size, self.batch_size)

        # Algorithm parameters
        self.gamma = GAMMA  # discount factor
        self.tau = TAU  # for soft update of target parameters

    def reset_episode(self):
        self.noise.reset()
        state = self.task.reset()
        self.last_state = state
        return state

    def step(self, action, reward, next_state, done):
         # Save experience / reward
        self.memory.add(self.last_state, action, reward, next_state, done)

        # Learn, if enough samples are available in memory
        if len(self.memory) > self.batch_size:
            experiences = self.memory.sample()
            self.learn(experiences)

        # Roll over last state and action
        self.last_state = next_state

    def act(self, state):
        """Returns actions for given state(s) as per current policy."""
        state = np.reshape(state, [-1, self.state_size])
        action = self.actor_local.model.predict(state)[0]
        ah = self.action_high
        al = self.action_low
        return np.maximum(np.minimum(list(action + self.noise.sample()),[ah,ah,ah,ah]),[1.,1.,1.,1.])  # add some noise for exploration

    def learn(self, experiences):
        """Update policy and value parameters using given batch of experience tuples."""
        # Convert experience tuples to separate arrays for each element (states, actions, rewards, etc.)
        states = np.vstack([e.state for e in experiences if e is not None])
        actions = np.array([e.action for e in experiences if e is not None]).astype(np.float32).reshape(-1, self.action_size)
        rewards = np.array([e.reward for e in experiences if e is not None]).astype(np.float32).reshape(-1, 1)
        dones = np.array([e.done for e in experiences if e is not None]).astype(np.uint8).reshape(-1, 1)
        next_states = np.vstack([e.next_state for e in experiences if e is not None])

        # Get predicted next-state actions and Q values from target models
        #     Q_targets_next = critic_target(next_state, actor_target(next_state))
        actions_next = self.actor_target.model.predict_on_batch(next_states)
        Q_targets_next = self.critic_target.model.predict_on_batch([next_states, actions_next])

        # Compute Q targets for current states and train critic model (local)
        Q_targets = rewards + self.gamma * Q_targets_next * (1 - dones)
        self.critic_local.model.train_on_batch(x=[states, actions], y=Q_targets)

        # Train actor model (local)
        action_gradients = np.reshape(self.critic_local.get_action_gradients([states, actions, 0]), (-1, self.action_size))
        self.actor_local.train_fn([states, action_gradients, 1])  # custom training function

        # Soft-update target models
        self.soft_update(self.critic_local.model, self.critic_target.model)
        self.soft_update(self.actor_local.model, self.actor_target.model)   

    def soft_update(self, local_model, target_model):
        """Soft update model parameters."""
        local_weights = np.array(local_model.get_weights())
        target_weights = np.array(target_model.get_weights())

        assert len(local_weights) == len(target_weights), "Local and target model parameters must have the same size"

        new_weights = self.tau * local_weights + (1 - self.tau) * target_weights
        target_model.set_weights(new_weights)

def main():
    try:
        # Remove all files in episodes folder
        for root, dirs, files in os.walk('episodes/'):
            for f in files:
                os.unlink(os.path.join(root, f))
            for d in dirs:
                shutil.rmtree(os.path.join(root, d))

        # env = filter_env.makeFilteredEnv(gym.make(ENV_NAME)) 
        init_pose = INIT_POSE
        target_pos = TARGET_POS
        init_velocities = INIT_VELOCITIES
        env=Task(init_pose = init_pose,target_pos = target_pos,init_velocities=init_velocities,runtime=5.)
        ave_reward_arr = []
        ave_reward_per_ep_arr = []

        prev_ave_reward_per_ep = 0.

        agent = DDPG(env)

        file_output = 'data.txt'
        labels = ['ep','i','j','time', 'x', 'y', 'z', 'phi', 'theta', 'psi', 'x_velocity', \
            'y_velocity', 'z_velocity', 'phi_velocity', 'theta_velocity', \
            'psi_velocity', 'rotor_speed1', 'rotor_speed2', 'rotor_speed3', 'rotor_speed4','r']

        with open(file_output, 'w') as csvfile:
            writer_csv = csv.writer(csvfile, delimiter='\t')
            writer_csv.writerow(labels)

        for episode in range(EPISODES):
            agent.reset_episode()
            #print "episode:",episode
            # Train
            for step in range(TIMESTEPS):
                action = agent.act(agent.last_state)
                # De-normalize rotor speeds to 0..900 instead of -450 to 450
                # action = action + env.action_low + (env.action_high - env.action_low)/2
                next_state,reward,done = env.step(action)
                agent.step(action,reward,next_state,done)
                if done:
                    print('TRAINING', episode)
                    print('Final rotor speeds:',np.around(env.rotor_speeds,decimals = 2))
                    print('Final coordinates:',np.around(env.sim.pose[:3],decimals = 2),'\n')
                    break
            # Testing:
            if episode % 10 == 0 and episode >= 10:
                total_reward = 0
                if episode >= 20:
                    prev_ave_reward_per_ep = ave_reward_per_ep
                ave_reward_per_ep = 0
                total_j = 0
                for i in range(TEST):
                    agent.reset_episode()

                    results = {x : [] for x in labels}

                    for j in range(TIMESTEPS):
                        #env.render()
                        action = agent.act(agent.last_state) # direct action for test
                        
                        # De-normalize rotor speeds to 0..900 instead of -450 to 450
                        # action = action + env.action_low + (env.action_high - env.action_low)/2
                
                        state,reward,done = env.step(action)
                        total_reward += reward

                        with open(file_output, 'a') as csvfile:
                            writer_csv = csv.writer(csvfile, delimiter='\t')
                            to_write = [episode] + [i] + [j] + [env.sim.time] + list(env.sim.pose) + list(env.sim.v) + \
                                        list(env.sim.angular_v) + list(action) + [reward]
                            for ii in range(len(labels)):
                                results[labels[ii]].append(to_write[ii])
                            writer_csv.writerow(to_write)

                        if done:
                            total_j += j
                            print('TESTING',episode,i)
                            print('Final rotor speeds:',np.around(env.rotor_speeds,decimals = 2))
                            print('Final coordinates:',np.around(env.sim.pose[:3],decimals = 2),'\n')
                            break

                    ave_reward_per_ep += total_reward/total_j

                    # print(results['x'])
                    if len(results['x']) >= 40:
                        fig = plt.figure()
                        ax = fig.gca(projection='3d')
                        ax.set_xlim(-150,150)
                        ax.set_ylim(-150,150)
                        ax.set_zlim(0,300)
                        x = results['x']
                        y = results['y']
                        z = results['z']

                        # #2 colored by index (same in this example since z is a linspace too)
                        N = len(z)
                        ax.scatter(x, y, z, c = plt.cm.jet(np.linspace(0,1,N)))

                        plt.savefig('episodes/ep_{0:03d}_{1:02d}.png'.format(episode,i))

                        plt.close()

                ave_reward = total_reward/TEST
                ave_reward_arr.append([episode,ave_reward])
                avg_j = total_j/TEST
                ave_reward_per_ep = ave_reward_per_ep/TEST
                ave_reward_per_ep_arr.append([episode,ave_reward_per_ep])
                ave_reward_delta = ave_reward_per_ep - prev_ave_reward_per_ep
                print('\nepisode:',episode,'AVG Reward:',ave_reward,'AVG Reward per ep:', \
                    ave_reward_per_ep,'AVG reward delta:',ave_reward_delta,'avg j:',avg_j,'\n')
        # env.monitor.close()

    except KeyboardInterrupt:
        print('\nSuccesses:',env.successes)
        # print(ave_reward_arr)

        plt.plot([row[0] for row in ave_reward_arr],[row[1] for row in ave_reward_arr])
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.savefig('episodes/_final_reward.png')
        plt.close()

        plt.plot([row[0] for row in ave_reward_per_ep_arr],[row[1] for row in ave_reward_per_ep_arr])
        plt.xlabel('Episode')
        plt.ylabel('Reward per ep')
        plt.savefig('episodes/_final_reward_per_ep.png')
        plt.close()

# if __name__ == '__main__':
main()