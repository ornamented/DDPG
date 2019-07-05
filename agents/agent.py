""" 
Implementation of DDPG - Deep Deterministic Policy Gradient

Algorithm and hyperparameter details can be found here: 
    http://arxiv.org/pdf/1509.02971v2.pdf

The algorithm is tested on the Pendulum-v0 OpenAI gym task 
and developed with tflearn + Tensorflow

Author: Patrick Emami
"""
import sys
sys.path.append("..") # this is to import task from one folder above

import tensorflow as tf
import numpy as np
import gym
# from gym import wrappers
import tflearn
import argparse
import pprint as pp


from task import Task
import csv
import glob
import os
import re


from replay_buffer import ReplayBuffer

# ===========================
#   Actor and Critic DNNs
# ===========================

class ActorNetwork(object):
    """
    Input to the network is the state, output is the action
    under a deterministic policy.

    The output layer activation is a tanh to keep the action
    between -action_bound and action_bound
    """

    def __init__(self, sess, state_dim, action_dim, action_bound, learning_rate, tau, batch_size):
        self.sess = sess
        self.s_dim = state_dim
        self.a_dim = action_dim
        self.action_bound = action_bound
        self.learning_rate = learning_rate
        self.tau = tau
        self.batch_size = batch_size

        # Actor Network
        self.inputs, self.out, self.scaled_out = self.create_actor_network()

        self.network_params = tf.trainable_variables()

        # Target Network
        self.target_inputs, self.target_out, self.target_scaled_out = self.create_actor_network()

        self.target_network_params = tf.trainable_variables()[
            len(self.network_params):]

        # Op for periodically updating target network with online network
        # weights
        self.update_target_network_params = \
            [self.target_network_params[i].assign(tf.multiply(self.network_params[i], self.tau) +
                                                  tf.multiply(self.target_network_params[i], 1. - self.tau))
                for i in range(len(self.target_network_params))]

        # This gradient will be provided by the critic network
        self.action_gradient = tf.placeholder(tf.float32, [None, self.a_dim])

        # Combine the gradients here
        self.unnormalized_actor_gradients = tf.gradients(
            self.scaled_out, self.network_params, -self.action_gradient)
        self.actor_gradients = list(map(lambda x: tf.div(x, self.batch_size), self.unnormalized_actor_gradients))

        # Optimization Op
        self.optimize = tf.train.AdamOptimizer(self.learning_rate).\
            apply_gradients(zip(self.actor_gradients, self.network_params))

        self.num_trainable_vars = len(
            self.network_params) + len(self.target_network_params)

    def create_actor_network(self):
        inputs = tflearn.input_data(shape=[None, self.s_dim])
        net = tflearn.fully_connected(inputs, 400) # original - 400
        net = tflearn.layers.normalization.batch_normalization(net)
        # Added dropouts, they seem to help a bit to get out of the local minima
        net = tflearn.layers.core.dropout(net,0.3)
        net = tflearn.activations.relu(net)
        net = tflearn.fully_connected(net, 300) # original - 300
        net = tflearn.layers.normalization.batch_normalization(net)
        net = tflearn.layers.core.dropout(net,0.3)
        net = tflearn.activations.relu(net)
        # Final layer weights are init to Uniform[-3e-3, 3e-3]
        # w_init = tflearn.initializations.uniform(minval=-0.003, maxval=0.003)
        # made positive so copter doesn't start falling right away
        w_init = tflearn.initializations.uniform(minval=0.001, maxval=0.006)
        out = tflearn.fully_connected(
            net, self.a_dim, activation='tanh', weights_init=w_init)
        # Scale output to -action_bound to action_bound
        scaled_out = tf.multiply(out, self.action_bound)
        return inputs, out, scaled_out

    def train(self, inputs, a_gradient):
        self.sess.run(self.optimize, feed_dict={
            self.inputs: inputs,
            self.action_gradient: a_gradient
        })

    def predict(self, inputs):
        return self.sess.run(self.scaled_out, feed_dict={
            self.inputs: inputs
        })

    def predict_target(self, inputs):
        return self.sess.run(self.target_scaled_out, feed_dict={
            self.target_inputs: inputs
        })

    def update_target_network(self):
        self.sess.run(self.update_target_network_params)

    def get_num_trainable_vars(self):
        return self.num_trainable_vars


class CriticNetwork(object):
    """
    Input to the network is the state and action, output is Q(s,a).
    The action must be obtained from the output of the Actor network.

    """

    def __init__(self, sess, state_dim, action_dim, learning_rate, tau, gamma, num_actor_vars):
        self.sess = sess
        self.s_dim = state_dim
        self.a_dim = action_dim
        self.learning_rate = learning_rate
        self.tau = tau
        self.gamma = gamma

        # Create the critic network
        self.inputs, self.action, self.out = self.create_critic_network()

        self.network_params = tf.trainable_variables()[num_actor_vars:]

        # Target Network
        self.target_inputs, self.target_action, self.target_out = self.create_critic_network()

        self.target_network_params = tf.trainable_variables()[(len(self.network_params) + num_actor_vars):]

        # Op for periodically updating target network with online network
        # weights with regularization
        self.update_target_network_params = \
            [self.target_network_params[i].assign(tf.multiply(self.network_params[i], self.tau) \
            + tf.multiply(self.target_network_params[i], 1. - self.tau))
                for i in range(len(self.target_network_params))]

        # Network target (y_i)
        self.predicted_q_value = tf.placeholder(tf.float32, [None, 1])

        # Define loss and optimization Op
        self.loss = tflearn.mean_square(self.predicted_q_value, self.out)
        self.optimize = tf.train.AdamOptimizer(
            self.learning_rate).minimize(self.loss)

        # Get the gradient of the net w.r.t. the action.
        # For each action in the minibatch (i.e., for each x in xs),
        # this will sum up the gradients of each critic output in the minibatch
        # w.r.t. that action. Each output is independent of all
        # actions except for one.
        self.action_grads = tf.gradients(self.out, self.action)

    def create_critic_network(self):
        inputs = tflearn.input_data(shape=[None, self.s_dim])
        action = tflearn.input_data(shape=[None, self.a_dim])
        net = tflearn.fully_connected(inputs, 400)
        net = tflearn.layers.normalization.batch_normalization(net)
        # Trying to add dropout here too...
        net = tflearn.layers.core.dropout(net,0.3)
        net = tflearn.activations.relu(net)

        # Add the action tensor in the 2nd hidden layer
        # Use two temp layers to get the corresponding weights and biases
        t1 = tflearn.fully_connected(net, 300)
        t2 = tflearn.fully_connected(action, 300)

        net = tflearn.activation(
            tf.matmul(net, t1.W) + tf.matmul(action, t2.W) + t2.b, activation='relu')

        # linear layer connected to 1 output representing Q(s,a)
        # Weights are init to Uniform[-3e-3, 3e-3]
        w_init = tflearn.initializations.uniform(minval=-0.003, maxval=0.003)
        out = tflearn.fully_connected(net, 1, weights_init=w_init)
        return inputs, action, out

    def train(self, inputs, action, predicted_q_value):
        return self.sess.run([self.out, self.optimize], feed_dict={
            self.inputs: inputs,
            self.action: action,
            self.predicted_q_value: predicted_q_value
        })

    def predict(self, inputs, action):
        return self.sess.run(self.out, feed_dict={
            self.inputs: inputs,
            self.action: action
        })

    def predict_target(self, inputs, action):
        return self.sess.run(self.target_out, feed_dict={
            self.target_inputs: inputs,
            self.target_action: action
        })

    def action_gradients(self, inputs, actions):
        return self.sess.run(self.action_grads, feed_dict={
            self.inputs: inputs,
            self.action: actions
        })

    def update_target_network(self):
        self.sess.run(self.update_target_network_params)

# Taken from https://github.com/openai/baselines/blob/master/baselines/ddpg/noise.py, which is
# based on http://math.stackexchange.com/questions/1287634/implementing-ornstein-uhlenbeck-in-matlab
class OrnsteinUhlenbeckActionNoise:
    def __init__(self, mu, sigma=0.3, theta=.15, dt=1e-2, x0=None):
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.x0 = x0
        self.reset()

    def __call__(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + \
                self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        self.x_prev = x
        return x

    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)

    def __repr__(self):
        return 'OrnsteinUhlenbeckActionNoise(mu={}, sigma={})'.format(self.mu, self.sigma)

# ===========================
#   Tensorflow Summary Ops
# ===========================

def build_summaries():
    episode_reward = tf.Variable(0.)
    tf.summary.scalar("Reward", episode_reward)
    episode_avg_reward = tf.Variable(0.)
    tf.summary.scalar("Avg Reward", episode_avg_reward)
    episode_ave_max_q = tf.Variable(0.)
    tf.summary.scalar("Qmax Value", episode_ave_max_q)
    episode_len = tf.Variable(0.)
    tf.summary.scalar("Episode Len", episode_len)

    summary_vars = [episode_reward, episode_avg_reward, episode_ave_max_q, episode_len]
    summary_ops = tf.summary.merge_all()

    return summary_ops, summary_vars

# ===========================
#   Agent Training
# ===========================

def train(sess, env, args, actor, critic, actor_noise):

    # Set up summary Ops
    summary_ops, summary_vars = build_summaries()

    # Put each run into a separate folder with the next highest digit compared 
    # to what's already there. This is for good summaries in TensorBoard
    list_of_files = glob.glob(args['summary_dir'] + '/*') # * means all if need specific format then *.csv
    latest_file = max(list_of_files, key=os.path.getctime)
    latest_digit = int(re.search('(?<=/run).*$',latest_file).group(0))
    latest_digit += 1
    if latest_digit is None: 
        latest_digit = 0

    sess.run(tf.global_variables_initializer())
    writer = tf.summary.FileWriter(args['summary_dir'] + '/run' + str(latest_digit), sess.graph)

    # Initialize target network weights
    actor.update_target_network()
    critic.update_target_network()

    # Initialize replay memory
    replay_buffer = ReplayBuffer(int(args['buffer_size']), int(args['random_seed']))

    file_output = 'data.txt'
    labels = ['i','j','time', 'x', 'y', 'z', 'phi', 'theta', 'psi', 'x_velocity', \
      'y_velocity', 'z_velocity', 'phi_velocity', 'theta_velocity', \
      'psi_velocity', 'rotor_speed1', 'rotor_speed2', 'rotor_speed3', 'rotor_speed4','r']

    with open(file_output, 'w') as csvfile:
        writer_csv = csv.writer(csvfile, delimiter='\t')
        writer_csv.writerow(labels)

    # Needed to enable BatchNorm. 
    # This hurts the performance on Pendulum but could be useful
    # in other environments.
    # tflearn.is_training(True)

    # IN this is the same as the part in ipynb under The Agent heading
    for i in range(int(args['max_episodes'])):

        s = env.reset()
        r0 = env.get_reward()

        ep_reward = 0.
        ep_avg_reward = 0.
        ep_ave_max_q = 0.

        results = {x : [] for x in labels}

        for j in range(int(args['max_episode_len'])):

            # if args['render_env']:
            #     env.render()

            # Added exploration noise
            #a = actor.predict(np.reshape(s, (1, 3))) + (1. / (1. + i))
            a = actor.predict(np.reshape(s, (1, actor.s_dim))) + actor_noise()
            # print(a)
            # print(actor.action_bound)

            # Bounding to -450..450 to prevent weird things happening in sim
            # Need to add 0.1 for lower bound so physics_sim doesn't freak out on line 114...
            for k in range(a[0].size):
                a[0][k] = max(-1*actor.action_bound+0.1,a[0][k])
                a[0][k] = min(actor.action_bound,a[0][k])
            # print(a)

            # De-normalize rotor speeds to 0..900 instead of -450 to 450
            a[0] = a[0] + env.action_low + (env.action_high - env.action_low)/2
            # print(a[0])
            # print(a)

            s2, r, terminal = env.step(a[0])
            
            # s2, r, terminal, info = env.step(a[0])

            replay_buffer.add(np.reshape(s, (actor.s_dim,)), np.reshape(a, (actor.a_dim,)), r,
                              terminal, np.reshape(s2, (actor.s_dim,)))

            # Keep adding experience to the memory until
            # there are at least minibatch size samples
            if replay_buffer.size() > int(args['minibatch_size']):
                s_batch, a_batch, r_batch, t_batch, s2_batch = \
                    replay_buffer.sample_batch(int(args['minibatch_size']))

                # Calculate targets
                target_q = critic.predict_target(
                    s2_batch, actor.predict_target(s2_batch))

                y_i = []
                for k in range(int(args['minibatch_size'])):
                    if t_batch[k]:
                        y_i.append(r_batch[k])
                    else:
                        y_i.append(r_batch[k] + critic.gamma * target_q[k])

                # Update the critic given the targets
                predicted_q_value, _ = critic.train(
                    s_batch, a_batch, np.reshape(y_i, (int(args['minibatch_size']), 1)))

                ep_ave_max_q += np.amax(predicted_q_value)

                # Update the actor policy using the sampled gradient
                a_outs = actor.predict(s_batch)
                grads = critic.action_gradients(s_batch, a_outs)
                actor.train(s_batch, grads[0])

                # Update target networks
                actor.update_target_network()
                critic.update_target_network()

            s = s2
            ep_reward += r

            # print('i: {:d}, j: {:d} r:{:.4f}, ep_reward: {:.4f}'.format(i,j,r,ep_reward))

            # if i == int(args['max_episodes']):      
            with open(file_output, 'a') as csvfile:
                writer_csv = csv.writer(csvfile, delimiter='\t')
                to_write = [i] + [j] + [env.sim.time] + list(env.sim.pose) + list(env.sim.v) + \
                            list(env.sim.angular_v) + list(a[0]) + [r]
                for ii in range(len(labels)):
                    results[labels[ii]].append(to_write[ii])
                writer_csv.writerow(to_write)


            if terminal:

                summary_str = sess.run(summary_ops, feed_dict={
                    summary_vars[0]: ep_reward,
                    summary_vars[1]: ep_reward / float(j+1),
                    summary_vars[2]: ep_ave_max_q / float(j+1),
                    summary_vars[3]: j
                })

                writer.add_summary(summary_str, i)
                writer.flush()

                print('| Reward: {:.4f} | Avg reward: {:.4f} | Qmax: {:.4f} | Episode Len: {:d} | Episode: {:d}'.format(float(ep_reward), \
                        float(ep_reward)/(j+1), (ep_ave_max_q / float(j+1)), j+1, i))
                break

            


def main(args):

    with tf.Session() as sess:

        target_pos = np.array(args['target_pos'])
        init_pose = np.array(args['init_pose'])
        init_velocities = np.array(args['init_velocities'])
        env=Task(init_pose = init_pose,target_pos = target_pos,init_velocities=init_velocities,runtime=30.)
        # env = gym.make(args['env'])
        # print(args['random_seed'])
        np.random.seed(int(args['random_seed']))
        tf.set_random_seed(int(args['random_seed']))
        # env.seed(int(args['random_seed']))

        # state_dim = env.observation_space.shape[0]
        # action_dim = env.action_space.shape[0]
        # action_bound = env.action_space.high
        state_dim = env.state_size
        print('state_dim: {:d}'.format(state_dim))
        action_dim = env.action_size
        print('action_dim: {:d}'.format(action_dim))
        action_bound = (env.action_high - env.action_low)/2
        print('action_bound: {:4f}'.format(action_bound))

        # print(action_bound)
        # Ensure action bound is symmetric
        # assert (env.action_high == -env.action_low)

        actor = ActorNetwork(sess, state_dim, action_dim, action_bound,
                             float(args['actor_lr']), float(args['tau']),
                             int(args['minibatch_size']))

        critic = CriticNetwork(sess, state_dim, action_dim,
                               float(args['critic_lr']), float(args['tau']),
                               float(args['gamma']),
                               actor.get_num_trainable_vars())
        
        actor_noise = OrnsteinUhlenbeckActionNoise(mu=np.zeros(action_dim))

        # if args['use_gym_monitor']:
        #     if not args['render_env']:
        #         env = wrappers.Monitor(
        #             env, args['monitor_dir'], video_callable=False, force=True)
        #     else:
        #         env = wrappers.Monitor(env, args['monitor_dir'], force=True)

        train(sess, env, args, actor, critic, actor_noise)

        # if args['use_gym_monitor']:
        #     env.monitor.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='provide arguments for DDPG agent')

    # agent parameters
    parser.add_argument('--actor-lr', help='actor network learning rate', default=0.001) # original 0.0001
    parser.add_argument('--critic-lr', help='critic network learning rate', default=0.0001) # original 0.001
    parser.add_argument('--gamma', help='discount factor for critic updates', default=0.99) # original 0.99
    parser.add_argument('--tau', help='soft target update parameter', default=0.001)
    parser.add_argument('--buffer-size', help='max size of the replay buffer', default=1000000)
    parser.add_argument('--minibatch-size', help='size of minibatch for minibatch-SGD', default=64) # original 64

    # run parameters
    parser.add_argument('--target-pos', help='target copter position', default=[0., 0., 150.])
    parser.add_argument('--init-pose', help='initial copter pose', default=[0., 0., 150., 0., 0., 0.])
    parser.add_argument('--init-velocities', help='initial copter velocities', default=[0., 0., 0.])
    # parser.add_argument('--env', help='choose the gym env- tested on {Pendulum-v0}', default='Pendulum-v0') # not used
    parser.add_argument('--random-seed', help='random seed for repeatability', default=1234)
    parser.add_argument('--max-episodes', help='max num of episodes to do while training', default=1000)
    parser.add_argument('--max-episode-len', help='max length of 1 episode', default=1000) # this is more bound by runtime
    # parser.add_argument('--render-env', help='render the gym env', action='store_true')
    # parser.add_argument('--use-gym-monitor', help='record gym results', action='store_true')
    # parser.add_argument('--monitor-dir', help='directory for storing gym results', default='./results/gym_ddpg')
    parser.add_argument('--summary-dir', help='directory for storing tensorboard info', default='./results/tf_ddpg')

    parser.set_defaults(render_env=False)
    parser.set_defaults(use_gym_monitor=True)
    
    args = vars(parser.parse_args())
    
    pp.pprint(args)

    main(args)
