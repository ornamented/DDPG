# import filter_env
from ddpg import *
from task import Task
import csv
import gc
gc.enable()

# ENV_NAME = 'InvertedPendulum-v1'
EPISODES = 100000
TIMESTEPS = 1000
TEST = 10

def main():
    # env = filter_env.makeFilteredEnv(gym.make(ENV_NAME)) 
    init_pose = [0., 0., 10., 0., 0., 0.]
    target_pos = [0., 0., 150.]
    init_velocities = [0., 0., 0.]
    env=Task(init_pose = init_pose,target_pos = target_pos,init_velocities=init_velocities,runtime=30.)

    agent = DDPG(env)
    # env.monitor.start('experiments/' + ENV_NAME,force=True)


    file_output = 'data.txt'
    labels = ['ep','i','j','time', 'x', 'y', 'z', 'phi', 'theta', 'psi', 'x_velocity', \
        'y_velocity', 'z_velocity', 'phi_velocity', 'theta_velocity', \
        'psi_velocity', 'rotor_speed1', 'rotor_speed2', 'rotor_speed3', 'rotor_speed4','r']

    with open(file_output, 'w') as csvfile:
        writer_csv = csv.writer(csvfile, delimiter='\t')
        writer_csv.writerow(labels)

    for episode in range(EPISODES):
        state = env.reset()
        #print "episode:",episode
        # Train
        for step in range(TIMESTEPS):
            action = agent.noise_action(state)
            # De-normalize rotor speeds to 0..900 instead of -450 to 450
            action = action + env.action_low + (env.action_high - env.action_low)/2
            next_state,reward,done = env.step(action)
            agent.perceive(state,action,reward,next_state,done)
            state = next_state
            if done:
                break
        # Testing:
        if episode % 10 == 0 and episode > 10:
            total_reward = 0
            total_j = 0
            for i in range(TEST):
                state = env.reset()

                results = {x : [] for x in labels}

                for j in range(TIMESTEPS):
                    #env.render()
                    action = agent.action(state) # direct action for test
                    
                    # De-normalize rotor speeds to 0..900 instead of -450 to 450
                    action = action + env.action_low + (env.action_high - env.action_low)/2
            
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
                        break
            ave_reward = total_reward/TEST
            avg_j = total_j/TEST
            print('episode: ',episode,'Evaluation Average Reward:',ave_reward,'avg j: ',avg_j)
    # env.monitor.close()

if __name__ == '__main__':
    main()
