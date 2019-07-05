import numpy as np
import math
from physics_sim import PhysicsSim

np.set_printoptions(formatter={'all':lambda x: str(x)})

class Task():
    """Task (environment) that defines the goal and provides feedback to the agent."""
    def __init__(self, init_pose=None, init_velocities=None, 
        init_angle_velocities=None, runtime=5., target_pos=None, target_v=None):
        """Initialize a Task object.
        Params
        ======
            init_pose: initial position of the quadcopter in (x,y,z) dimensions and the Euler angles
            init_velocities: initial velocity of the quadcopter in (x,y,z) dimensions
            init_angle_velocities: initial radians/second for each of the three Euler angles
            runtime: time limit for each episode
            target_pos: target/goal (x,y,z) position for the agent
        """
        # Simulation
        self.sim = PhysicsSim(init_pose, init_velocities, init_angle_velocities, runtime) 
        self.action_repeat = 3

        # For pose only
        self.state_size = self.action_repeat * 6 # base case, pose only
        
        # For pose and velocity
        # self.state_size = self.action_repeat * 12 # adding all the velocities in
        
        self.action_low = 0
        self.action_high = 900
        self.action_size = 4
        self.successes = 0

        self.rotor_speeds = []

        self.last_post = self.sim.pose

        # Goal
        self.target_pos = target_pos if target_pos is not None else np.array([0., 0., 10.]) 
        self.target_v = target_v if target_v is not None else np.array([0., 0., 0.]) 

    def get_reward(self):
        """Uses current pose of sim to return reward."""

        # Base reward
        # reward = 1-.3*(abs(self.sim.pose[:3] - self.target_pos[:3])).sum() \
        
        # reward = 1 - .3*((abs(self.sim.pose[:3]-self.target_pos)/np.maximum(abs(self.target_pos-self.sim.lower_bounds),abs(self.target_pos-self.sim.upper_bounds)))^0.4).sum()


        # Dist reward without velocity discount

        def dist_reward(current,goal,lb,ub):
            dist_reward = (abs(goal - current) / max(abs(goal-lb),abs(goal-ub))) ** 0.8
            return dist_reward

        def partial_reward(current,goal,lb,ub):
            partial_reward = 0.
            for i in range(len(goal)):
                temp = dist_reward(current[i],goal[i],lb[i],ub[i])
                # print('i:',i,'dist_reward:',temp)
                # print('i: ',i,'dist_reward: ',dist_reward(current[i],goal[i],lb[i],ub[i]))
                # print('i: ',i,'velocity_discount: ',velocity_discount(current[i],goal[i],lb[i],ub[i],v[i],ub_v[i]))
                # print('i: ',i,'reward: ',temp)
                partial_reward += temp
            return partial_reward

        reward = (1 - .3*partial_reward(self.sim.pose[:3],self.target_pos, \
                self.sim.lower_bounds,self.sim.upper_bounds)) \
                # / self.action_repeat

        # add penalty for angular positions being outta whack
        penalty = (1. - abs(math.sin(self.sim.pose[3])))
        penalty *= (1. - abs(math.sin(self.sim.pose[4])))
        penalty *= (1. - abs(math.sin(self.sim.pose[5])))

        reward *= penalty

        # Compare all dimensions to target, if within 5m then add 10 as reward
        if np.array_equal(np.maximum(abs(self.sim.pose[:3]-self.target_pos),5.),[5.,5.,5.]):
            reward += 100.0

        # Bonus for being kinda close
        # if np.array_equal(np.maximum(abs(self.sim.pose[:3]-self.target_pos),5.),[10.,10.,10.]):
        #     reward += 2.0

        # Penalty for crossing boundaries
        for i in range(len(self.sim.pose[:3])):
            if abs(self.sim.pose[i]-self.sim.lower_bounds[i]) < 0.1:
                reward += -10.0
            if abs(self.sim.pose[i]-self.sim.upper_bounds[i]) < 0.1:
                reward += -10.0


        # reward from https://github.com/StillSad/RL-Quadcopter-2/blob/master/take_off.py

        # #1、计算当前位置与目标位置差值的绝对值
        # temp = abs(self.sim.pose[:3] - self.target_pos)
        # #2、(self.sim.pose[2] - self.target_pos[2]) 当前位置与目标位置高度的差值作为奖励的第一部分，
        # # 当前位置低于目标位置奖励为负值，当前位置高于目标位置奖励为正值；
        # # 奖励的第二部分是当前位置与目标位置xy轴上的偏差，偏差越大奖励越小，防止上升时xy轴上有较大偏移
        # reward = (self.sim.pose[2] - self.target_pos[2]) - 0.5 * temp[:2].sum()
        # # 3、奖励的第三部分是当起飞高度大于目标位置高度时奖励增加30，小于目标位置高度时奖励减少10；可以让飞行器更快的达到起飞高度
        # if self.sim.pose[2] >= self.target_pos[2]:
        #     reward += 30.0  # bonus reward
        # else:
        #     reward -= 10
        # #4、奖励的第四部分是当前位置高度与上次位置高度的差值，若当前位置高度低于上次位置高度说明飞行器在下降奖励减少，
        # # 当前位置高度等于上次位置高度奖励不变，当前位置高度高于上次位置高度说明飞行器在上升，奖励增加
        # temp = self.sim.pose - self.last_post
        # reward += temp[2]


        # Reward from https://github.com/parkitny/rl-project/blob/master/RL-Quadcopter-2/land_task.py
        # Include a factor to ensure there is no exagerated tilting
        # Penalise for large angular movement, example theta:
        # 1 - sin(theta), = 1 if theta = 0 and = 0 if theta -> 90 degrees
        # Introduce factor (1-sin(theta) * (1 - sin(phi)) * (1 - sin(psi))

        #
        # Keep reward in the vicinity of 0 - 1, introduce another penalty
        # exp(-1/r) where r is the pythagorean distance from current 
        # position to target.

        #
        # exp(-1/r) -> 0 as r -> 0
        # exp(-1/r) -> 1 as r -> inf

        # penalty = (1. - abs(math.sin(self.sim.pose[3])))
        # penalty *= (1. - abs(math.sin(self.sim.pose[4])))
        # penalty *= (1. - abs(math.sin(self.sim.pose[5])))

        # delta = abs(self.sim.pose[:3] - self.target_pos)
        # r = math.sqrt(np.dot(delta, delta))
        
        # if(r > 0.01): decay = math.exp(-1/r) # Give range -1 to 1
        # else: decay = 0
        # reward = 1. - decay
        # reward *= penalty
        # return reward


        # Reward from https://github.com/harshitandro/RL-Quadcopter
        # This was used for takeoff task
        # reward = np.tanh(1 - 0.003*(abs(self.sim.pose[:3] - self.target_pos))).sum()


        # Trying to use a more complex reward function w/ velocity discount

        # def dist_reward(current,goal,lb,ub):
        #     dist_reward = (abs(goal - current) / max(abs(goal)-lb,abs(goal-ub))) ** 0.4
        #     return dist_reward

        # def velocity_discount(current,goal,lb,ub,v,ub_v):
        #     # if 1 - max(abs(v/ub_v),0.1) < 0:
        #     #     print('curr_v:',self.sim.v)
        #     #     print('curr_pose:',self.sim.pose)
        #     #     print('velocity_discount: ',1 - max(abs(v/ub_v),0.1),1/max(dist_reward(current,goal,lb,ub),0.1))
        #     #     print('v:',v,'ub_v:',ub_v)
        #     discount = pow(max(1 - max(abs(v/ub_v),0.1),0),1/max(dist_reward(current,goal,lb,ub),0.1))
        #     return discount

        # def partial_reward(current,goal,lb,ub,v,ub_v):
        #     partial_reward = 0.
        #     for i in range(len(goal)):
        #         temp = dist_reward(current[i],goal[i],lb[i],ub[i]) * velocity_discount(current[i],goal[i],lb[i],ub[i],v[i],ub_v[i])
        #         # print('i: ',i,'dist_reward: ',dist_reward(current[i],goal[i],lb[i],ub[i]))
        #         # print('i: ',i,'velocity_discount: ',velocity_discount(current[i],goal[i],lb[i],ub[i],v[i],ub_v[i]))
        #         # print('i: ',i,'reward: ',temp)
        #         partial_reward += temp
        #     return partial_reward

        # reward = (1 - .3*partial_reward(self.sim.pose[:3],self.target_pos, \
        #         self.sim.lower_bounds,self.sim.upper_bounds, self.sim.v[:3],[50,50,200])) \
        #         / self.action_repeat


        # reward = 1-.3*(abs(self.sim.pose[:2] - self.target_pos[:2])**2).sum() \
        # Scale z higher than x/y since you need to overcome gravity
        # - 1.*(abs(self.sim.pose[2:3] - self.target_pos[2:])**2).sum() \
        
        # Add scoring based on target velocities
        # - .3*(abs(self.sim.v[:3] - self.target_v[:3])).sum()
        
        # Penalize really high velocities
        # - (np.maximum(0, self.sim.v[:3] - 10)).sum()

        # + np.minimum(abs(self.sim.pose[:3]-self.sim.upper_bounds),abs(self.sim.pose[:3]-self.sim.lower_bounds),0.001).sum()
        # - (np.minimum(0.8*self.sim.lower_bounds- self.sim.pose[:3],0)**2).sum()
        # - (np.minimum(self.sim.pose[:3] - 0.8*self.sim.upper_bounds,0)**2).sum()


        
        # Reward from https://github.com/sksq96/deep-rl-quadcopter
        # Used for hover
        # reward = -(abs(self.sim.pose[2:3] - self.target_pos[2:])).sum()


        # Adding rotor speed difference penalties
        # for i in range(len(self.rotor_speeds)-1):
            # for j in range(i+1,len(self.rotor_speeds)):
            #     if abs(self.rotor_speeds[i]-self.rotor_speeds[j]) > 400.0:
            #         reward += -0.1
        
        
        # def is_equal(x, y, delta=0.0):
        #     return abs(x-y) <= delta

        # if is_equal(self.sim.pose[2:3], self.target_pos[2:], delta=5.):
        #     reward += 10.0  # bonus reward

        # 1/x +ve reward and 10000/y bounds
        # reward = (1/(abs(self.sim.pose[:3] - self.target_pos)+0.01)).sum() \
        # - 10000*(1/(abs(self.sim.pose[:3] - self.sim.lower_bounds[:])+0.001)).sum() \
        # - 10000*(1/(abs(self.sim.pose[:3] - self.sim.upper_bounds[:])+0.001)).sum()
        
        # Negative squared
        # reward = 1-.3*(abs(self.sim.pose[:3] - self.target_pos)**2).sum() 

        # Incentivize copter from flying out of bounds - the closer to bound, the greater the negative reward
        # - (1/abs(self.sim.pose[:3] - self.sim.lower_bounds[:])).sum() \
        # - (1/abs(self.sim.pose[:3] - self.sim.upper_bounds[:])).sum()
        return reward

    def step(self, rotor_speeds):
        """Uses action to obtain next state, reward, done."""
        reward = 0
        pose_all = []
        self.rotor_speeds = rotor_speeds
        
        # For pose only
        for _ in range(self.action_repeat):
            done = self.sim.next_timestep(rotor_speeds) # update the sim pose and velocities
            reward += self.get_reward() 
            pose_all.append(self.sim.pose) # 6

        # For pose and velocity
        # for _ in range(self.action_repeat):
        #     done = self.sim.next_timestep(rotor_speeds) # update the sim pose and velocities
        #     reward += self.get_reward() 
        #     pose_all.append(self.sim.pose) # 6
        #     pose_all.append(self.sim.v) # 3
        #     pose_all.append(self.sim.angular_v) # 3
        
        next_state = np.concatenate(pose_all)

        # Terminate episode if distance to goal is within an acceptable threshold
        if np.array_equal(np.maximum(abs(self.sim.pose[:3]-self.target_pos),5.),[5.,5.,5.]):
            print('*******')
            print(' ',abs(self.sim.pose[:3]-self.target_pos))
            print(' ',self.sim.pose[:3])
            print(' ',self.target_pos)
            done = True
            print(' Target reached!\n*******\n')
            self.successes += 1
        # print(next_state) 
        # if done:
        #     print('Final rotor speeds:',np.around(rotor_speeds,decimals = 2))
        #     print('Final coordinates:',np.around(self.sim.pose[:3],decimals = 2),'\n')
        return next_state, reward, done

    def reset(self):
        """Reset the sim to start a new episode."""
        self.sim.reset()
        # For pose only
        state = np.concatenate([self.sim.pose] * self.action_repeat)
        
        # For pose and velocity
        # arr_list = (self.sim.pose, self.sim.v, self.sim.angular_v)
        # state = np.concatenate(arr_list * self.action_repeat)
        
        return state