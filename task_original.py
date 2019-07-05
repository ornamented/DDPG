import numpy as np
from physics_sim import PhysicsSim

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

        self.state_size = self.action_repeat * 12 # adding all the velocities in
        self.action_low = 0
        self.action_high = 900
        self.action_size = 4

        # Goal
        self.target_pos = target_pos if target_pos is not None else np.array([0., 0., 10.]) 
        self.target_v = target_v if target_v is not None else np.array([0., 0., 0.]) 

    def get_reward(self):
        """Uses current pose of sim to return reward."""



        # reward = 1-.3*(abs(self.sim.pose[:2] - self.target_pos[:2])).sum() \
        # # Scale z higher than x/y since you need to overcome gravity
        # - 1.*(abs(self.sim.pose[2:3] - self.target_pos[2:])).sum() \
        # # Add scoring based on target velocities
        # # - .3*(abs(self.sim.v[:3] - self.target_v[:3])).sum()
        # # Penalize really high velocities
        # - (np.maximum(0, self.sim.v[:3] - 10)).sum()
        # + np.minimum(abs(self.sim.pose[:3]-self.sim.upper_bounds),abs(self.sim.pose[:3]-self.sim.lower_bounds),0.001).sum()
        # - (np.minimum(0.8*self.sim.lower_bounds- self.sim.pose[:3],0)**2).sum()
        # - (np.minimum(self.sim.pose[:3] - 0.8*self.sim.upper_bounds,0)**2).sum()

        # Reward from https://github.com/harshitandro/RL-Quadcopter
        # This was used for takeoff task
        # reward = np.tanh(1 - 0.003*(abs(self.sim.pose[:3] - self.target_pos))).sum()
        
        # Reward from https://github.com/sksq96/deep-rl-quadcopter
        # Used for hover
        reward = -(abs(self.sim.pose[2:3] - self.target_pos[2:])).sum()
        
        def is_equal(x, y, delta=0.0):
            return abs(x-y) <= delta

        if is_equal(self.sim.pose[2:3], self.target_pos[2:], delta=1):
            reward += 10.0  # bonus reward
            # done = True

        # Base reward
        # reward = 1-.3*(abs(self.sim.pose[:3] - self.target_pos[:3])).sum() \
        
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
        for _ in range(self.action_repeat):
            done = self.sim.next_timestep(rotor_speeds) # update the sim pose and velocities
            reward += self.get_reward() 
            pose_all.append(self.sim.pose) # 6
            pose_all.append(self.sim.v) # 3
            pose_all.append(self.sim.angular_v) # 3
        next_state = np.concatenate(pose_all)
        # print(next_state) 
        return next_state, reward, done

    def reset(self):
        """Reset the sim to start a new episode."""
        self.sim.reset()
        arr_list = (self.sim.pose, self.sim.v, self.sim.angular_v)
        state = np.concatenate(arr_list * self.action_repeat)
        return state