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

        def dist_reward(current,goal,lb,ub):
            dist_reward = (abs(goal - current) / max(abs(goal-lb),abs(goal-ub))) ** 0.8
            return dist_reward

        def partial_reward(current,goal,lb,ub):
            partial_reward = 0.
            for i in range(len(goal)):
                temp = dist_reward(current[i],goal[i],lb[i],ub[i])
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

        # Penalty for crossing boundaries
        for i in range(len(self.sim.pose[:3])):
            if abs(self.sim.pose[i]-self.sim.lower_bounds[i]) < 0.1:
                reward += -10.0
            if abs(self.sim.pose[i]-self.sim.upper_bounds[i]) < 0.1:
                reward += -10.0
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
        return next_state, reward, done

    def reset(self):
        """Reset the sim to start a new episode."""
        self.sim.reset()
        # For pose only
        state = np.concatenate([self.sim.pose] * self.action_repeat)
        
        return state