#!/usr/bin/env python3
import os
import gym
import argparse
from baselines.bench import  Monitor
from baselines import logger
#from baselines.common.cmd_util import make_atari_env, atari_arg_parser
from baselines.common.vec_env.vec_frame_stack import VecFrameStack
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
from baselines.a2c.a2c import learn
from baselines.ppo2.policies import CnnPolicy, LstmPolicy, LnLstmPolicy
#from gym_gridworld_opeanai_viz import gym_grid
import gym_gridworld


def train(env_id, num_timesteps, seed, policy, lrschedule, num_env):
    if policy == 'cnn':
        policy_fn = CnnPolicy
    elif policy == 'lstm':
        policy_fn = LstmPolicy
    elif policy == 'lnlstm':
        policy_fn = LnLstmPolicy
    #env = VecFrameStack(make_atari_env(env_id, num_env, seed), 4)
    env = VecFrameStack(make_custom_env('gridworld-v0', num_env, seed), 1)
    act = learn(policy_fn, env, seed, total_timesteps=int(num_timesteps * 1.1), lrschedule=lrschedule)
    act.save('a2c_bopen.pkl')
    env.close()

def make_custom_env(env_id, num_env, seed, wrapper_kwargs=None, start_index=0):
    """
    Create a wrapped, monitored SubprocVecEnv for Atari.
    """
    if wrapper_kwargs is None: wrapper_kwargs = {}
    def make_env(rank): # pylint: disable=C0111
        def _thunk():
            env = gym.make(env_id)
            env.seed(seed + rank)
            env = Monitor(env, logger.get_dir() and os.path.join(logger.get_dir(), str(rank)))
            return env
        return _thunk
    #set_global_seeds(seed)
    return SubprocVecEnv([make_env(i + start_index) for i in range(num_env)])

def main():
    #parser = atari_arg_parser()
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--policy', help='Policy architecture', choices=['cnn', 'lstm', 'lnlstm'], default='cnn')
    parser.add_argument('--lrschedule', help='Learning rate schedule', choices=['constant', 'linear'], default='constant')
    parser.add_argument('--env', type=str, default='gridworld-v0')
    parser.add_argument('--num_timesteps', type=int, default=int(100))
    parser.add_argument('--seed', help='RNG seed', type=int, default=2)
    args = parser.parse_args()
    #args.env = 'gridworld-v0'
    logger.configure()
    train(args.env, num_timesteps=args.num_timesteps, seed=args.seed,
        policy=args.policy, lrschedule=args.lrschedule, num_env=20)

if __name__ == '__main__':
    main()
