import gym
from baselines.ppo2.policies import CnnPolicy, LstmPolicy, LnLstmPolicy
from baselines.a2c import a2c
from baselines.a2c.a2c import Model
import gym_gridworld
import tensorflow as tf


def main():
    env = gym.make("gridworld-v0")
    policy=CnnPolicy
    nsteps = 5
    total_timesteps = int(80e6)
    vf_coef = 0.5
    ent_coef = 0.01
    max_grad_norm = 0.5
    lr = 7e-4
    lrschedule = 'linear'
    epsilon = 1e-5
    alpha = 0.99
    gamma = 0.99
    log_interval = 100
    ob_space = env.observation_space
    ac_space = env.action_space
    nenvs = env.num_envs
    #with tf.Session() as sess:
    with tf.Graph().as_default(), tf.Session().as_default():
        #model = a2c.learn(policy=CnnPolicy, env=env,  total_timesteps=int(0), seed=0)
        model=Model(policy=policy, ob_space=ob_space, ac_space=ac_space, nenvs=nenvs, nsteps=nsteps, ent_coef=ent_coef, vf_coef=vf_coef,
        max_grad_norm=max_grad_norm, lr=lr, alpha=alpha, epsilon=epsilon, total_timesteps=total_timesteps, lrschedule=lrschedule)
        model.load("/Users/constantinos/Documents/Projects/cmu_gridworld/cmu_gym/a2c_open.pkl")

        while True:
            obs, done = env.reset(), False
            episode_rew = 0
            while not done:
                env.render()
                obs, rew, done, _ = env.step(model(obs[None])[0])
                episode_rew += rew
            print("Episode reward", episode_rew)


if __name__ == '__main__':
    main()