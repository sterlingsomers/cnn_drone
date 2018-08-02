from collections import namedtuple
from pysc2.lib import actions

import numpy as np
import sys
from actorcritic.agent import ActorCriticAgent, ACMode
from common.preprocess import ObsProcesser, ActionProcesser, FEATURE_KEYS
from common.util import calculate_n_step_reward, general_n_step_advantage, combine_first_dimensions, \
    dict_of_lists_to_list_of_dicst
import tensorflow as tf
from absl import flags
from time import sleep

PPORunParams = namedtuple("PPORunParams", ["lambda_par", "batch_size", "n_epochs"])


class Runner(object):
    def __init__(
            self,
            envs,
            agent: ActorCriticAgent,
            n_steps=5,
            discount=0.99,
            do_training=True,
            ppo_par: PPORunParams = None,
            n_envs=1
    ):
        self.envs = envs
        self.n_envs = n_envs
        self.agent = agent
        self.obs_processer = ObsProcesser()
        self.action_processer = ActionProcesser(dim=flags.FLAGS.resolution)
        self.n_steps = n_steps
        self.discount = discount
        self.do_training = do_training
        self.ppo_par = ppo_par
        self.batch_counter = 0
        self.episode_counter = 0
        self.score = 0.0
        assert self.agent.mode in [ACMode.PPO, ACMode.A2C]
        self.is_ppo = self.agent.mode == ACMode.PPO
        if self.is_ppo:
            assert ppo_par is not None
            assert n_steps * envs.n_envs % ppo_par.batch_size == 0
            assert n_steps * envs.n_envs >= ppo_par.batch_size
            self.ppo_par = ppo_par

    def reset(self):
        #self.score = 0.0
        obs = self.envs.reset()
        self.latest_obs = self.obs_processer.process(obs)

    def reset_demo(self):
        #self.score = 0.0
        obs = self.envs.reset()
        self.latest_obs = self.obs_processer.process([obs])

    def _log_score_to_tb(self, score):
        summary = tf.Summary()
        summary.value.add(tag='sc2/episode_score', simple_value=score)
        self.agent.summary_writer.add_summary(summary, self.episode_counter)

    def _handle_episode_end(self, timestep):
        #(MINE) This timestep is actually the last set of feature observations
        #score = timestep.observation["score_cumulative"][0]
        self.score = (self.score + timestep) # //self.episode_counter # It is zero at the beginning so you get inf
        print(">>>>>>>>>>>>>>>>>>>>>>>>>>>episode %d ended. Score %f" % (self.episode_counter, self.score))
        self._log_score_to_tb(self.score)
        self.episode_counter += 1
        #self.reset() # Error if Monitor doesnt have the option to reset without an env to be done (THIS RESETS ALL ENVS!!! YOU NEED remot.send(env.reset) to reset a specific env. Else restart within the env

    def _train_ppo_epoch(self, full_input):
        total_obs = self.n_steps * self.envs.n_envs
        shuffle_idx = np.random.permutation(total_obs)
        batches = dict_of_lists_to_list_of_dicst({
            k: np.split(v[shuffle_idx], total_obs // self.ppo_par.batch_size)
            for k, v in full_input.items()
        })
        for b in batches:
            self.agent.train(b)

    def run_batch(self):
        #(MINE) MAIN LOOP!!!
        mb_actions = []
        mb_obs = []
        mb_values = np.zeros((self.envs.num_envs, self.n_steps + 1), dtype=np.float32)
        mb_rewards = np.zeros((self.envs.num_envs, self.n_steps), dtype=np.float32)
        mb_done = np.zeros((self.envs.num_envs, self.n_steps), dtype=np.int32)

        latest_obs = self.latest_obs # (MINE) =state(t)

        for n in range(self.n_steps):
            # could calculate value estimate from obs when do training
            # but saving values here will make n step reward calculation a bit easier
            action_ids, value_estimate = self.agent.step(latest_obs)
            print('|step:', n, '|actions:', action_ids)  # (MINE) If you put it after the envs.step the SUCCESS appears at the envs.step so it will appear oddly
            # (MINE) Store actions and value estimates for all steps
            mb_values[:, n] = value_estimate
            mb_obs.append(latest_obs)
            mb_actions.append((action_ids))
            # (MINE)  do action, return it to environment, get new obs and reward, store reward
            #actions_pp = self.action_processer.process(action_ids) # Actions have changed now need to check: BEFORE: actions.FunctionCall(actions.FUNCTIONS.no_op.id, []) NOW: actions.FUNCTIONS.no_op()
            obs_raw = self.envs.step(action_ids)
            #obs_raw.reward = reward
            latest_obs = self.obs_processer.process(obs_raw[0]) # For obs_raw as tuple! #(MINE) =state(t+1). Processes all inputs/obs from all timesteps (and envs)
            print('-->|rewards:', np.round(np.mean(obs_raw[1]), 3))
            mb_rewards[:, n] = [t for t in obs_raw[1]]
            mb_done[:, n] = [t for t in obs_raw[2]]

            #Check for all t (timestep/observation in obs_raw which t has the last state true, meaning it is the last state
            # IF MAX_STEPS OR GOAL REACHED
            # You can use as below for obs_raw[4] which is success of failure
            #print(obs_raw[2])
            indx=0
            for t in obs_raw[2]:
                if t == True: # done=true
                    # Put reward in scores
                    self._handle_episode_end(obs_raw[1][indx]) # The printing score process is NOT a parallel process apparrently as you input every reward (t) independently
                indx = indx + 1
            # for t in obs_raw:
            #     if t.last():
            #         self._handle_episode_end(t)

        print(">> Avg. Reward:",np.round(np.mean(mb_rewards),3))
        mb_values[:, -1] = self.agent.get_value(latest_obs) # We bootstrap from last step if not terminal! although he doesnt use any check here

        n_step_advantage = general_n_step_advantage(
            mb_rewards,
            mb_values,
            self.discount,
            mb_done,
            lambda_par=self.ppo_par.lambda_par if self.is_ppo else 1.0
        )

        full_input = {
            # these are transposed because action/obs
            # processers return [time, env, ...] shaped arrays
            FEATURE_KEYS.advantage: n_step_advantage.transpose(),
            FEATURE_KEYS.value_target: (n_step_advantage + mb_values[:, :-1]).transpose()
        }
        #(MINE) Probably we combine all experiences from every worker below
        full_input.update(self.action_processer.combine_batch(mb_actions))
        full_input.update(self.obs_processer.combine_batch(mb_obs))
        full_input = {k: combine_first_dimensions(v) for k, v in full_input.items()}

        if not self.do_training:
            pass
        elif self.agent.mode == ACMode.A2C:
            self.agent.train(full_input)
        elif self.agent.mode == ACMode.PPO:
            for epoch in range(self.ppo_par.n_epochs):
                self._train_ppo_epoch(full_input)
            self.agent.update_theta()

        self.latest_obs = latest_obs
        self.batch_counter += 1
        print('Batch %d finished' % self.batch_counter)
        sys.stdout.flush()

    def run_trained_batch(self):
        sleep(2.0)
        # state = state(0), initialized by the env.reset() in run_agent
        latest_obs = self.latest_obs # (MINE) =state(t)
        # action = agent(state)
        action_ids, value_estimate = self.agent.step_eval(latest_obs) # (MINE) AGENT STEP = INPUT TO NN THE CURRENT STATE AND OUTPUT ACTION
        print('|actions:', action_ids)
        obs_raw = self.envs.step(action_ids) # It will also visualize the next observation if all the episodes have ended as after success it retunrs the obs from reset
        latest_obs = self.obs_processer.process(obs_raw[0:-3])  # (MINE) =process(state(t+1)). Processes all inputs/obs from all timesteps
        print('-->|rewards:', np.round(np.mean(obs_raw[1]), 3))



        if obs_raw[2]:
            # for r in obs_raw[1]: # You will double count here as t
            self._handle_episode_end(obs_raw[1])  # The printing score process is NOT a parallel process apparrently as you input every reward (t) independently

        # Check for all t (timestep/observation in obs_raw which t has the last state true, meaning it is the last state
        # for t in obs_raw:
        #     if t.last():
        #         self._handle_episode_end(t)

        self.latest_obs = latest_obs # (MINE) state(t) = state(t+1), the usual s=s'
        self.batch_counter += 1
        #print('Batch %d finished' % self.batch_counter)
        sys.stdout.flush()
