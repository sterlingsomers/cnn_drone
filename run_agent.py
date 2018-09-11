# Fully Convolutional Network (the 2nd in DeepMind's paper)
import logging
import os
import shutil
import sys
from datetime import datetime
from time import sleep
import numpy as np
#from functools import partial
#
from absl import flags
from actorcritic.agent import ActorCriticAgent, ACMode
from actorcritic.runner import Runner, PPORunParams
# from common.multienv import SubprocVecEnv, make_sc2env, SingleEnv, make_env

import tensorflow as tf
# import tensorflow.contrib.eager as tfe
# tf.enable_eager_execution()

#import argparse
from baselines import logger
from baselines.bench import Monitor
#from baselines.common.misc_util import boolean_flag
from baselines.common import set_global_seeds
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
from baselines.common.vec_env.vec_frame_stack import VecFrameStack
#from baselines.common.vec_env.vec_normalize import VecNormalize
import gym
#import gym_gridworld
#from gym_grid.envs import GridEnv
import gym_gridworld

FLAGS = flags.FLAGS
flags.DEFINE_bool("visualize", False, "Whether to render with pygame.")
flags.DEFINE_integer("resolution", 32, "Resolution for screen and minimap feature layers.")
flags.DEFINE_integer("step_mul", 1, "Game steps per agent step.")
flags.DEFINE_integer("n_envs", 20, "Number of environments to run in parallel")
flags.DEFINE_integer("episodes", 10, "Number of complete episodes")
flags.DEFINE_integer("n_steps_per_batch", None,
    "Number of steps per batch, if None use 8 for a2c and 128 for ppo")  # (MINE) TIMESTEPS HERE!!! You need them cauz you dont want to run till it finds the beacon especially at first episodes - will take forever
flags.DEFINE_integer("all_summary_freq", 50, "Record all summaries every n batch")
flags.DEFINE_integer("scalar_summary_freq", 5, "Record scalar summaries every n batch")
flags.DEFINE_string("checkpoint_path", "_files/models", "Path for agent checkpoints")
flags.DEFINE_string("summary_path", "_files/summaries", "Path for tensorboard summaries")
flags.DEFINE_string("model_name", "Drop_rand_positive_reward_reducedsteppenalty", "Name for checkpoints and tensorboard summaries")
flags.DEFINE_integer("K_batches", 50000, # Batch is like a training epoch!
    "Number of training batches to run in thousands, use -1 to run forever") #(MINE) not for now
flags.DEFINE_string("map_name", "DefeatRoaches", "Name of a map to use.")
flags.DEFINE_float("discount", 0.95, "Reward-discount for the agent")
flags.DEFINE_boolean("training", True,
    "if should train the model, if false then save only episode score summaries"
)
flags.DEFINE_enum("if_output_exists", "continue", ["fail", "overwrite", "continue"],
    "What to do if summary and model output exists, only for training, is ignored if notraining")
flags.DEFINE_float("max_gradient_norm", 500.0, "good value might depend on the environment")
flags.DEFINE_float("loss_value_weight", 1.0, "good value might depend on the environment") # orig:1.0
flags.DEFINE_float("entropy_weight_spatial", 1e-6,
    "entropy of spatial action distribution loss weight") # orig:1e-6
flags.DEFINE_float("entropy_weight_action", 1e-6, "entropy of action-id distribution loss weight") # orig:1e-6
flags.DEFINE_float("ppo_lambda", 0.95, "lambda parameter for ppo")
flags.DEFINE_integer("ppo_batch_size", None, "batch size for ppo, if None use n_steps_per_batch")
flags.DEFINE_integer("ppo_epochs", 3, "epochs per update")
flags.DEFINE_enum("agent_mode", ACMode.A2C, [ACMode.A2C, ACMode.PPO], "if should use A2C or PPO")

### NEW FLAGS ####
#flags.DEFINE_bool("render", True, "Whether to render with pygame.")
# point_flag.DEFINE_point("feature_screen_size", 32,
#                         "Resolution for screen feature layers.")
# point_flag.DEFINE_point("feature_minimap_size", 32,
#                         "Resolution for minimap feature layers.")
flags.DEFINE_integer("rgb_screen_size", 128,
                        "Resolution for rendered screen.") # type None if you want only features
# point_flag.DEFINE_point("rgb_minimap_size", 64,
#                         "Resolution for rendered minimap.") # type None if you want only features
# flags.DEFINE_enum("action_space", 'FEATURES', sc2_env.ActionSpace._member_names_,  # pylint: disable=protected-access, # type None if you want only features
#                   "Which action space to use. Needed if you take both feature "
#                   "and rgb observations.") # "RGB" or "FEATURES", None if only one is specified in the dimensions
# flags.DEFINE_bool("use_feature_units", True,
#                   "Whether to include feature units.")
# flags.DEFINE_bool("disable_fog", False, "Whether to disable Fog of War.")

flags.DEFINE_integer("max_agent_steps", 0, "Total agent steps.")
#flags.DEFINE_integer("game_steps_per_episode", None, "Game steps per episode.")
#flags.DEFINE_integer("max_episodes", 0, "Total episodes.")
#flags.DEFINE_integer("step_mul", 8, "Game steps per agent step.")

# flags.DEFINE_string("agent", "pysc2.agents.random_agent.RandomAgent",
#                     "Which agent to run, as a python path to an Agent class.")
# flags.DEFINE_enum("agent_race", "random", sc2_env.Race._member_names_,  # pylint: disable=protected-access
#                   "Agent 1's race.")
#
# flags.DEFINE_string("agent2", "Bot", "Second agent, either Bot or agent class.")
# flags.DEFINE_enum("agent2_race", "random", sc2_env.Race._member_names_,  # pylint: disable=protected-access
#                   "Agent 2's race.")
# flags.DEFINE_enum("difficulty", "very_easy", sc2_env.Difficulty._member_names_,  # pylint: disable=protected-access
#                   "If agent2 is a built-in Bot, it's strength.")

flags.DEFINE_bool("profile", False, "Whether to turn on code profiling.")
flags.DEFINE_bool("trace", False, "Whether to trace the code execution.")
#flags.DEFINE_integer("parallel", 1, "How many instances to run in parallel.")

flags.DEFINE_bool("save_replay", False, "Whether to save a replay at the end.")

#flags.DEFINE_string("map", None, "Name of a map to use.")


FLAGS(sys.argv)

#TODO this runner is maybe too long and too messy..
full_chekcpoint_path = os.path.join(FLAGS.checkpoint_path, FLAGS.model_name)

if FLAGS.training:
    full_summary_path = os.path.join(FLAGS.summary_path, FLAGS.model_name)
else:
    full_summary_path = os.path.join(FLAGS.summary_path, "no_training", FLAGS.model_name)


def check_and_handle_existing_folder(f):
    if os.path.exists(f):
        if FLAGS.if_output_exists == "overwrite":
            shutil.rmtree(f)
            print("removed old folder in %s" % f)
        elif FLAGS.if_output_exists == "fail":
            raise Exception("folder %s already exists" % f)


def _print(i):
    print(datetime.now())
    print("# batch %d" % i)
    sys.stdout.flush()


def _save_if_training(agent):
    agent.save(full_chekcpoint_path)
    agent.flush_summaries()
    sys.stdout.flush()

def make_custom_env(env_id, num_env, seed, wrapper_kwargs=None, start_index=0):
    """
    Create a wrapped, monitored SubprocVecEnv for Atari.
    """
    if wrapper_kwargs is None: wrapper_kwargs = {}
    def make_env(rank): # pylint: disable=C0111
        def _thunk():
            env = gym.make(env_id)
            env.seed(seed + rank)
            # Monitor should take care of reset!
            env = Monitor(env, logger.get_dir() and os.path.join(logger.get_dir(), str(rank)), allow_early_resets=False) # SUBPROC NEEDS 4 OUTPUS FROM STEP FUNCTION
            return env
        return _thunk
    #set_global_seeds(seed)
    return SubprocVecEnv([make_env(i + start_index) for i in range(num_env)])

def main():
    if FLAGS.training:
        check_and_handle_existing_folder(full_chekcpoint_path)
        check_and_handle_existing_folder(full_summary_path)

    #(MINE) Create multiple parallel environements (or a single instance for testing agent)
    if FLAGS.training and FLAGS.n_envs != 1:
        #envs = SubprocVecEnv((partial(make_sc2env, **env_args),) * FLAGS.n_envs)
        #envs = SubprocVecEnv([make_env(i,**env_args) for i in range(FLAGS.n_envs)])
        envs = make_custom_env('gridworld-v0', FLAGS.n_envs, 1)
    else:
        #envs = make_custom_env('gridworld-v0', 1, 1)
        envs = gym.make('gridworld-v0')

        #envs = SingleEnv(make_sc2env(**env_args))
    #envs = gym.make('gridworld-v0')
    # envs = SubprocVecEnv([make_env(i) for i in range(FLAGS.n_envs)])
    # envs = VecNormalize(env)
    # use for debugging 'Breakout-v0', Grid-v0, gridworld-v0
    #envs = VecFrameStack(make_custom_env('gridworld-v0', FLAGS.n_envs, 1), 1) # One is number of frames to stack within each env
    #envs = make_custom_env('gridworld-v0', FLAGS.n_envs, 1)
    print("Requested environments created successfully")
    #env = gym.make('gridworld-v0')
    tf.reset_default_graph()
    # The following lines fix the problem with using more than 2 envs!!!
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    #sess = tf.Session()

    agent = ActorCriticAgent(
        mode=FLAGS.agent_mode,
        sess=sess,
        spatial_dim=FLAGS.resolution, # Here you pass the resolution which is used in the step for the output probabilities map
        unit_type_emb_dim=5,
        loss_value_weight=FLAGS.loss_value_weight,
        entropy_weight_action_id=FLAGS.entropy_weight_action,
        entropy_weight_spatial=FLAGS.entropy_weight_spatial,
        scalar_summary_freq=FLAGS.scalar_summary_freq,
        all_summary_freq=FLAGS.all_summary_freq,
        summary_path=full_summary_path,
        max_gradient_norm=FLAGS.max_gradient_norm,
        num_actions=envs.action_space.n
    )
    # Build Agent
    agent.build_model()
    if os.path.exists(full_chekcpoint_path):
        agent.load(full_chekcpoint_path) #(MINE) LOAD!!!
    else:
        agent.init()
# (MINE) Define TIMESTEPS per episode (batch as each worker has its own episodes -- different timelines)
    # If it is not training you don't need that many steps. You need one to take the decision...Actually seem to be game steps
    if FLAGS.n_steps_per_batch is None:
        n_steps_per_batch = 128 if FLAGS.agent_mode == ACMode.PPO else 8
    else:
        n_steps_per_batch = FLAGS.n_steps_per_batch

    if FLAGS.agent_mode == ACMode.PPO:
        ppo_par = PPORunParams(
            FLAGS.ppo_lambda,
            batch_size=FLAGS.ppo_batch_size or n_steps_per_batch,
            n_epochs=FLAGS.ppo_epochs
        )
    else:
        ppo_par = None

    runner = Runner(
        envs=envs,
        agent=agent,
        discount=FLAGS.discount,
        n_steps=n_steps_per_batch,
        do_training=FLAGS.training,
        ppo_par=ppo_par
    )

    # runner.reset() # Reset env which means you get first observation. You need reset if you run episodic tasks!!! SC2 is not episodic task!!!

    if FLAGS.K_batches >= 0:
        n_batches = FLAGS.K_batches  # (MINE) commented here so no need for thousands * 1000
    else:
        n_batches = -1


    if FLAGS.training:
        i = 0

        try:
            runner.reset()
            while True:
                #runner.reset()
                if i % 500 == 0:
                    _print(i)
                if i % 4000 == 0:
                    _save_if_training(agent)
                runner.run_batch()  # (MINE) HERE WE RUN MAIN LOOP for while true
                #runner.run_batch_solo_env()
                i += 1
                if 0 <= n_batches <= i: #when you reach the certain amount of batches break
                    break
        except KeyboardInterrupt:
            pass
    else: # Test the agent
        # try:
        #     runner.reset_demo() # Cauz of differences in the arrangement of the dictionaries
        #     while runner.episode_counter <= (FLAGS.episodes - 1):
        #         #runner.reset()
        #         # You need the -1 as counting starts from zero so for counter 3 you do 4 episodes
        #         runner.run_trained_batch()  # (MINE) HERE WE RUN MAIN LOOP for while true
        # except KeyboardInterrupt:
        #     pass
        try:
            import pygame
            import time
            import random
            # pygame.font.get_fonts() # Run it to get a list of all system fonts
            display_w = 800
            display_h = 720

            BLUE = (128, 128, 255)
            DARK_BLUE = (1, 50, 130)
            RED = (255, 192, 192)
            BLACK = (0, 0, 0)
            WHITE = (255, 255, 255)

            pygame.init()
            gameDisplay = pygame.display.set_mode((display_w, display_h))
            gameDisplay.fill(DARK_BLUE)
            pygame.display.set_caption('Neural Introspection')
            clock = pygame.time.Clock()

            def screen_mssg_variable(text, variable, area):
                font = pygame.font.SysFont('arial', 16)
                txt = font.render(text + str(variable), True, WHITE)
                gameDisplay.blit(txt, area)
                #pygame.display.update()

            def process_img(img, x,y):
                # swap the axes else the image will come not the same as the matplotlib one
                img = img.transpose(1,0,2)
                surf = pygame.surfarray.make_surface(img)
                surf = pygame.transform.scale(surf, (300, 300))
                gameDisplay.blit(surf, (x, y))

            running = True
            while runner.episode_counter <= (FLAGS.episodes - 1) and running==True:
                print('Episode: ', runner.episode_counter)
                runner.reset_demo()  # Cauz of differences in the arrangement of the dictionaries
                map_xy = runner.envs.map_image
                map_alt = runner.envs.alt_view
                process_img(map_xy, 20, 20)
                process_img(map_alt, 20, 400)
                pygame.display.update()
                # Quit pygame if the (X) button is pressed on the top left of the window
                # Seems that without this for event quit doesnt show anything!!!
                # Also it seems that the pygame.event.get() is responsible to REALLY updating the screen contents
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False
                sleep(1.5)
                # Timestep counter
                t=0
                rewards = []
                done = 0
                while done==0:
                    # RUN THE MAIN LOOP
                    obs, action, value, reward, done = runner.run_trained_batch()

                    rewards.append(reward)
                    if done:
                        score = sum(rewards)
                        print(">>>>>>>>>>>>>>>>>>>>>>>>>>> episode %d ended in %d steps. Score %f" % (runner.episode_counter, t, score))
                        runner.episode_counter += 1

                    screen_mssg_variable("Value    : ", np.round(value,3), (168, 350))
                    screen_mssg_variable("Reward: ", np.round(reward,3), (168, 372))
                    pygame.display.update()
                    pygame.event.get()
                    sleep(1.5)

                    if action==15:
                        screen_mssg_variable("Package state:", runner.envs.package_state, (20, 350)) # The update of the text will be at the same time with the update of state
                        pygame.display.update()
                        pygame.event.get()  # Update the screen
                        sleep(1.5)

                    # BLIT!!!
                    # First Background covering everyything from previous session
                    gameDisplay.fill(DARK_BLUE)
                    map_xy = obs[0]['img']
                    map_alt = obs[0]['nextstepimage']
                    process_img(map_xy, 20, 20)
                    process_img(map_alt, 20, 400)
                    # Update finally the screen with all the images you blitted in the run_trained_batch
                    pygame.display.update() # Updates only the blitted parts of the screen, pygame.display.flip() updates the whole screen
                    pygame.event.get() # Show the last state and then reset
                    sleep(1.2)
                    t += 1
                clock.tick(15)
        except KeyboardInterrupt:
            pass

    print("Okay. Work is done")
    #_print(i)
    if FLAGS.training:
        _save_if_training(agent)
    if not FLAGS.training and FLAGS.save_replay:
        #envs.env.save_replay('/Users/constantinos/Documents/StarcraftMAC/MyAgents/')
        envs.env.save_replay('./Replays/MyAgents/')

    envs.close()


if __name__ == "__main__":
    main()
