from multiprocessing import Process, Pipe
from pysc2.env import sc2_env, available_actions_printer
import os
from baselines import logger
from baselines.common import set_global_seeds
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
from baselines.bench import Monitor
import gym


class SingleEnv:
    """
    This works like SubprocVecEnv but runs only one environment in the main process
    """
    def __init__(self, env):
        self.env = env
        self.n_envs = 1

    def step(self, actions):
        """
        :param actions: List[FunctionCall]
        :return:
        """
        assert len(actions) == 1  # only 1 environment
        action = actions[0]
        return [self.env.step([action])[0]]

    def reset_done_envs(self):
        pass

    def reset(self):
        return [self.env.reset()[0]]

    def close(self):
        self.env.close()


# below (worker, CloudpickleWrapper, SubprocVecEnv) copied from
# https://github.com/openai/baselines/blob/master/baselines/common/vec_env/subproc_vec_env.py
# with some sc2 specific modifications
def worker(remote, env_fn_wrapper):
    """
    Handling the:
    action -> [action] and  [timestep] -> timestep
    single-player conversions here
    """
    #parent_remote.close()
    env = env_fn_wrapper.x()
    while True:
        cmd, action = remote.recv()
        if cmd == 'step':
            timesteps = env.step([action])
            assert len(timesteps) == 1
            remote.send(timesteps[0])
        elif cmd == 'reset':
            timesteps = env.reset()
            assert len(timesteps) == 1
            remote.send(timesteps[0])
        elif cmd == 'close':
            remote.close()
            break
        else:
            raise NotImplementedError


class CloudpickleWrapper(object):
    """
    Uses cloudpickle to serialize contents (otherwise multiprocessing tries to use pickle)
    """

    def __init__(self, x):
        self.x = x

    def __getstate__(self):
        import cloudpickle
        return cloudpickle.dumps(self.x)

    def __setstate__(self, ob):
        import pickle
        self.x = pickle.loads(ob)


class SubprocVecEnv:
    def __init__(self, env_fns):
        self.closed = False
        n_envs = len(env_fns)
        self.remotes, self.work_remotes = zip(*[Pipe() for _ in range(n_envs)])
        # self.ps = [Process(target=worker, args=(work_remote, remote, CloudpickleWrapper(env_fn)))
        #     for (work_remote, remote, env_fn) in zip(self.work_remotes, self.remotes, env_fns)]
        self.ps = [Process(target=worker, args=(work_remote, CloudpickleWrapper(env_fn)))
            for (work_remote, env_fn) in zip(self.work_remotes, env_fns)]
        for p in self.ps:
            p.daemon = True  # if the main process crashes, we should not cause things to hang
            p.start()
        # for remote in self.work_remotes:
        #     remote.close()

        self.n_envs = n_envs

    def _step_or_reset(self, command, actions=None):
        actions = actions or [None] * self.n_envs
        for remote, action in zip(self.remotes, actions):
            remote.send((command, action))
            # (MINE) Get all observations (timesteps) from all workers
        timesteps = [remote.recv() for remote in self.remotes]
        # for remote in self.remotes:
        #     t[remote] = self.remotes[remote].recv()
        return timesteps

    # def save_rep(self):
    #     self.save_replay('/Users/constantinos/Documents/StarcraftMAC/MyAgents/')

    def step(self, actions):
        return self._step_or_reset("step", actions)

    def reset(self):
        return self._step_or_reset("reset", None)

    def close(self):
        for remote in self.remotes:
            remote.send(('close', None))
        for p in self.ps:
            p.join()

    def reset_done_envs(self):
        pass


def make_sc2env(**kwargs):
    env = sc2_env.SC2Env(**kwargs)
    # env = available_actions_printer.AvailableActionsPrinter(env)
    return env

def make_env(rank,**kwargs):
        def _thunk():
            # agent_interface = features.parse_agent_interface_format(
            #     feature_screen=64,
            #     feature_minimap=64
            # )
            # env = sc2_env.SC2Env(map_name=map_name,
            #                      step_mul=step_mul,
            #                      agent_interface_format=agent_interface,
            #                      # screen_size_px=(screen_size, screen_size),
            #                      # minimap_size_px=(minimap_size, minimap_size),
            #                      visualize=False)
            env = sc2_env.SC2Env(**kwargs)
            return env
        return _thunk


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
    set_global_seeds(seed)
    return SubprocVecEnv([make_env(i + start_index) for i in range(num_env)])


