import collections
import os
import numpy as np
import tensorflow as tf
from pysc2.lib import actions
from tensorflow.contrib import layers
from tensorflow.contrib.layers.python.layers.optimizers import OPTIMIZER_SUMMARIES
from actorcritic.policy import FullyConvPolicy
from common.preprocess import ObsProcesser, FEATURE_KEYS, AgentInputTuple
from common.util import weighted_random_sample, select_from_each_row, ravel_index_pairs
import tensorboard.plugins.beholder as beholder_lib

#LOG_DIRECTORY = '/tmp/beholder-demo/SCII'
LOG_DIRECTORY = '_files/summaries/Test'
def _get_placeholders(spatial_dim):
    sd = spatial_dim
    feature_list = [
        (FEATURE_KEYS.alt0_grass, tf.float32, [None, 20, 20]),
        (FEATURE_KEYS.alt0_bush, tf.float32, [None, 20, 20]),
    # FEATURE_KEYS.available_action_ids: get_available_actions_flags(obs),
        (FEATURE_KEYS.alt0_drone, tf.float32, [None, 20, 20]),
        (FEATURE_KEYS.alt0_hiker, tf.float32, [None, 20, 20]),
        (FEATURE_KEYS.alt1_pine, tf.float32, [None, 20, 20]),  # numpy.array is redundant
        (FEATURE_KEYS.alt1_pines, tf.float32, [None, 20, 20]),
        (FEATURE_KEYS.alt1_drone, tf.float32, [None, 20, 20]),
        (FEATURE_KEYS.alt2_drone, tf.float32, [None, 20, 20]),
        (FEATURE_KEYS.alt3_drone, tf.float32, [None, 20, 20]),
        (FEATURE_KEYS.minimap_numeric, tf.float32, [None, sd, sd, ObsProcesser.N_MINIMAP_CHANNELS]),
        (FEATURE_KEYS.screen_numeric, tf.float32, [None, sd, sd, ObsProcesser.N_SCREEN_CHANNELS]),
        (FEATURE_KEYS.screen_unit_type, tf.int32, [None, sd, sd]),
        (FEATURE_KEYS.is_spatial_action_available, tf.float32, [None]),
        (FEATURE_KEYS.available_action_ids, tf.float32, [None, len(actions.FUNCTIONS)]),
        (FEATURE_KEYS.selected_spatial_action, tf.int32, [None, 2]),
        (FEATURE_KEYS.selected_action_id, tf.int32, [None]),
        (FEATURE_KEYS.value_target, tf.float32, [None]),
        (FEATURE_KEYS.rgb_screen, tf.float32, [None, 50, 50, 3]),
        (FEATURE_KEYS.alt_view, tf.float32, [None, 50, 50, 3]),
        (FEATURE_KEYS.player_relative_screen, tf.int32, [None, sd, sd]),
        (FEATURE_KEYS.player_relative_minimap, tf.int32, [None, sd, sd]),
        (FEATURE_KEYS.advantage, tf.float32, [None]),
    ]
    return AgentInputTuple(
        **{name: tf.placeholder(dtype, shape, name) for name, dtype, shape in feature_list}
    )


class ACMode:
    A2C = "a2c"
    PPO = "ppo"


SelectedLogProbs = collections.namedtuple("SelectedLogProbs", ["action_id", "total"])


class ActorCriticAgent:
    _scalar_summary_key = "scalar_summaries"

    def __init__(self,
            sess: tf.Session,
            summary_path: str,
            all_summary_freq: int,
            scalar_summary_freq: int,
            spatial_dim: int,
            mode: str,
            clip_epsilon=0.2,
            unit_type_emb_dim=4,
            loss_value_weight=1.0,
            entropy_weight_spatial=1e-6,
            entropy_weight_action_id=1e-5,
            max_gradient_norm=None,
            optimiser="adam",
            optimiser_pars: dict = None,
            policy=FullyConvPolicy,
            num_actions=4
    ):
        """
        Actor-Critic Agent for learning pysc2-minigames
        https://arxiv.org/pdf/1708.04782.pdf
        https://github.com/deepmind/pysc2

        Can use
        - A2C https://blog.openai.com/baselines-acktr-a2c/ (synchronous version of A3C)
        or
        - PPO https://arxiv.org/pdf/1707.06347.pdf

        :param summary_path: tensorflow summaries will be created here
        :param all_summary_freq: how often save all summaries
        :param scalar_summary_freq: int, how often save scalar summaries
        :param spatial_dim: dimension for both minimap and screen
        :param mode: a2c or ppo
        :param clip_epsilon: epsilon for clipping the ratio in PPO (no effect in A2C)
        :param loss_value_weight: value weight for a2c update
        :param entropy_weight_spatial: spatial entropy weight for a2c update
        :param entropy_weight_action_id: action selection entropy weight for a2c update
        :param max_gradient_norm: global max norm for gradients, if None then not limited
        :param optimiser: see valid choices below
        :param optimiser_pars: optional parameters to pass in optimiser
        :param policy: Policy class
        """

        assert optimiser in ["adam", "rmsprop"]
        assert mode in [ACMode.A2C, ACMode.PPO]
        self.mode = mode
        self.sess = sess
        self.spatial_dim = spatial_dim
        self.loss_value_weight = loss_value_weight
        self.entropy_weight_spatial = entropy_weight_spatial
        self.entropy_weight_action_id = entropy_weight_action_id
        self.unit_type_emb_dim = unit_type_emb_dim
        self.summary_path = summary_path
        os.makedirs(summary_path, exist_ok=True)
        self.summary_writer = tf.summary.FileWriter(summary_path)
        self.all_summary_freq = all_summary_freq
        self.scalar_summary_freq = scalar_summary_freq
        self.train_step = 0
        self.max_gradient_norm = max_gradient_norm
        self.clip_epsilon = clip_epsilon
        self.policy = policy
        self.num_actions= num_actions

        opt_class = tf.train.AdamOptimizer if optimiser == "adam" else tf.train.RMSPropOptimizer
        if optimiser_pars is None:
            pars = {
                "adam": {
                    "learning_rate": 1e-4,
                    "epsilon": 5e-7
                },
                "rmsprop": {
                    "learning_rate": 2e-4
                }
            }[optimiser]
        else:
            pars = optimiser_pars
        self.optimiser = opt_class(**pars)

    def init(self):
        self.sess.run(self.init_op)
        if self.mode == ACMode.PPO:
            self.update_theta()

    def _get_select_action_probs(self, pi):
        action_id = select_from_each_row(
            pi.action_id_log_probs, self.placeholders.selected_action_id
        )
        # spatial = select_from_each_row(
        #     pi.spatial_action_log_probs, selected_spatial_action_flat
        # )
        # total = spatial + action_id
        total = action_id

        return SelectedLogProbs(action_id, total)

    def _scalar_summary(self, name, tensor):
        tf.summary.scalar(name, tensor,
            collections=[tf.GraphKeys.SUMMARIES, self._scalar_summary_key])

    def build_model(self):
        self.placeholders = _get_placeholders(self.spatial_dim)
        # Wtihin theta you build the policy net. Check the graph in tensoflow and expand theta to see the nets
        with tf.variable_scope("theta"):
            self.theta = self.policy(self, trainable=True).build() # (MINE) from policy.py you build the net. Theta is
            # actually the policy and contains the actions ids and spatial action dstrs

        # selected_spatial_action_flat = ravel_index_pairs(
        #     self.placeholders.selected_spatial_action, self.spatial_dim
        # )

        selected_log_probs = self._get_select_action_probs(self.theta)

        # maximum is to avoid 0 / 0 because this is used to calculate some means
        # sum_spatial_action_available = tf.maximum(
        #     1e-10, tf.reduce_sum(self.placeholders.is_spatial_action_available)
        # )

        # neg_entropy_spatial = tf.reduce_sum(
        #     self.theta.spatial_action_probs * self.theta.spatial_action_log_probs
        # ) / sum_spatial_action_available
        neg_entropy_action_id = tf.reduce_mean(tf.reduce_sum(self.theta.action_id_probs * self.theta.action_id_log_probs, axis=1))
        # neg_entropy_action_id = tf.reduce_sum(self.theta.action_id_probs * self.theta.action_id_log_probs, axis=1)
        # (MINE) Sample now actions from the corresponding dstrs defined by the policy network theta
        if self.mode == ACMode.PPO:
            # could also use stop_gradient and forget about the trainable
            with tf.variable_scope("theta_old"):
                theta_old = self.policy(self, trainable=False).build()

            new_theta_var = tf.global_variables("theta/")
            old_theta_var = tf.global_variables("theta_old/")

            assert len(tf.trainable_variables("theta/")) == len(new_theta_var)
            assert not tf.trainable_variables("theta_old/")
            assert len(old_theta_var) == len(new_theta_var)

            self.update_theta_op = [
                tf.assign(t_old, t_new) for t_new, t_old in zip(new_theta_var, old_theta_var)
            ]

            selected_log_probs_old = self._get_select_action_probs(theta_old)
            ratio = tf.exp(selected_log_probs.total - selected_log_probs_old.total)
            clipped_ratio = tf.clip_by_value(
                ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon
            )
            l_clip = tf.minimum(
                ratio * self.placeholders.advantage,
                clipped_ratio * self.placeholders.advantage
            )
            self.sampled_action_id = weighted_random_sample(theta_old.action_id_probs)
            self.sampled_spatial_action = weighted_random_sample(theta_old.spatial_action_probs)
            self.value_estimate = theta_old.value_estimate
            self._scalar_summary("action/ratio", tf.reduce_mean(clipped_ratio))
            self._scalar_summary("action/ratio_is_clipped",
                tf.reduce_mean(tf.to_float(tf.equal(ratio, clipped_ratio))))
            policy_loss = -tf.reduce_mean(l_clip)
        else:
            self.sampled_action_id = weighted_random_sample(self.theta.action_id_probs)
            # self.sampled_spatial_action = weighted_random_sample(self.theta.spatial_action_probs)
            self.value_estimate = self.theta.value_estimate
            policy_loss = -tf.reduce_mean(selected_log_probs.total * self.placeholders.advantage)
            #policy_loss = -tf.reduce_sum(selected_log_probs.total * self.placeholders.advantage)

        value_loss = tf.losses.mean_squared_error(self.placeholders.value_target, self.theta.value_estimate) # Target comes from runner/run_batch when you specify the full input
        # value_loss = tf.reduce_sum(tf.square(tf.reshape(self.placeholders.value_target,[-1]) - tf.reshape(self.value_estimate, [-1])))

        loss = (
            policy_loss
            + value_loss * self.loss_value_weight
            + neg_entropy_action_id * self.entropy_weight_action_id
        )

        self.train_op = layers.optimize_loss(
            loss=loss,
            global_step=tf.train.get_global_step(),
            optimizer=self.optimiser,
            clip_gradients=self.max_gradient_norm, # Caps the gradients at the value self.max_gradient_norm
            summaries=OPTIMIZER_SUMMARIES,
            learning_rate=None,
            name="train_op"
        )

        self._scalar_summary("value/estimate", tf.reduce_mean(self.value_estimate))
        self._scalar_summary("value/target", tf.reduce_mean(self.placeholders.value_target))
        # self._scalar_summary("action/is_spatial_action_available",
        #     tf.reduce_mean(self.placeholders.is_spatial_action_available))
        self._scalar_summary("action/selected_id_log_prob",
            tf.reduce_mean(selected_log_probs.action_id))
        self._scalar_summary("loss/policy", policy_loss)
        self._scalar_summary("loss/value", value_loss)
        #self._scalar_summary("loss/neg_entropy_spatial", neg_entropy_spatial)
        self._scalar_summary("loss/neg_entropy_action_id", neg_entropy_action_id)
        self._scalar_summary("loss/total", loss)
        self._scalar_summary("value/advantage", tf.reduce_mean(self.placeholders.advantage))
        self._scalar_summary("action/selected_total_log_prob",
            tf.reduce_mean(selected_log_probs.total))
        # self._scalar_summary("action/selected_spatial_log_prob",
        #     tf.reduce_sum(selected_log_probs.spatial) / sum_spatial_action_available)

        #tf.summary.image('convs output', tf.reshape(self.theta.map_output,[-1,25,25,64]))

        self.init_op = tf.global_variables_initializer()
        self.saver = tf.train.Saver(max_to_keep=2)
        self.all_summary_op = tf.summary.merge_all(tf.GraphKeys.SUMMARIES)
        self.scalar_summary_op = tf.summary.merge(tf.get_collection(self._scalar_summary_key))
        #self.beholder = beholder_lib.Beholder(logdir=LOG_DIRECTORY)
        #tf.summary.image('spatial policy', tf.reshape(self.theta.spatial_action_logits, [-1, 32, 32, 1]))

    def _input_to_feed_dict(self, input_dict):
        return {k + ":0": v for k, v in input_dict.items()} # Add the :0 after the name of each feature

    def step(self, obs):
        # (MINE) Pass the observations through the net
        feed_dict = self._input_to_feed_dict(obs)

        action_id, value_estimate, convs_im = self.sess.run(
            [self.sampled_action_id, self.value_estimate, self.theta.map_output],
            feed_dict=feed_dict
        )

        # spatial_action_2d = np.array(
        #     np.unravel_index(spatial_action, (self.spatial_dim,) * 2)
        # ).transpose()

        ##### BEHOLDER
        #activations= [np.reshape(images[0], (32, 32, 32)), np.reshape(images[1], (32, 32, 32)), np.reshape(images[2], (32, 32))]
        # spatial_policy_im = np.reshape(images[1], (32, 32))
        #image = np.reshape(convs_im[0], (13,13,64)) # DRONE
        # # Create 3 channel-image
        # spatial_policy_im = np.stack((spatial_policy_im)*3, -1).transpose()
        #activations= [images[0], images[1]]#, spatial_policy_im]
        # self.beholder.update(
        #     session=self.sess,
        #     arrays=image,#activations,# + [first_of_batch] + gradient_arrays
        #     frame=image,
        # )
        ########
        #self.summary_writer.add_summary(images[2]) # seems not working cauz of merging all

        return action_id, value_estimate

    def step_eval(self, obs):
        # (MINE) Pass the observations through the net
        ob = np.zeros((1, 50, 50, 3))
        obsb =np.zeros((1, 50, 50, 3))
        ob[0] = obs['rgb_screen']
        obsb[0] = obs['alt_view']

        feed_dict = {'rgb_screen:0' : ob,
                     'alt_view:0': obsb}

        action_id, value_estimate, representation = self.sess.run(
            [self.sampled_action_id, self.value_estimate, self.theta.map_output],
            feed_dict=feed_dict
        )

        return action_id, value_estimate, representation

    def train(self, input_dict):
        feed_dict = self._input_to_feed_dict(input_dict)
        ops = [self.train_op] # (MINE) From build model above the train_op contains all the operations for training

        write_all_summaries = (
            (self.train_step % self.all_summary_freq == 0) and
            self.summary_path is not None
        )
        write_scalar_summaries = (
            (self.train_step % self.scalar_summary_freq == 0) and
            self.summary_path is not None
        )

        if write_all_summaries:
            ops.append(self.all_summary_op)
        elif write_scalar_summaries:
            ops.append(self.scalar_summary_op)

        r = self.sess.run(ops, feed_dict)  # (MINE) TRAIN!!!

        if write_all_summaries or write_scalar_summaries:
            self.summary_writer.add_summary(r[-1], global_step=self.train_step)

        self.train_step += 1

    def get_value(self, obs):
        feed_dict = self._input_to_feed_dict(obs)
        return self.sess.run(self.value_estimate, feed_dict=feed_dict)

    def flush_summaries(self):
        self.summary_writer.flush()

    def save(self, path, step=None):
        os.makedirs(path, exist_ok=True)
        step = step or self.train_step
        print("saving model to %s, step %d" % (path, step))
        self.saver.save(self.sess, path + '/model.ckpt', global_step=step)

    def load(self, path):
        ckpt = tf.train.get_checkpoint_state(path)
        self.saver.restore(self.sess, ckpt.model_checkpoint_path)
        self.train_step = int(ckpt.model_checkpoint_path.split('-')[-1])
        print("loaded old model with train_step %d" % self.train_step)
        self.train_step += 1

    def update_theta(self):
        if self.mode == ACMode.PPO:
            self.sess.run(self.update_theta_op)
