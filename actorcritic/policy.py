import tensorflow as tf
from pysc2.lib import actions
from pysc2.lib.features import SCREEN_FEATURES, MINIMAP_FEATURES
from tensorflow.contrib import layers


class FullyConvPolicy:
    """
    FullyConv network structure from https://arxiv.org/pdf/1708.04782.pdf
    Some implementation ideas are borrowed from https://github.com/xhujoy/pysc2-agents
    """

    def __init__(self,
            agent,
            trainable: bool = True
    ):
        # type agent: ActorCriticAgent
        self.placeholders = agent.placeholders
        self.trainable = trainable
        self.unittype_emb_dim = agent.unit_type_emb_dim
        self.num_actions = agent.num_actions

    def _build_convs(self, inputs, name):
        conv1 = layers.conv2d(
            inputs=inputs,
            data_format="NHWC",
            num_outputs=32,
            kernel_size=8,
            stride=4,
            padding='SAME',
            activation_fn=tf.nn.relu,
            scope="%s/conv1" % name,
            trainable=self.trainable
        )
        conv2 = layers.conv2d(
            inputs=conv1,
            data_format="NHWC",
            num_outputs=64,
            kernel_size=4,
            stride=1,#2,#
            padding='SAME',
            activation_fn=tf.nn.relu,
            scope="%s/conv2" % name,
            trainable=self.trainable
        )
        # conv3 = layers.conv2d(
        #     inputs=conv2,
        #     data_format="NHWC",
        #     num_outputs=64,
        #     kernel_size=3,
        #     stride=1,
        #     padding='SAME',
        #     activation_fn=tf.nn.relu,
        #     scope="%s/conv3" % name,
        #     trainable=self.trainable
        # )

        if self.trainable:
            layers.summarize_activation(conv1)
            layers.summarize_activation(conv2)
            # layers.summarize_activation(conv3)

        return conv2
        # return conv3

    def build(self):
        # units_embedded = layers.embed_sequence(
        #     self.placeholders.screen_unit_type,
        #     vocab_size=SCREEN_FEATURES.unit_type.scale, # 1850
        #     embed_dim=self.unittype_emb_dim, # 5
        #     scope="unit_type_emb",
        #     trainable=self.trainable
        # )
        #
        # # Let's not one-hot zero which is background
        # player_relative_screen_one_hot = layers.one_hot_encoding(
        #     self.placeholders.player_relative_screen,
        #     num_classes=SCREEN_FEATURES.player_relative.scale
        # )[:, :, :, 1:]
        # player_relative_minimap_one_hot = layers.one_hot_encoding(
        #     self.placeholders.player_relative_minimap,
        #     num_classes=MINIMAP_FEATURES.player_relative.scale
        # )[:, :, :, 1:]
        #
        #channel_axis = 2
        # alt0_all = tf.concat(
        #     [self.placeholders.alt0_grass, self.placeholders.alt0_bush, self.placeholders.alt0_drone, self.placeholders.alt0_hiker],
        #     axis=channel_axis
        # )
        # alt1_all = tf.concat(
        #     [self.placeholders.alt1_pine, self.placeholders.alt1_pines, self.placeholders.alt1_drone],
        #     axis=channel_axis
        # )
        # alt2_all = tf.concat(
        #     [self.placeholders.alt2_drone],
        #     axis=channel_axis
        # )
        # alt3_all = tf.concat(
        #     [self.placeholders.alt3_drone],
        #     axis=channel_axis
        # )

        # VOLUMETRIC APPROACH
        # alt_all = tf.concat(
        #     [self.placeholders.alt0_grass, self.placeholders.alt0_bush, self.placeholders.alt0_drone, self.placeholders.alt0_hiker,
        #      self.placeholders.alt1_pine, self.placeholders.alt1_pines, self.placeholders.alt1_drone, self.placeholders.alt2_drone,
        #      self.placeholders.alt3_drone],
        #     axis=channel_axis
        # )
        # self.spatial_action_logits = layers.conv2d(
        #     alt_all,
        #     data_format="NHWC",
        #     num_outputs=1,
        #     kernel_size=1,
        #     stride=1,
        #     activation_fn=None,
        #     scope='spatial_action',
        #     trainable=self.trainable
        # )
        # self.screen_output = self._build_convs(screen_numeric_all, "screen_network")
        # self.minimap_output = self._build_convs(minimap_numeric_all, "minimap_network")
        screen_px = tf.cast(self.placeholders.rgb_screen, tf.float32) / 255. # rgb_screen are integers (0-255) and here we convert to float and normalize
        alt_px = tf.cast(self.placeholders.alt_view, tf.float32) / 255.
        self.screen_output = self._build_convs(screen_px, "screen_network")
        self.alt_output = self._build_convs(alt_px, "alt_network")
        #minimap_px = tf.cast(self.placeholders.rgb_screen, tf.float32) / 255.
        # self.alt0_output = self._build_convs(alt0_all, "alt0_network")
        # self.alt1_output = self._build_convs(alt1_all, "alt1_network")
        # self.alt2_output = self._build_convs(alt2_all, "alt2_network")
        # self.alt3_output = self._build_convs(alt3_all, "alt3_network")

        # VOLUMETRIC APPROACH
        # self.alt0_output = self._build_convs(self.spatial_action_logits, "alt0_network")

        """(MINE) As described in the paper, the state representation is then formed by the concatenation
        of the screen and minimap outputs as well as the broadcast vector stats, along the channer dimension"""
        # State representation (last layer before separation as described in the paper)
        #self.map_output = tf.concat([self.alt0_output, self.alt1_output, self.alt2_output, self.alt3_output], axis=2)
        #self.map_output = tf.concat([self.alt0_output, self.alt1_output], axis=2)
        
        self.map_output = tf.concat([self.screen_output, self.alt_output], axis=2)

        #self.map_output = self.screen_output
        # The output layer (conv) of the spatial action policy with one ouput. So this means that there is a 1-1 mapping
        # (no filter that convolvues here) between layer and output. So eventually for every position of the layer you get
        # one value. Then you flatten it and you pass it into a softmax to get probs.
        # self.spatial_action_logits = layers.conv2d(
        #     self.map_output,
        #     data_format="NHWC",
        #     num_outputs=1,
        #     kernel_size=1,
        #     stride=1,
        #     activation_fn=None,
        #     scope='spatial_action',
        #     trainable=self.trainable
        # )
        #
        # spatial_action_probs = tf.nn.softmax(layers.flatten(self.spatial_action_logits))

        map_output_flat = layers.flatten(self.map_output)
        # (MINE) This is the last layer (fully connected -fc) for the non-spatial (categorical) actions
        self.fc1 = layers.fully_connected(
            map_output_flat,
            num_outputs=256,
            activation_fn=tf.nn.relu,
            scope="fc1",
            trainable=self.trainable
        )
        # (MINE) From the previous layer you extract action_id_probs (non spatial - categorical - actions) and value
        # estimate
        action_id_probs = layers.fully_connected(
            self.fc1,
            num_outputs=self.num_actions,#len(actions.FUNCTIONS),
            activation_fn=tf.nn.softmax,
            scope="action_id",
            trainable=self.trainable
        )
        value_estimate = tf.squeeze(layers.fully_connected(
            self.fc1,
            num_outputs=1,
            activation_fn=None,
            scope='value',
            trainable=self.trainable
        ), axis=1)

        # disregard non-allowed actions by setting zero prob and re-normalizing to 1 ((MINE) THE MASK)
        # action_id_probs *= self.placeholders.available_action_ids
        # action_id_probs /= tf.reduce_sum(action_id_probs, axis=1, keepdims=True)

        def logclip(x):
            return tf.log(tf.clip_by_value(x, 1e-12, 1.0))

        # spatial_action_log_probs = (
        #     logclip(spatial_action_probs)
        #     * tf.expand_dims(self.placeholders.is_spatial_action_available, axis=1)
        # )

        # non-available actions get log(1e-10) value but that's ok because it's never used
        action_id_log_probs = logclip(action_id_probs)

        self.value_estimate = value_estimate
        self.action_id_probs = action_id_probs
        #self.spatial_action_probs = spatial_action_probs
        self.action_id_log_probs = action_id_log_probs
       # self.spatial_action_log_probs = spatial_action_log_probs
        return self
