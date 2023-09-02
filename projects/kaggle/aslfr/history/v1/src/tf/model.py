#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ==============================================================================
#          \file   model.py
#        \author   chenghuige
#          \date   2023-06-26 15:47:48.601207
#   \Description
# ==============================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from gezi.common import *
import melt as mt
from src.config import *
from src.tf.preprocess import PreprocessLayer
from src.tf.util import *

# Initiailizers
INIT_HE_UNIFORM = tf.keras.initializers.he_uniform
INIT_GLOROT_UNIFORM = tf.keras.initializers.glorot_uniform
# Activations
GELU = tf.keras.activations.gelu

class ECA(tf.keras.layers.Layer):
    def __init__(self, kernel_size=5, **kwargs):
        super().__init__(**kwargs)
        self.supports_masking = True
        self.kernel_size = kernel_size
        self.conv = tf.keras.layers.Conv1D(1, kernel_size=kernel_size, strides=1, padding="same", use_bias=False)

    def call(self, inputs, mask=None):
        nn = tf.keras.layers.GlobalAveragePooling1D()(inputs, mask=mask)
        nn = tf.expand_dims(nn, -1)
        nn = self.conv(nn)
        nn = tf.squeeze(nn, -1)
        nn = tf.nn.sigmoid(nn)
        nn = nn[:,None,:]
        return inputs * nn

class CausalDWConv1D(tf.keras.layers.Layer):
    def __init__(self, 
        kernel_size=17,
        dilation_rate=1,
        use_bias=False,
        depthwise_initializer='glorot_uniform',
        name='', **kwargs):
        super().__init__(name=name,**kwargs)
        self.causal_pad = tf.keras.layers.ZeroPadding1D((dilation_rate*(kernel_size-1),0),name=name + '_pad')
        self.dw_conv = tf.keras.layers.DepthwiseConv1D(
                            kernel_size,
                            strides=1,
                            dilation_rate=dilation_rate,
                            padding='valid',
                            use_bias=use_bias,
                            depthwise_initializer=depthwise_initializer,
                            name=name + '_dwconv')
        self.supports_masking = True
        
    def call(self, inputs):
        x = self.causal_pad(inputs)
        x = self.dw_conv(x)
        return x

def Conv1DBlock(channel_size,
          kernel_size,
          dilation_rate=1,
          drop_rate=0.0,
          expand_ratio=2,
          se_ratio=0.25,
          activation='swish',
          name=None):
    '''
    efficient conv1d block, @hoyso48
    '''
    if name is None:
        name = str(tf.keras.backend.get_uid("mbblock"))
    # Expansion phase
    def apply(inputs):
        channels_in = tf.keras.backend.int_shape(inputs)[-1]
        channels_expand = channels_in * expand_ratio

        skip = inputs

        x = tf.keras.layers.Dense(
            channels_expand,
            use_bias=True,
            activation=activation,
            name=name + '_expand_conv')(inputs)

        # Depthwise Convolution
        x = CausalDWConv1D(kernel_size,
            dilation_rate=dilation_rate,
            use_bias=False,
            name=name + '_dwconv')(x)

        x = tf.keras.layers.BatchNormalization(momentum=0.95, name=name + '_bn')(x)

        x  = ECA()(x)

        x = tf.keras.layers.Dense(
            channel_size,
            use_bias=True,
            name=name + '_project_conv')(x)

        if drop_rate > 0:
            x = tf.keras.layers.Dropout(drop_rate, noise_shape=(None,1,1), name=name + '_drop')(x)

        if (channels_in == channel_size):
            x = tf.keras.layers.add([x, skip], name=name + '_add')
        return x

    return apply
  
class Conv1DLayer(tf.keras.layers.Layer):
  def __init__(self, units, ksize=17, **kwargs):
    super().__init__()
    self.units = units
    self.ksize = ksize
    self.conv = Conv1DBlock(dim,ksize,drop_rate=0.2)
    
  def call(self, x):
    dim = self.units
    ksize = self.ksize
    x = self.conv(x)
    x = self.conv(x)
    x = self.conv(x)

    return x

# Embeds a landmark using fully connected layers
class LandmarkEmbedding(tf.keras.Model):

  def __init__(self, units, name):
    super(LandmarkEmbedding, self).__init__(name=f'{name}_embedding')
    self.units = units

  def build(self, input_shape):
    # Embedding for missing landmark in frame, initizlied with zeros
    self.empty_embedding = self.add_weight(
        name=f'{self.name}_empty_embedding',
        shape=[self.units],
        initializer=FLAGS.emb_init,
    )
    # Embedding
    if FLAGS.landmark_emb == 'dense':
      self.emb = tf.keras.Sequential([
          tf.keras.layers.Dense(self.units * 2,
                                name=f'{self.name}_dense_1',
                                use_bias=False,
                                kernel_initializer=INIT_GLOROT_UNIFORM,
                                activation=GELU),
          tf.keras.layers.Dense(self.units,
                                name=f'{self.name}_dense_2',
                                use_bias=False,
                                kernel_initializer=INIT_HE_UNIFORM),
      ],
                                      name=f'{self.name}_dense')
    elif FLAGS.landmark_emb == 'conv':
      # self.emb = Conv1DLayer(self.units, ksize=17)
      ksize = 17
      self.emb = tf.keras.Sequential([
        tf.keras.layers.Conv1D(
          self.units, ksize, strides=1, padding="same", activation="relu"
        ),
        tf.keras.layers.Conv1D(
          self.units, ksize, strides=1, padding="same", activation="relu"
        ),
        tf.keras.layers.Conv1D(
          self.units, ksize, strides=1, padding="same", activation="relu"
        ),
      ])
    else:
      raise ValueError(f'Unknown landmark embedding type: {FLAGS.landmark_emb}')

  def call(self, x):
    if FLAGS.dominant_emb:
      return tf.where(
          # Checks whether landmark is missing in frame
          tf.reduce_sum(x, axis=2, keepdims=True) == 0,
          # If so, the empty embedding is used
          self.empty_embedding,
          # Otherwise the landmark data is embedded
          self.emb(x),
      )
    # ic(self.emb(x).shape)
    return self.emb(x)


# Creates embedding for each frame
class Embedding(tf.keras.Model):

  def __init__(self):
    super(Embedding, self).__init__()

  def build(self, input_shape):
    # Positional embedding for each frame index
    self.positional_embedding = self.add_weight(
        name=f'positional_embedding',
        shape=[FLAGS.n_frames, FLAGS.encoder_units],
        initializer=FLAGS.emb_init,
    )
    # Embedding layer for Landmarks
    self.landmark_embedding = LandmarkEmbedding(FLAGS.encoder_units,
                                                'landmark_embedding')

  def call(self, x, training=False):
    if FLAGS.norm_frames:
      # x = tf.where(
      #     tf.math.equal(x, 0.0),
      #     tf.constant(0.0, dtype=x.dtype),
      #     (x - tf.convert_to_tensor(STATS['means'], x.dtype)) / tf.convert_to_tensor(STATS['stds'], x.dtype),
      # )
      x = tf.where(
          tf.math.equal(x, 0.0),
          0.0,
          (x - STATS['means']) / STATS['stds'],
      )
    x = self.landmark_embedding(x)
    x = x + self.positional_embedding

    return x


# based on: https://stackoverflow.com/questions/67342988/verifying-the-implementation-of-multihead-attention-in-transformer
# replaced softmax with softmax layer to support masked softmax
def scaled_dot_product(q, k, v, softmax, attention_mask):
  #calculates Q . K(transpose)
  qkt = tf.matmul(q, k, transpose_b=True)
  #caculates scaling factor
  dk = tf.math.sqrt(tf.cast(q.shape[-1], dtype=q.dtype))
  # ic(q.dtype, dk.dtype)
  scaled_qkt = qkt / dk
  softmax = softmax(scaled_qkt, mask=attention_mask)
  z = tf.matmul(softmax, v)
  #shape: (m,Tx,depth), same shape as q,k,v
  return z


class MultiHeadAttention(tf.keras.layers.Layer):

  def __init__(self, d_model, num_of_heads, dropout=0., d_out=None):
    super(MultiHeadAttention, self).__init__()
    self.d_model = d_model
    self.num_of_heads = num_of_heads
    self.depth = d_model // num_of_heads
    self.depth = int(self.depth * FLAGS.mhatt_depth_ratio)
    self.wq = [
        tf.keras.layers.Dense(self.depth, use_bias=False)
        for i in range(num_of_heads)
    ]
    self.wk = [
        tf.keras.layers.Dense(self.depth, use_bias=False)
        for i in range(num_of_heads)
    ]
    self.wv = [
        tf.keras.layers.Dense(self.depth, use_bias=False)
        for i in range(num_of_heads)
    ]
    self.wo = tf.keras.layers.Dense(d_model if d_out is None else d_out,
                                    use_bias=False)
    self.softmax = tf.keras.layers.Softmax()
    self.drop = tf.keras.layers.Dropout(dropout)
    self.supports_masking = True

  def call(self, q, k, v, attention_mask=None, training=False):

    multi_attn = []
    for i in range(self.num_of_heads):
      Q = self.wq[i](q)
      K = self.wk[i](k)
      V = self.wv[i](v)
      multi_attn.append(
          scaled_dot_product(Q, K, V, self.softmax, attention_mask))

    multi_head = tf.concat(multi_attn, axis=-1)
    multi_head_attention = self.wo(multi_head)
    multi_head_attention = self.drop(multi_head_attention, training=training)

    return multi_head_attention


# Encoder based on multiple transformer blocks
class Encoder(tf.keras.Model):

  def __init__(self):
    super(Encoder, self).__init__(name='encoder')
    self.num_blocks = FLAGS.encoder_layers

  def build(self, input_shape):
    self.ln_1s = []
    self.mhas = []
    self.ln_2s = []
    self.mlps = []
    # Make Transformer Blocks
    for i in range(self.num_blocks):
      # First Layer Normalisation
      self.ln_1s.append(
          tf.keras.layers.LayerNormalization(epsilon=FLAGS.layer_norm_eps))
      # Multi Head Attention
      self.mhas.append(MultiHeadAttention(FLAGS.encoder_units, FLAGS.mhatt_heads, FLAGS.mhatt_drop))
      # Second Layer Normalisation
      self.ln_2s.append(
          tf.keras.layers.LayerNormalization(epsilon=FLAGS.layer_norm_eps))
      # Multi Layer Perception
      self.mlps.append(
          tf.keras.Sequential([
              tf.keras.layers.Dense(FLAGS.encoder_units * FLAGS.mlp_ratio,
                                    activation=GELU,
                                    kernel_initializer=INIT_GLOROT_UNIFORM),
              tf.keras.layers.Dropout(FLAGS.mlp_drop),
              tf.keras.layers.Dense(FLAGS.encoder_units,
                                    kernel_initializer=INIT_HE_UNIFORM),
          ]))

  def call(self, x, x_inp):
    if FLAGS.ignore_nan_frames:
      # Attention mask to ignore missing frames
      attention_mask = tf.where(
          tf.math.reduce_sum(x_inp, axis=[2]) == 0.0, 0.0, 1.0)
      attention_mask = tf.expand_dims(attention_mask, axis=1)
      attention_mask = tf.repeat(attention_mask, repeats=FLAGS.n_frames, axis=1)
    else:
      attention_mask = None
    
    # Iterate input over transformer blocks
    for ln_1, mha, ln_2, mlp in zip(self.ln_1s, self.mhas, self.ln_2s,
                                    self.mlps):
      x = ln_1(x + mha(x, x, x, attention_mask=attention_mask))
      x = ln_2(x + mlp(x))

    return x


# Decoder based on multiple transformer blocks
class Decoder(tf.keras.Model):

  def __init__(self):
    super().__init__(name='decoder')
    self.num_blocks = FLAGS.decoder_layers

  def build(self, input_shape):
    self.positional_embedding = self.add_weight(
        name=f'positional_embedding',
        shape=[FLAGS.n_frames, FLAGS.decoder_units],
        initializer=FLAGS.emb_init,
    )
    # Character Embedding
    self.char_emb = tf.keras.layers.Embedding(
        VOCAB_SIZE, FLAGS.decoder_units, embeddings_initializer=FLAGS.emb_init)
    # Positional Encoder MHA
    self.pos_emb_mha = MultiHeadAttention(FLAGS.decoder_units, FLAGS.mhatt_heads, FLAGS.mhatt_drop)
    self.pos_emb_ln = tf.keras.layers.LayerNormalization(
        epsilon=FLAGS.layer_norm_eps)
    # First Layer Normalisation
    self.ln_1s = []
    self.mhas = []
    self.ln_2s = []
    self.mlps = []
    # Make Transformer Blocks
    for i in range(self.num_blocks):
      # First Layer Normalisation
      self.ln_1s.append(
          tf.keras.layers.LayerNormalization(epsilon=FLAGS.layer_norm_eps))
      # Multi Head Attention
      self.mhas.append(MultiHeadAttention(FLAGS.decoder_units, FLAGS.mhatt_heads, FLAGS.mhatt_drop))
      # Second Layer Normalisation
      self.ln_2s.append(
          tf.keras.layers.LayerNormalization(epsilon=FLAGS.layer_norm_eps))
      # Multi Layer Perception
      self.mlps.append(
          tf.keras.Sequential([
              tf.keras.layers.Dense(FLAGS.decoder_units * FLAGS.mlp_ratio,
                                    activation=GELU,
                                    kernel_initializer=INIT_GLOROT_UNIFORM),
              tf.keras.layers.Dropout(FLAGS.mlp_drop),
              tf.keras.layers.Dense(FLAGS.decoder_units,
                                    kernel_initializer=INIT_HE_UNIFORM),
          ]))

  def get_causal_attention_mask(self, B):
    i = tf.range(FLAGS.n_frames)[:, tf.newaxis]
    j = tf.range(FLAGS.n_frames)
    mask = tf.cast(i >= j, dtype=tf.int32)
    mask = tf.reshape(mask, (1, FLAGS.n_frames, FLAGS.n_frames))
    mult = tf.concat(
        [tf.expand_dims(B, -1),
         tf.constant([1, 1], dtype=tf.int32)],
        axis=0,
    )
    mask = tf.tile(mask, mult)
    mask = tf.cast(mask, tf.float32)
    # mask = tf.cast(mask, tf.float32) if not FLAGS.fp16 else tf.cast(mask, tf.float16)
    return mask

  def call(self, encoder_outputs, phrase):
    # Batch Size
    B = tf.shape(encoder_outputs)[0]
    # Cast to INT32
    phrase = tf.cast(phrase, tf.int32)
    phrase = phrase[:, :-1]
    # Prepend SOS Token
    phrase = tf.pad(phrase, [[0, 0], [1, 0]],
                    constant_values=SOS_TOKEN,
                    name='prepend_sos_token')
    # # Pad With PAD Token
    phrase = tf.pad(phrase,
                    [[0, 0], [0, FLAGS.n_frames - FLAGS.max_phrase_len]],
                    constant_values=PAD_TOKEN,
                    name='append_pad_token')
    # Positional Embedding
    # ic(self.positional_embedding.shape, phrase.shape, self.char_emb(phrase).shape)
    x = self.positional_embedding + self.char_emb(phrase)
    # Causal Attention
    causal_mask = self.get_causal_attention_mask(B)
    # ic(causal_mask)
    causal_mask = tf.cast(causal_mask, x.dtype)
    # ic(x.dtype, causal_mask.dtype)
    # ic(causal_mask.shape)
    x = self.pos_emb_ln(x +
                        self.pos_emb_mha(x, x, x, attention_mask=causal_mask))
    # Iterate input over transformer blocks
    for ln_1, mha, ln_2, mlp in zip(self.ln_1s, self.mhas, self.ln_2s,
                                    self.mlps):
      x = ln_1(
          x +
          mha(x, encoder_outputs, encoder_outputs, attention_mask=causal_mask))
      x = ln_2(x + mlp(x))
    # Slice 31 Characters
    x = tf.slice(x, [0, 0, 0], [-1, FLAGS.max_phrase_len, -1])

    return x


class Model(mt.Model):

  def __init__(self, **kwargs):
    super().__init__(**kwargs)

    self.embedding = Embedding()
    self.encoder = Encoder()
    # from src.tf.rnn import Encoder as RNN_Encoder
    # self.encoder = RNN_Encoder()
    if FLAGS.rnn_encoder:
      RNN = getattr(tf.keras.layers, FLAGS.rnn)
      self.rnn_encoder = tf.keras.Sequential(
        [
          tf.keras.layers.Dropout(FLAGS.rnn_drop),
          tf.keras.layers.Bidirectional(
            RNN(FLAGS.encoder_units, return_sequences=True))
        ], name='rnn_encoder')
    if FLAGS.use_decoder:
      self.decoder = Decoder()
    self.classifer = tf.keras.Sequential(
        [
            tf.keras.layers.Dropout(FLAGS.cls_drop),
            tf.keras.layers.Dense(
                VOCAB_SIZE,
                activation=None,
                kernel_initializer=INIT_HE_UNIFORM),
        ],
        name='classifier')

  def forward(self, frames, phrase):
    x = self.encode(frames)
    x = self.decode(x, phrase)
    return x

  @tf.function()
  def preprocess(self, frames):
    frames = tf.reshape(frames, (-1, FLAGS.n_frames, N_COLS))
    # frames = tf.reshape(frames, (tf.shape(frames)[0], -1, N_COLS))
    return frames

  def encode(self, frames):
    x = self.embedding(frames)
    x = self.encoder(x, frames)
    # if not FLAGS.use_decoder:
    #   x = tf.reshape(x, (-1, FLAGS.max_phrase_len, FLAGS.n_frames // FLAGS.max_phrase_len, FLAGS.decoder_units))
    #   x = tf.reduce_mean(x, axis=2)
    if FLAGS.rnn_encoder:
      x = self.rnn_encoder(x)
    return x

  def decode(self, x, phrase):
    if FLAGS.use_decoder:
      x = self.decoder(x, phrase)
    x = self.classifer(x)
    if not FLAGS.use_decoder:
      x = x[:,:FLAGS.max_phrase_len]
    return x

  ## TODO why could not use tf.function here ? bad results after training a few epochs
  # @tf.function()
  def call(self, inputs, training=False):
    if FLAGS.work_mode == 'train':
      self.input_ = inputs
    frames = inputs['frames']
    frames = self.preprocess(frames)
    phrase = inputs['phrase']

    x = self.forward(frames, phrase)

    return x

  def get_loss_fn(self):

    def loss_fn(y_true, y_pred, x):
      y_pred = tf.cast(y_pred, tf.float32)
      # One Hot Encode Sparsely Encoded Target Sign
      y_true = tf.cast(y_true, tf.int32)

      mask = y_true != PAD_TOKEN
      # mask_ = tf.argmax(y_pred, axis=-1) != PAD_TOKEN
      # mask = tf.math.logical_or(mask, mask_)

      y_true = tf.one_hot(y_true, VOCAB_SIZE, axis=2)
      y_true = y_true[:, :FLAGS.max_phrase_len, :]
      # y_pred = y_pred[:, :FLAGS.max_phrase_len, :]
      # Categorical Crossentropy with native label smoothing support
      loss_obj = tf.keras.losses.CategoricalCrossentropy(
          from_logits=True,
          label_smoothing=FLAGS.label_smoothing,
          reduction='none')
      loss = loss_obj(y_true, y_pred)
      
      if FLAGS.mix_sup:
        loss *= x['weight']

      if FLAGS.weighted_loss:
        weights = tf.tile(tf.range(FLAGS.max_phrase_len)[None], [tf.shape(y_true)[0], 1])
        weights = FLAGS.max_phrase_len + 1 - weights
        weights = tf.cast(weights, tf.float32)
        if FLAGS.log_weights:
          weights = tf.math.log(weights + 1.)
        loss = loss * weights

      if FLAGS.masked_loss:
        loss = mt.masked_loss(loss, mask)
      else:
        loss = tf.reduce_mean(loss)
      return loss

    loss_fn = self.loss_wrapper(loss_fn)
    return loss_fn

  @tf.function()
  def infer(self, frames):
    x = self.encode(frames)
    
    phrase = tf.fill([tf.shape(frames)[0], FLAGS.max_phrase_len], SOS_TOKEN)
    if FLAGS.use_decoder:
      for idx in tf.range(FLAGS.max_phrase_len):
        # phrase = tf.cast(phrase, tf.int8)
        outputs = self.decode(x, phrase)
        # phrase = tf.cast(phrase, tf.int32)
        phrase = tf.where(
            tf.range(FLAGS.max_phrase_len) < idx + 1,
            tf.argmax(outputs, axis=2, output_type=tf.int32),
            phrase,
        )
    else:
      # phrase = tf.cast(phrase, tf.int8)
      outputs = self.decode(x, phrase)
      # phrase = tf.cast(phrase, tf.int32)
      phrase = tf.argmax(outputs, axis=2, output_type=tf.int32)

    # Squeeze outputs
    phrase = tf.one_hot(phrase, VOCAB_SIZE)

    # Return a dictionary with the output tensor
    return phrase

  # TODO not work
  # TypeError: You are passing KerasTensor(type_spec=TensorSpec(shape=(None, 128, 164), dtype=tf.float32, name='frames'), name='frames',
  #  description="created by layer 'frames'"), an intermediate Keras symbolic input/output, to a TF API that does not allow registering custom
  #  dispatchers, such as `tf.cond`, `tf.function`, gradient tapes, or `tf.map_fn`. Keras Functional model construction only supports TF API calls that
  #  *do* support dispatching, such as `tf.math.add` or `tf.reshape`. Other APIs cannot be called directly on symbolic Kerasinputs/outputs. You can work
  #  around this limitation by putting the operation in a custom Keras layer `call` and calling that layer on this symbolic input/output.
  def infer_model(self):
    frames_inp = tf.keras.layers.Input([FLAGS.n_frames, N_COLS],
                                       dtype=tf.float32,
                                       name='frames')
    out = self.infer(frames_inp)
    model = tf.keras.models.Model(frames_inp, out)
    return model


# TFLite model for submission
class TFLiteModel(tf.keras.Model):

  def __init__(self, model):
    super(TFLiteModel, self).__init__()

    # Load the feature generation and main models
    self.preprocess_layer = PreprocessLayer(FLAGS.n_frames)
    self.model = model

  @tf.function(jit_compile=True)
  def encode(self, frames):
    return self.model.encode(frames)

  @tf.function(jit_compile=True)
  def decode(self, x, phrase):
    return self.model.decode(x, phrase)

  @tf.function(input_signature=[
      tf.TensorSpec(shape=[None, N_COLS], dtype=tf.float32, name='inputs')
  ])
  def call(self, inputs):
    # Number Of Input Frames
    N_INPUT_FRAMES = tf.shape(inputs)[0]
    # Preprocess Data
    frames_inp = self.preprocess_layer(inputs)
    # Add Batch Dimension
    frames_inp = tf.expand_dims(frames_inp, axis=0)
    # Get Encoding
    encoding = self.encode(frames_inp)
    # Make Prediction
    phrase = tf.fill([1, FLAGS.max_phrase_len], SOS_TOKEN)

    if FLAGS.use_decoder:
      # Predict One Token At A Time
      # stop = False
      stop = tf.constant(False)
      for idx in tf.range(FLAGS.max_phrase_len):
        # Cast phrase to int8
        # phrase = tf.cast(phrase, tf.int8)
        # If EOS token is predicted, stop predicting
        outputs = tf.cond(
            stop, 
            lambda: tf.one_hot(tf.cast(phrase, tf.int32), VOCAB_SIZE),
            lambda: self.decode(encoding, phrase))
        # phrase = tf.cast(phrase, tf.int32)
        phrase = tf.where(
            tf.range(FLAGS.max_phrase_len) < idx + 1,
            tf.argmax(outputs, axis=2, output_type=tf.int32),
            phrase,
        )

        stop = tf.cond(stop, 
                       lambda: stop, 
                       lambda: phrase[0, idx] == EOS_TOKEN)
    else:
      # Cast phrase to int8
      # phrase = tf.cast(phrase, tf.int8)
      outputs = self.decode(encoding, phrase)
      # phrase = tf.cast(phrase, tf.int32)
      phrase = tf.argmax(outputs, axis=2, output_type=tf.int32)

    # Squeeze outputs
    outputs = tf.squeeze(phrase, axis=0)
    outputs = tf.one_hot(outputs, VOCAB_SIZE)
    if FLAGS.decode_phrase_type:
      ouputs = outputs[1:]

    # Return a dictionary with the output tensor
    return {'outputs': outputs}
