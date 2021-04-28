import tensorflow.keras as keras

from tensorflow.keras import layers, Input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, Flatten, Embedding, Reshape, BatchNormalization, Lambda
from tensorflow.keras.layers.experimental import EinsumDense

from keras.regularizers import l2

from spektral.layers import GCNConv
from spektral.layers import GlobalSumPool, GlobalMaxPool,  GlobalAvgPool , GlobalAttentionPool, GlobalAttnSumPool
from spektral.layers import MinCutPool, TopKPool, SAGPool, SortPool


class MPNN:

  def __init__(self,
               VOCAB_SIZE,
               EMB_SIZE,
               NUM_CLASSES,
               FIRST_CONV_SIZE,
               FIRST_CONV_ACTIVATION,
               DROPOUT_RATE,
               L2_RATE,
               LR,
               LR_FIRST_UPDATE,
               LR_UPDATE_PER,
               POOLING,
               BATCH_NORM,
               PRETRAINED,
               pretrained_weights=pretrained_weights,
               **kwargs):

    self.lr_first_update = LR_FIRST_UPDATE
    self.lr_update_per_epoch = LR_UPDATE_PER

    n_node_features = train_spektral.n_node_features  # Dimension of node features
    n_out = train_spektral.n_labels                   # Dimension of the target

    X_in = Input(shape=(MAX_NODES, n_node_features), name="Nodes_input")
    A_in = Input(shape=(MAX_NODES, MAX_NODES), sparse=False, name='Adj_input')

    if PRETRAINED:
      X_1 = Embedding(input_dim=VOCAB_SIZE,
                      output_dim=EMB_SIZE,
                      weights=[pretrained_weights],
                      input_length=MAX_NODES, name='Embeddings')(X_in)
    else:
      X_1 = Embedding(input_dim=VOCAB_SIZE,
                      output_dim=EMB_SIZE,
                      input_length=MAX_NODES, name='Embeddings')(X_in)

    X_1 = Reshape(target_shape=(MAX_NODES, EMB_SIZE), name='Reshape')(X_1)

    if BATCH_NORM:
      X_1 = BatchNormalization(name='BatchNorm')(X_1)

    MP = Lambda(ein, arguments={'equation': '...ij,...jk->...ijk'})([A_in, X_1])
    # X_1 = tf.keras.layers.GlobalMaxPool1D()(MP)
    X_1 = GlobalMaxPool(name='MaxPool')(MP)

    if POOLING == 'GlobalSum':
      X_1 = GlobalSumPool(name='Pooling')(X_1)
    elif POOLING == 'GlobalMax':
      X_1 = GlobalMaxPool(name='Pooling')(X_1)
    elif POOLING == 'GlobalAvg':
      X_1 = GlobalAvgPool(name='Pooling')(X_1)
    else:
      X_1 = GlobalSumPool(name='Pooling')(X_1)

    X_1 = Dropout(DROPOUT_RATE, name='Dropout')(X_1)

    output = Dense(n_out, activation="softmax", kernel_regularizer=l2(L2_RATE), name='Dense')(X_1)

    self.model = Model(inputs=[X_in, A_in], outputs=output)
    _optimizer = tf.optimizers.Adam(learning_rate=LR)
    self.model.compile(optimizer=_optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    self.model.summary()
    return


  def scheduler(self, epoch, lr):
      if epoch % self.lr_first_update == 0:
          self.lr_first_update += self.lr_update_per_epoch
          return  lr / 2
      else:
          return lr
