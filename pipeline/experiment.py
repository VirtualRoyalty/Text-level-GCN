import gc
import keras
import tensorflow as tf
from spektral.data import BatchLoader
from spektral.transforms import GCNFilter, AdjToSpTensor

import neptune
from neptunecontrib.monitoring.keras import NeptuneMonitor

import model.gcn as model_gcn
import preprocess.utils as putils
import preprocess.general as pgeneral



class Experiment:

    def __init__(self, params=None):
        pass

    def run(self,
            NAME,
            tags,
            model,
            X_train_df,
            X_test_df,
            MIN_TF=1,
            MAX_TF=2500,
            MASTER_NODE=False,
            MAX_NODES_QUANTILE=1.0,
            WINDOW_SIZE=3,
            MODEL_TYPE='GCN',
            PMI=False
            EMB_TYPE=None,
            BATCH_SIZE=32,
            PATIENCE=10,
            EPOCH=100,
            DRAW=False,
            NEPTUNE=True,
            pretrained_weights=None,
            params=None,
            **kwargs
            ):



        vocab_terms, term2id = putils.build_vocab(X_train_df['tokens'],
                                                  min_tf=MIN_TF,
                                                  max_tf=MAX_TF,
                                                  master_node=MASTER_NODE)

        # print('VOCAB SIZE:', len(vocab_terms))

        MAX_NODES = putils.get_max_nodes(X_train_df, term2id, quantile=MAX_NODES_QUANTILE, master_node=MASTER_NODE)
        _ = putils.get_max_nodes(X_test_df, term2id, quantile=MAX_NODES_QUANTILE, master_node=MASTER_NODE)
        # print('MAX NODES:', MAX_NODES)

        X_train_df = X_train_df[X_train_df.n_unique > 1]
        X_test_df = X_test_df[X_test_df.n_unique > 1]


        try:
            del X_adj_test
            del X_adj_train
            del X_emb_test
            del X_emb_train
        except:
            pass


        X_adj_train, X_emb_train, Y_train = pgeneral.get_dataset_from_df(X_train_df,
                                                                         MAX_NODES,
                                                                         window_size=WINDOW_SIZE,
                                                                         term2id=term2id,
                                                                        )

        X_adj_test, X_emb_test, Y_test = pgeneral.get_dataset_from_df(X_test_df,
                                                                      MAX_NODES,
                                                                      window_size=WINDOW_SIZE,
                                                                      term2id=term2id,
                                                                    )
        gc.collect()

        train_spektral = pgeneral.CustomDataset(X_emb_train,
                                                X_adj_train,
                                                Y_train,
                                                transforms=[GCNFilter()]) #,  AdjToSpTensor()])
        test_spektral = pgeneral.CustomDataset(X_emb_test,
                                               X_adj_test,
                                               Y_test,
                                               transforms=[GCNFilter()])#,  AdjToSpTensor()])
        gc.collect()


        NUM_CLASSES =  train_spektral.n_labels


        train_loader = BatchLoader(train_spektral, batch_size=BATCH_SIZE)
        test_loader = BatchLoader(test_spektral, batch_size=BATCH_SIZE)
        lr_scheduler = keras.callbacks.LearningRateScheduler(model.scheduler)
        early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=PATIENCE)


        callbacks = list(lr_scheduler, early_stop)
        if DRAW:
            plot_losses = putils.PlotLosses()
            callbacks.append(plot_losses)
        if NEPTUNE:
            callbacks.append(NeptuneMonitor())

        model = model_gcn.GCN(pretrained_weights=pretrained_weights, **params)
        params['MODEL_PARAMS'] = model.model.count_params()
        params['LAYERS'] = [layer.name for layer in model.model.layers]

        if NEPTUNE:
            neptune.create_experiment(NAME, params=params, tags=tags)

        model.model.fit(train_loader.load(),
                        validation_data=test_loader,
                        validation_steps=test_loader.steps_per_epoch,
                        steps_per_epoch=train_loader.steps_per_epoch,
                        epochs=EPOCH,
                        callbacks=callbacks,
                        verbose=0
                  )
        neptune.stop()

        return