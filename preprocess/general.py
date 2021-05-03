import numpy as np
import tensorflow as tf
import networkx as nx
from tqdm import tqdm, tqdm_notebook

from preprocess.graph import doc2graph


def prepare2gcn(doc,
                max_nodes,
                window_size,
                term2id,
                is_directed=True,
                is_weighted_edges=False,
                infranodus_weights=False,
                MASTER_NODE=False,
                pmi_matrix=None):

    #   doc = doc[-max_nodes:] #doc[:max_nodes]
    # assert len(set(doc)) <= max_nodes

    doc = [token for token in doc if token in term2id]
    G = doc2graph(doc,
                  max_nodes=max_nodes,
                  window_size=window_size,
                  term2id=term2id,
                  pmi_matrix=pmi_matrix,
                  is_directed=is_directed,
                  is_weighted_edges=is_weighted_edges,
                  infranodus_weights=False)
    A = nx.adjacency_matrix(G).todense()
    padded = np.zeros((max_nodes, max_nodes),  dtype='float32')
    padded[:A.shape[0], :A.shape[1]] = A

    embs = []
    if MASTER_NODE:
      embs.append(term2id['MASTER_NODE'])
      padded = np.concatenate((np.ones(padded.shape[1]), b), axis=0)
    for i, token in enumerate(list(G.nodes())):
      vec = term2id[token]
      embs.append(vec)
    for i in range(max_nodes-len(G.nodes())):
      embs.append(term2id['PAD'])
    embs = np.array(embs, dtype='int32')

    return padded, embs


def get_dataset_from_df(df,
                        max_nodes,
                        term2id,
                        window_size=3,
                        token_col='tokens',
                        label_col='label',
                        pmi_matrix=None,
                        is_directed=is_directed,
                        is_weighted_edges=is_weighted_edges,
                        infranodus_weights=False):

  X_adj, X_emb, Y = list(), list(),  list()
  for i in tqdm_notebook(range(len(df))):
    tokens  = df[token_col].iloc[i]
    target  = df[label_col].iloc[i]
    if len(tokens) > 1:
        A, embs = prepare2gcn(tokens,
                              max_nodes=max_nodes,
                              window_size=window_size,
                              term2id=term2id,
                              pmi_matrix=pmi_matrix,
                              is_directed=is_directed,
                              is_weighted_edges=is_weighted_edges,
                              infranodus_weights=False)
      if pmi_matrix is not None:
        np.fill_diagonal(A, 1)
      X_adj.append(A.astype('float32'))
      X_emb.append(embs.astype('int32'))
      Y.append(target)

  X_adj = np.array(X_adj, dtype='float32')
  X_emb = np.array(X_emb, dtype='int32')
  # X_emb = np.expand_dims(X_emb, axis=-1)

  Y = np.array(Y)
  Y = tf.one_hot(Y, df[label_col].nunique(), dtype='float32')
  return X_adj, X_emb, Y


from spektral.data import BatchLoader, Dataset, Graph,  PackedBatchLoader

class CustomDataset(Dataset):

    def __init__(self, emb, adj, y, **kwargs):
        self.emb = emb
        self.adj = adj
        self.y = y

        super().__init__(**kwargs)

    def read(self):
        return [Graph(x=emb.reshape(-1, 1), a=adj, y=y) for emb, adj, y in zip(self.emb, self.adj, self.y)]
