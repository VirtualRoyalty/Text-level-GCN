import networkx as nx

def doc2graph(doc,
              max_nodes,
              window_size=3,
              is_directed=True,
              is_weighted_edges=False,
              pmi_matrix=None,
              term2id=None,
              infranodus_weights=False):

    if is_directed:
        G = nx.DiGraph()
    else:
        G = nx.Graph()
    for i, term in enumerate(doc):
        if G.number_of_nodes() >= max_nodes:
            return G
        for j in range(1, window_size+1):
            try:
                next_term = doc[i + j]
                if not G.has_node(term):
                    G.add_node(term)
                    G.nodes[term]['count'] = 1
                else:
                    G.nodes[term]['count'] += 1
                if not G.has_node(next_term):
                    G.add_node(next_term)
                    G.nodes[next_term]['count'] = 0

                if not G.has_edge(term, next_term):
                    if pmi_matrix:
                        G.add_edge(term, next_term, weight=pmi_matrix[term2id[term]][term2id[next_term]])
                    elif infranodus_weights:
                        G.add_edge(term, next_term, weight=window_size-j+1)
                    else:
                        G.add_edge(term, next_term, weight=1)
                else:
                    if is_weighted_edges and not pmi_matrix:
                        G[term][next_term]['weight'] += 1
                    else:
                        pass
            except IndexError:
                if not G.has_node(term):
                    G.add_node(term)
                    G.nodes[term]['count'] = 1
                else:
                    G.nodes[term]['count'] += 1
    return G
