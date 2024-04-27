import numpy as np
from sklearn.metrics import pairwise_distances as pair
from sklearn.preprocessing import normalize
def construct_graph(features, label, method, name, topk):

    fname = 'graph_{}_{}/{}_graph.txt'.format(method, topk, name)

    print('construct the graph of {}'.format(name))

    num = len(label)

    dist = None
    if method == 'heat':
        dist = -0.5 * pair(features, metric='manhattan') ** 2
        dist = np.exp(dist)

    elif method == 'cos':
        features[features > 0] = 1
        dist = np.dot(features, features.T)

    elif method == 'ncos':
        features[features > 0] = 1
        features = normalize(features, axis=1, norm='l1')
        dist = np.dot(features, features.T)

    elif method == 'p':
        y = features.T - np.mean(features.T)
        features = features - np.mean(features)
        dist = np.dot(features, features.T) / (np.linalg.norm(features) * np.linalg.norm(y))

    inds = []
    for i in range(dist.shape[0]):
        ind = np.argpartition(dist[i, :], -(topk + 1))[-(topk + 1):]
        inds.append(ind)

    f = open(fname, 'w')
    counter = 0
    A = np.zeros_like(dist)
    for i, v in enumerate(inds):
        mutual_knn = False
        for vv in v:
            if vv == i:
                pass
            else:
                if label[vv] != label[i]:
                    counter += 1
                f.write('{} {}\n'.format(i, vv))
    f.close()
    error = counter / (num * topk)
    print('finished {}'.format(name))
    return error

datalists = [

'10X_PBMC',
'human_kidney_counts',
'worm_neuron_cell',
]

ks = [
      5,
      ]
methods=['p']

for name in datalists:
    for topk in ks:
        for method in methods:
            file_out = open('graph_{}_{}/{}_error_rate.txt'.format(method, topk, name), 'a')
            File = ['data/{}.txt'.format(name), 'data/{}_label.txt'.format(name)]
            Para = [1024, 1e-3, 200, 5, 20]
            number = np.loadtxt(File[0], dtype=float)
            label = np.loadtxt(File[1], dtype=int)
            error = construct_graph(number, label, method, name, topk)
            print('{}'.format(name)+' {}'.format(method)+' {}'.format(topk)+' {}'.format(error), file=file_out)
            file_out.close()