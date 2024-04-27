from __future__ import print_function, division
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
import argparse
import numpy as np
from sklearn.cluster import KMeans
import torch
torch.cuda.current_device()
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.optim import Adam
from torch.nn import Linear
from torch.utils.data import DataLoader, TensorDataset
import h5py
import scanpy as sc
from datetime import datetime
import time
import sys
import os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)
from utils import load_data, load_graph
from evaluation import eva
from preprocess import read_dataset, normalize
from layers import ZINBLoss, MeanAct, DispAct
from GNN import GNNLayer
import utilsForMain
import pandas as pd
import scipy as sp

class AE(nn.Module):
    def __init__(self, n_enc_1, n_enc_2, n_enc_3, n_dec_1, n_dec_2, n_dec_3,
                 n_input, n_z):
        super(AE, self).__init__()
        # encoder
        self.enc_1 = Linear(n_input, n_enc_1)
        self.BN1 = nn.BatchNorm1d(n_enc_1)
        self.enc_2 = Linear(n_enc_1, n_enc_2)
        self.BN2 = nn.BatchNorm1d(n_enc_2)
        self.enc_3 = Linear(n_enc_2, n_enc_3)
        self.BN3 = nn.BatchNorm1d(n_enc_3)
        self.z_layer = Linear(n_enc_3, n_z)

        # decoder
        self.dec_1 = Linear(n_z, n_dec_1)
        self.BN4 = nn.BatchNorm1d(n_dec_1)
        self.dec_2 = Linear(n_dec_1, n_dec_2)
        self.BN5 = nn.BatchNorm1d(n_dec_2)
        self.dec_3 = Linear(n_dec_2, n_dec_3)
        self.BN6 = nn.BatchNorm1d(n_dec_3)
        self.x_bar_layer = Linear(n_dec_3, n_input)
    def forward(self, x):
        enc_h1 = F.relu(self.BN1(self.enc_1(x)))
        enc_h2 = F.relu(self.BN2(self.enc_2(enc_h1)))
        enc_h3 = F.relu(self.BN3(self.enc_3(enc_h2)))

        z = self.z_layer(enc_h3)

        dec_h1 = F.relu(self.BN4(self.dec_1(z)))
        dec_h2 = F.relu(self.BN5(self.dec_2(dec_h1)))
        dec_h3 = F.relu(self.BN6(self.dec_3(dec_h2)))
        x_bar = self.x_bar_layer(dec_h3)

        return x_bar, enc_h1, enc_h2, enc_h3, z, dec_h3

class MLP_L(nn.Module):
    def __init__(self, n_mlp):
        super(MLP_L, self).__init__()
        self.wl = Linear(n_mlp, 5)

    def forward(self, mlp_in):
        weight_output = F.softmax(F.leaky_relu(self.wl(mlp_in)), dim=1)

        return weight_output


class MLP_1(nn.Module):
    def __init__(self, n_mlp):
        super(MLP_1, self).__init__()
        self.w1 = Linear(n_mlp, 2)

    def forward(self, mlp_in):
        weight_output = F.softmax(F.leaky_relu(self.w1(mlp_in)), dim=1)

        return weight_output


class MLP_2(nn.Module):
    def __init__(self, n_mlp):
        super(MLP_2, self).__init__()
        self.w2 = Linear(n_mlp, 2)

    def forward(self, mlp_in):
        weight_output = F.softmax(F.leaky_relu(self.w2(mlp_in)), dim=1)

        return weight_output


class MLP_3(nn.Module):
    def __init__(self, n_mlp):
        super(MLP_3, self).__init__()
        self.w3 = Linear(n_mlp, 2)

    def forward(self, mlp_in):
        weight_output = F.softmax(F.leaky_relu(self.w3(mlp_in)), dim=1)

        return weight_output
class MFGNN(nn.Module):
    def __init__(self, n_enc_1, n_enc_2, n_enc_3, n_dec_1, n_dec_2, n_dec_3,
                 n_input, n_z, n_clusters, v=1):
        super(MFGNN, self).__init__()
        # AE to obtain internal information
        self.ae = AE(
            n_enc_1=n_enc_1,
            n_enc_2=n_enc_2,
            n_enc_3=n_enc_3,
            n_dec_1=n_dec_1,
            n_dec_2=n_dec_2,
            n_dec_3=n_dec_3,
            n_input=n_input,
            n_z=n_z)

        self.ae.load_state_dict(torch.load(args.pretrain_path, map_location='cpu'))
        self.agcn_0 = GNNLayer(n_input, n_enc_1)
        self.agcn_1 = GNNLayer(n_enc_1, n_enc_2)
        self.agcn_2 = GNNLayer(n_enc_2, n_enc_3)
        self.agcn_3 = GNNLayer(n_enc_3, n_z)
        self.agcn_z = GNNLayer(3020,n_clusters)
        self.mlp = MLP_L(3020)
        self.mlp1 = MLP_1(2*n_enc_1)
        self.mlp2 = MLP_2(2*n_enc_2)
        self.mlp3 = MLP_3(2*n_enc_3)
        # cluster layer
        self.cluster_layer = Parameter(torch.Tensor(n_clusters, n_z))
        torch.nn.init.xavier_normal_(self.cluster_layer.data)
        self._dec_mean = nn.Sequential(nn.Linear(n_dec_3, n_input), MeanAct())
        self._dec_disp = nn.Sequential(nn.Linear(n_dec_3, n_input), DispAct())
        self._dec_pi = nn.Sequential(nn.Linear(n_dec_3, n_input), nn.Sigmoid())
        self.v = v
        self.zinb_loss = ZINBLoss().cuda()

    def forward(self, x, adj):
        x_bar, h1, h2, h3, z, dec_h3 = self.ae(x)
        sigma = 0.5
        x_array = list(np.shape(x))
        n_x = x_array[0]
        z1 = self.agcn_0(x, adj)

        m1 = self.mlp1( torch.cat((h1,z1), 1) )
        m1 = F.normalize(m1, p=2)
        m11 = torch.reshape(m1[:,0], [n_x, 1])
        m12 = torch.reshape(m1[:,1], [n_x, 1])
        m11_broadcast =  m11.repeat(1, 500)
        m12_broadcast =  m12.repeat(1, 500)
        z2 = self.agcn_1( m11_broadcast.mul(z1)+m12_broadcast.mul(h1), adj)
        m2 = self.mlp2( torch.cat((h2,z2),1) )
        m2 = F.normalize(m2,p=2)
        m21 = torch.reshape(m2[:,0], [n_x, 1])
        m22 = torch.reshape(m2[:,1], [n_x, 1])
        m21_broadcast = m21.repeat(1, 500)
        m22_broadcast = m22.repeat(1, 500)
        z3 = self.agcn_2( m21_broadcast.mul(z2)+m22_broadcast.mul(h2), adj)
        m3 = self.mlp3( torch.cat((h3,z3),1) )# self.mlp3(h2)
        m3 = F.normalize(m3,p=2)
        m31 = torch.reshape(m3[:,0], [n_x, 1])
        m32 = torch.reshape(m3[:,1], [n_x, 1])
        m31_broadcast = m31.repeat(1, 2000)
        m32_broadcast = m32.repeat(1, 2000)
        z4 = self.agcn_3( m31_broadcast.mul(z3)+m32_broadcast.mul(h3), adj)
        u = self.mlp(torch.cat((z1, z2, z3, z4, z), 1))
        u = F.normalize(u, p=2)
        u0 = torch.reshape(u[:, 0], [n_x, 1])
        u1 = torch.reshape(u[:, 1], [n_x, 1])
        u2 = torch.reshape(u[:, 2], [n_x, 1])
        u3 = torch.reshape(u[:, 3], [n_x, 1])
        u4 = torch.reshape(u[:, 4], [n_x, 1])
        tile_u0 = u0.repeat(1, 500)
        tile_u1 = u1.repeat(1, 500)
        tile_u2 = u2.repeat(1, 2000)
        tile_u3 = u3.repeat(1, 10)
        tile_u4 = u4.repeat(1, 10)
        net_output = torch.cat((tile_u0.mul(z1), tile_u1.mul(z2), tile_u2.mul(z3), tile_u3.mul(z4), tile_u4.mul(z)), 1 )
        net_output = self.agcn_z(net_output, adj, active=False)
        _mean = self._dec_mean(dec_h3)
        _disp = self._dec_disp(dec_h3)
        _pi = self._dec_pi(dec_h3)
        zinb_loss = self.zinb_loss
        predict = F.softmax(net_output, dim=1)
        # qij
        q = 1.0 / (1.0 + torch.sum(torch.pow(z.unsqueeze(1) - self.cluster_layer, 2), 2) / self.v)
        q = q.pow((self.v + 1.0) / 2.0)
        q = (q.t() / torch.sum(q, 1)).t()

        return x_bar, q, predict, z, _mean, _disp, _pi, zinb_loss, net_output


def target_distribution(q):
    # Pij
    weight = q ** 2 / q.sum(0)
    return (weight.t() / weight.sum(1)).t()


def train_mfgnn(dataset, X_raw, sf, name, method, k):
    global p
    dataname = '{}'.format(name)
    eprm_state = 'result'

    file_out = open('output'+ dataname + '_' + eprm_state + '.txt', 'a')

    if dataname=='Adam':
        ld1 = 10
        ld2 = 10
        ld3 = 1
    elif dataname == '10X_PBMC':
        ld1 = 0.1
        ld2 = 1
        ld3 = 100
    elif dataname == 'Muraro':
        ld1 = 100
        ld2 = 1000
        ld3 = 0.001
    elif dataname == 'Quake_10x_Bladder':
        ld1 = 1
        ld2 = 0.01
        ld3 = 0.001
    elif dataname == 'Quake_10x_Limb_Muscle':
        ld1 = 0.01
        ld2 = 1000
        ld3 = 0.001
    elif dataname == 'Quake_10x_Spleen':
        ld1 = 0.001
        ld2 = 0.01
        ld3 = 1
    elif dataname == 'Quake_Smart-seq2_Diaphragm':
        ld1 = 1
        ld2 = 1
        ld3 = 0.1
    elif dataname == 'Quake_Smart-seq2_Limb_Muscle':
        ld1 = 10
        ld2 = 1000
        ld3 = 1
    elif dataname == 'Quake_Smart-seq2_Lung':
        ld1 = 10
        ld2 = 1
        ld3 = 100
    elif dataname == 'Quake_Smart-seq2_Trachea':
        ld1 = 100
        ld2 = 0.01
        ld3 = 0.001
    elif dataname == 'Romanov':
        ld1 = 1000
        ld2 = 0.001
        ld3 = 0.1
    elif dataname == 'Young':
        ld1 = 1000
        ld2 = 0.001
        ld3 = 0.001
    else:
        ld1 = 1000
        ld2 = 1000
        ld3 = 1
    model = MFGNN(500, 500, 2000, 2000, 500, 500,
                  n_input=args.n_input,
                  n_z=args.n_z,
                  n_clusters=args.n_clusters,
                  v=1.0).cuda()

    optimizer = Adam(model.parameters(), lr=args.lr)
    adj = load_graph(args.graph, method, k)
    adj = adj.cuda()
    data = torch.Tensor(dataset.x).to(device)
    y = dataset.y

    with torch.no_grad():
        _, _, _, _, z, _ = model.ae(data)

    iters10_kmeans_iter_Q = []
    iters10_NMI_iter_Q = []
    iters10_ARI_iter_Q = []

    iters10_kmeans_iter_Z = []
    iters10_NMI_iter_Z = []
    iters10_ARI_iter_Z = []


    iters10_kmeans_iter_P = []
    iters10_NMI_iter_P = []
    iters10_ARI_iter_P = []

    z_1st = z
    for i in range(1):

        kmeans = KMeans(n_clusters=args.n_clusters, n_init=20)
        y_pred = kmeans.fit_predict(z_1st.data.cpu().numpy())
        model.cluster_layer.data = torch.tensor(kmeans.cluster_centers_).cuda()

        kmeans_iter_Q = []
        NMI_iter_Q = []
        ARI_iter_Q = []


        kmeans_iter_Z = []
        NMI_iter_Z = []
        ARI_iter_Z = []


        kmeans_iter_P = []
        NMI_iter_P = []
        ARI_iter_P = []


        for epoch in range(200):

            if epoch % 1 == 0:
                _, tmp_q, pred, _, _, _, _, _, _ = model(data, adj)
                tmp_q = tmp_q.data
                p = target_distribution(tmp_q)

                res1 = tmp_q.cpu().numpy().argmax(1)
                res2 = pred.data.cpu().numpy().argmax(1)
                res3 = p.data.cpu().numpy().argmax(1)

                acc, nmi, ari = eva(y, res1, str(epoch) + 'Q')
                kmeans_iter_Q.append(acc)
                NMI_iter_Q.append(nmi)
                ARI_iter_Q.append(ari)



                acc, nmi, ari = eva(y, res2, str(epoch) + 'Z')
                kmeans_iter_Z.append(acc)
                NMI_iter_Z.append(nmi)
                ARI_iter_Z.append(ari)


                acc, nmi, ari = eva(y, res3, str(epoch) + 'P')
                kmeans_iter_P.append(acc)
                NMI_iter_P.append(nmi)
                ARI_iter_P.append(ari)



            x_bar, q, pred, z, meanbatch, dispbatch, pibatch, zinb_loss, _ = model(data, adj)

            kl_loss = F.kl_div(q.log(), p, reduction='batchmean')
            ce_loss = F.kl_div(pred.log(), p, reduction='batchmean')
            re_loss = F.mse_loss(x_bar, data)


            X_raw = torch.as_tensor(X_raw).cuda()
            sf = torch.as_tensor(sf).cuda()

            zinb_loss = zinb_loss(X_raw, meanbatch, dispbatch, pibatch, sf)

            loss = ld1 * kl_loss + ld2 * ce_loss + re_loss + ld3*zinb_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


    # _Q
    kmeans_max= np.max(kmeans_iter_Q)
    nmi_max= np.max(NMI_iter_Q)
    ari_max= np.max(ARI_iter_Q)
    # F1_max= np.max(F1_iter_Q)
    iters10_kmeans_iter_Q.append(round(kmeans_max,5))
    iters10_NMI_iter_Q.append(round(nmi_max,5))
    iters10_ARI_iter_Q.append(round(ari_max,5))
    # iters10_F1_iter_Q.append(round(F1_max,5))

    # _Z
    kmeans_max= np.max(kmeans_iter_Z)
    nmi_max= np.max(NMI_iter_Z)
    ari_max= np.max(ARI_iter_Z)
    # F1_max= np.max(F1_iter_Z)
    iters10_kmeans_iter_Z.append(round(kmeans_max,5))
    iters10_NMI_iter_Z.append(round(nmi_max,5))
    iters10_ARI_iter_Z.append(round(ari_max,5))
    # iters10_F1_iter_Z.append(round(F1_max,5))

    # _P
    kmeans_max= np.max(kmeans_iter_P)
    nmi_max= np.max(NMI_iter_P)
    ari_max= np.max(ARI_iter_P)
    # F1_max= np.max(F1_iter_P)
    iters10_kmeans_iter_P.append(round(kmeans_max,5))
    iters10_NMI_iter_P.append(round(nmi_max,5))
    iters10_ARI_iter_P.append(round(ari_max,5))
    print('{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}'.format(
        ld1,
        ld2,
        ld3,
        round(np.mean(iters10_kmeans_iter_Z), 5), round(np.mean(iters10_NMI_iter_Z), 5),
        round(np.mean(iters10_ARI_iter_Z), 5)), file=file_out)
    file_out.close()

def read_clean(data):
    assert isinstance(data, np.ndarray)
    if data.dtype.type is np.bytes_:
        data = utilsmyself.decode(data)
    if data.size == 1:
        data = data.flat[0]
    return data



def dict_from_group(group):
    assert isinstance(group, h5py.Group)
    d = utilsmyself.dotdict()
    for key in group:
        if isinstance(group[key], h5py.Group):
            value = dict_from_group(group[key])
        else:
            value = read_clean(group[key][...])
        d[key] = value
    return d



def read_data(filename, sparsify = False, skip_exprs = False):
    with h5py.File(filename, "r") as f:
        obs = pd.DataFrame(dict_from_group(f["obs"]), index = utilsmyself.decode(f["obs_names"][...]))
        var = pd.DataFrame(dict_from_group(f["var"]), index = utilsmyself.decode(f["var_names"][...]))
        uns = dict_from_group(f["uns"])
        if not skip_exprs:
            exprs_handle = f["exprs"]
            if isinstance(exprs_handle, h5py.Group):
                mat = sp.sparse.csr_matrix((exprs_handle["data"][...], exprs_handle["indices"][...],
                                               exprs_handle["indptr"][...]), shape = exprs_handle["shape"][...])
            else:
                mat = exprs_handle[...].astype(np.float32)
                if sparsify:
                    mat = sp.sparse.csr_matrix(mat)
        else:
            mat = sp.sparse.csr_matrix((obs.shape[0], var.shape[0]))
    return mat, obs, var, uns




def prepro(filename):
    data_path = filename
    mat, obs, var, uns = read_data(data_path, sparsify=False, skip_exprs=False)
    if isinstance(mat, np.ndarray):
        X = np.array(mat)
    else:
        X = np.array(mat.toarray())
    cell_name = np.array(obs["cell_type1"])
    cell_type, cell_label = np.unique(cell_name, return_inverse=True)
    return X, cell_label



if __name__ == "__main__":
    datalists = ['10X_PBMC']
    method = 'p'
    k = 5
    for i in range(10):
        for data in datalists:

            File = ['{}'.format(data), '{}'.format(data), 'model/{}.pkl'.format(data),
                    'data/{}_new.h5ad'.format(data)]
            torch.cuda.current_device()

            x = np.loadtxt('data/{}.txt'.format(data), dtype=float)
            print(x.shape)
            y = np.loadtxt('data/{}_label.txt'.format(data), dtype=int)
            n_input=x.shape[1]
            n_cluster = len(set(y))
            model_para = [500, 500, 2000]
            Para = [1024, 1e-4, 300]
            Cluster_para = [n_cluster, 10, n_input, 20]
            Balance_para = [0.1, 0.01, 1, 0.1]
            parser = argparse.ArgumentParser(
                description='train',
                formatter_class=argparse.ArgumentDefaultsHelpFormatter)

            parser.add_argument('--name', type=str, default=File[0])
            parser.add_argument('--graph', type=str, default=File[1])
            parser.add_argument('--pretrain_path', type=str, default=File[2])
            parser.add_argument('--n_enc_1', default=model_para[0], type=int)
            parser.add_argument('--n_enc_2', default=model_para[1], type=int)
            parser.add_argument('--n_enc_3', default=model_para[2], type=int)
            parser.add_argument('--n_dec_1', default=model_para[2], type=int)
            parser.add_argument('--n_dec_2', default=model_para[1], type=int)
            parser.add_argument('--n_dec_3', default=model_para[0], type=int)

            parser.add_argument('--k', type=int, default=None)
            parser.add_argument('--lr', type=float, default=Para[1])

            parser.add_argument('--n_clusters', default=Cluster_para[0], type=int)
            parser.add_argument('--n_z', default=Cluster_para[1], type=int)
            parser.add_argument('--n_input', type=int, default=Cluster_para[2])
            parser.add_argument('--n_init', type=int, default=Cluster_para[3])

            args = parser.parse_args()
            print(torch.cuda.current_device())

            args = parser.parse_args()
            args.cuda = torch.cuda.is_available()
            print("use cuda: {}".format(args.cuda))

            device = torch.device("cuda" if args.cuda else "cpu")
            args.pretrain_path = File[2]
            dataset = load_data(args.name)

            data_mat = h5py.File(File[3], "r+")
            x = np.array(data_mat['X'])
            y = np.array(data_mat['obs']['labels'])

            adata = sc.AnnData(x)
            adata.obs['Group'] = y
            adata = read_dataset(adata,
                                 transpose=False,
                                 test_split=False,
                                 copy=True)

            adata = normalize(adata,
                              size_factors=True,
                              normalize_input=True,
                              logtrans_input=True)
            X = adata.X
            X_raw = adata.raw.X
            sf = adata.obs.size_factors
            train_mfgnn(dataset, X_raw, sf, data, method=method, k=k)


