import numpy as np
import pandas as pd
import scanpy as sc
import h5py
import anndata
import scipy as sp
import utilsForMain


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




datalists=['Adam', 'Bach', 'Chen', 'Klein', 'Muraro', 'Plasschaert', 'Pollen',
           'Quake_10x_Bladder', 'Quake_10x_Limb_Muscle', 'Quake_10x_Spleen', 'Quake_10x_Trachea',
           'Quake_Smart-seq2_Diaphragm', 'Quake_Smart-seq2_Heart',
           'Quake_Smart-seq2_Limb_Muscle',
           'Quake_Smart-seq2_Lung', 'Quake_Smart-seq2_Trachea',
           'Romanov', 'Tosches_turtle', 'Wang_Lung', 'Young']



for name in datalists:

    x, y = prepro('./data/' + '{}'.format(name) + '/data.h5')
    sc.settings.verbosity = 3
    sc.settings.set_figure_params(dpi=80, facecolor='white')
    adata = sc.AnnData(x)
    adata.obs['labels'] = y
    adata.var_names_make_unique()
    print(adata)
    sc.pl.highest_expr_genes(adata, n_top=20, )
    sc.pp.filter_cells(adata, min_genes=200)
    sc.pp.filter_genes(adata, min_cells=3)
    adata.var['mt'] = adata.var_names.str.startswith('MT-')
    sc.pp.calculate_qc_metrics(adata, qc_vars=['mt'], percent_top=None, log1p=False, inplace=True)
    sc.pl.violin(adata, ['n_genes_by_counts', 'total_counts', 'pct_counts_mt'],
                 jitter=0.4, multi_panel=True)
    sc.pl.scatter(adata, x='total_counts', y='pct_counts_mt')
    sc.pl.scatter(adata, x='total_counts', y='n_genes_by_counts')
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=3, min_disp=0.5)
    sc.pl.highly_variable_genes(adata)
    adata.raw = adata
    adata = adata[:, adata.var.highly_variable]
    print(adata.X)
    adata.write_h5ad(r'data/'+name+'_new.h5ad')

    X = adata.X

    x = pd.DataFrame(X)
    y = adata.obs['labels']
    print(x.shape)
    print(y.shape)
    np.savetxt("data/{}.txt".format(name), x, fmt='%f',delimiter=' ')
    np.savetxt("data/{}_label.txt".format(name), y, fmt='%d',delimiter=' ')
