import os

args = [
      "Plasschaert",
]
dic = {
    '10X_PBMC':8,
    'human_kidney_counts':11,
    'worm_neuron_cell':10,
    "Wang_Lung": 2,
}
for arg in args:
    for i in range(10):
        os.system("python scMFGNN.py --name {} --n_clusters {}".format(arg, dic[arg]))

os.system("python /home/derek/ycm/scGAC-main/loop.py")