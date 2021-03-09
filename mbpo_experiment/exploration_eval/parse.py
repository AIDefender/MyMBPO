import pickle 
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import seaborn as sns
import os
import argparse
sns.set()
BATCH_DIFF_DATA = True
# BATCH_DIFF_DATA = False

def plot(data, style = 'reacher'):
    this_scale = scales[style]

    df = pd.DataFrame(data['obs'])
    
    if style.find('reacher') != -1:
        df = df.drop(labels=[0,1,2,3,4,5,6,7,10], axis=1)
        df['dist'] = (df[8] ** 2 + df[9] ** 2)**0.5

        dist_scale = 300

    if style == 'fetch':
        df = df[[6,7,8]]
        df['dist'] = (df[6] ** 2 + df[7] ** 2 + df[8]**2)**0.5
        dist_scale = 30

    if style.find('grid') != -1:
        df['dist'] = ((df[0] - 5) ** 2 + (df[1] - 5) ** 2)**0.5
        dist_scale = 3

    if BATCH_DIFF_DATA:
        df['inter_q_std'] = pd.Series(data['inter_q_std'])
        df['cross_q_std'] = pd.Series(data['cross_q_std'])
    else:
        df['q_std'] = pd.Series(data['q_std'])
    df['pi_std'] = pd.Series(data['pi_std'])

    df['dist_range'] = df['dist'].apply(lambda x: int(dist_scale*x))

    df.to_csv("data.csv")
    grp_df = df.groupby(['dist_range']).mean()
    grp_df_count = df.groupby(['dist_range']).count()
    grp_df.to_csv("mean.csv")
    grp_df_count.to_csv("count.csv")

    x = np.array(list(range(len(grp_df['pi_std']))))/dist_scale
    # plt.ylim([0.2, 2])
    # plt.ylim([0, 4.5])
    if BATCH_DIFF_DATA:
        # plt.plot(x, grp_df['inter_q_std'] * this_scale["q_std"], label = "std of inter-group Q(100x)", lw = 3)
        # plt.plot(x, grp_df['cross_q_std'] * this_scale["q_std"], label = "std of cross-group Q(100x)", lw = 3)
        plt.plot(x, grp_df['inter_q_std'] * this_scale["q_std"], label = "std of inter-group Q", lw = 3)
        plt.plot(x, grp_df['cross_q_std'] * this_scale["q_std"], label = "std of cross-group Q", lw = 3)
    else:
        plt.plot(x, grp_df['q_std'] * this_scale["q_std"], label = "std of ensembled Q", lw = 3)

    plt.plot(x, grp_df['pi_std'] * this_scale["pi_std"], label = "std of policy(10x)", lw = 3)
    # plt.plot(x, grp_df['pi_std'] * this_scale["pi_std"], label = "std of policy(1x)", lw = 3)
    plt.plot(x, np.log(grp_df_count['dist']) * this_scale["cnt"], label = "log Count(0.1x)", lw = 3)
    plt.legend(fontsize = 16)
    plt.xlabel("Distance from target", fontsize=16)
    plt.ylabel("Value", fontsize=16)
    plt.title("Comparison of Q and policy std on\n env Reacher, checkpoint %s"%EXP_INDEX, fontsize=18)

    plt.savefig("plot.png")

parser = argparse.ArgumentParser()
parser.add_argument("--pkl_dir", "-d", type=str)
parser.add_argument("--exp_index", "-i", type=str)
args = parser.parse_args()
# EXP_NAME = "reacher"
EXP_NAME = args.pkl_dir
EXP_INDEX = args.exp_index
SAVE_PATH="%s/%s"%(EXP_NAME, EXP_INDEX)
if not os.path.isdir(SAVE_PATH):
    os.mkdir(SAVE_PATH)
scales = {
    "grid_sac_data": {
        "q_std": 100,
        "pi_std": 1,
        "cnt": 0.1
    },
    "grid_sac3x3_data": {
        "q_std": 100,
        "pi_std": 1,
        "cnt": 0.1
    },
    "grid_sac3x3distinct_data": {
        "q_std": 100,
        "pi_std": 1,
        "cnt": 0.1
    },
    "reacher_sac_data": {
        "q_std": 1,
        "pi_std": 10,
        "cnt": 0.1
    },
    "reacher_sac3x3distinct_data": {
        "q_std": 1,
        "pi_std": 10,
        "cnt": 0.1
    }
}
with open("%s/%s.pkl"%(EXP_NAME, EXP_INDEX), 'rb') as f:
    data = pickle.load(f)
os.chdir(SAVE_PATH)
plot(data, style=EXP_NAME)