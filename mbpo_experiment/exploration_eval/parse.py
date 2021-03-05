import pickle 
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import os
sns.set()


def plot(data, style = 'reacher'):
    this_scale = scales[EXP_INDEX]

    df = pd.DataFrame(data['obs'])
    
    if style == 'reacher':
        df = df.drop(labels=[0,1,2,3,4,5,6,7,10], axis=1)
        df['dist'] = (df[8] ** 2 + df[9] ** 2)**0.5

        dist_scale = 300

    if style == 'fetch':
        df = df[[6,7,8]]
        df['dist'] = (df[6] ** 2 + df[7] ** 2 + df[8]**2)**0.5
        dist_scale = 30

    if style == 'grid':
        df['dist'] = ((df[0] - 5) ** 2 + (df[1] - 5) ** 2)**0.5
        dist_scale = 3

    df['q_std'] = pd.Series(data['q_std'])
    df['pi_std'] = pd.Series(data['pi_std'])

    df['dist_range'] = df['dist'].apply(lambda x: int(dist_scale*x))

    df.to_csv("data.csv")
    grp_df = df.groupby(['dist_range']).mean()
    grp_df_count = df.groupby(['dist_range']).count()
    grp_df.to_csv("mean.csv")
    grp_df_count.to_csv("count.csv")

    x = np.array(list(range(len(grp_df['q_std']))))/dist_scale
    plt.plot(x, grp_df['q_std'] * this_scale["q_std"], label = "std of ensembled Q(100x)")
    plt.plot(x, grp_df['pi_std'] * this_scale["pi_std"], label = "std of policy(1x)")
    # plt.plot(x, grp_df['pi_std'], label = "std of policy")
    plt.plot(x, np.log(grp_df_count['dist']) * this_scale["cnt"], label = "log Count(0.1x)")
    plt.legend()
    plt.xlabel("Distance from target")
    plt.ylabel("Value")
    plt.title("Comparison of Q and policy std on env Continuous Grid, checkpoint %s"%EXP_INDEX)

    plt.savefig("plot.png")

# EXP_NAME = "reacher"
EXP_NAME = "grid"
EXP_INDEX = "55"
SAVE_PATH="%s-%s"%(EXP_NAME, EXP_INDEX)
if not os.path.isdir(SAVE_PATH):
    os.mkdir(SAVE_PATH)
scales = {
    "5": {
        "q_std": 100,
        "pi_std": 1,
        "cnt": 0.1
    },
    "10": {
        "q_std": 100,
        "pi_std": 1,
        "cnt": 0.1
    },
    "15": {
        "q_std": 100,
        "pi_std": 1,
        "cnt": 0.1
    },
    "20": {
        "q_std": 100,
        "pi_std": 1,
        "cnt": 0.1
    },
    "25": {
        "q_std": 100,
        "pi_std": 1,
        "cnt": 0.1
    },
    "30": {
        "q_std": 100,
        "pi_std": 1,
        "cnt": 0.1
    },
    "35": {
        "q_std": 100,
        "pi_std": 1,
        "cnt": 0.1
    },
    "40": {
        "q_std": 100,
        "pi_std": 1,
        "cnt": 0.1
    },
    "45": {
        "q_std": 100,
        "pi_std": 1,
        "cnt": 0.1
    },
    "50": {
        "q_std": 100,
        "pi_std": 1,
        "cnt": 0.1
    },
    "55": {
        "q_std": 100,
        "pi_std": 1,
        "cnt": 0.1
    },
}
with open("grid_data/%s.pkl"%EXP_INDEX, 'rb') as f:
    data = pickle.load(f)
os.chdir(SAVE_PATH)
plot(data, style=EXP_NAME)