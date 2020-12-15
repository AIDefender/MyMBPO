import pickle 
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

def plot(data, style = 'reacher'):

    df = pd.DataFrame(data['obs'])
    
    if style == 'reacher':
        df = df.drop(labels=[0,1,2,3,4,5,6,7,10], axis=1)
        df['dist'] = (df[8] ** 2 + df[9] ** 2)**0.5

        dist_scale = 300

    if style == 'fetch':
        df = df[[6,7,8]]
        df['dist'] = (df[6] ** 2 + df[7] ** 2 + df[8]**2)**0.5
        dist_scale = 30

    df['q_std'] = pd.Series(data['q_std'])
    df['pi_std'] = pd.Series(data['pi_std'])

    df['dist_range'] = df['dist'].apply(lambda x: int(dist_scale*x))

    df.to_csv("data.csv")
    grp_df = df.groupby(['dist_range']).mean()
    grp_df_var = df.groupby(['dist_range']).var()
    grp_df_count = df.groupby(['dist_range']).count()
    grp_df.to_csv("mean.csv")
    grp_df_var.to_csv("var.csv")
    grp_df_count.to_csv("count.csv")
    grp_df.drop(labels="dist", axis=1)

    x = np.array(list(range(len(grp_df['q_std']))))/dist_scale
    plt.plot(x, grp_df['q_std'], label = "std of ensembled Q")
    plt.plot(x, grp_df['pi_std'] * 10, label = "std of policy(10x)")
    plt.legend()
    plt.xlabel("Distance from target")
    plt.ylabel("Value")
    plt.title("Comparison of Q and policy std on env Reacher-v2, checkpoint 60")

    plt.savefig("reacher-60.png")

# with open("data/fetch-40.pkl", 'rb') as f:
with open("data/reacher-60.pkl", 'rb') as f:
    data = pickle.load(f)

plot(data, style='reacher')