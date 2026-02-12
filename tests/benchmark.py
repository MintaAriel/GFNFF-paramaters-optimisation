import pandas as pd
import sqlite3
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
from scipy.ndimage import label

# from tabulate import tabulate

d10_rmsd = '/home/vito/PythonProjects/PythonProject/results/10_500.db'
d20_rmsd = '/home/vito/PythonProjects/PythonProject/results/20_500.db'
d30_rmsd = '/home/vito/PythonProjects/PythonProject/results/30_500.db'

d10_c_rmsd = '/home/vito/PythonProjects/PythonProject/results/10_500_rmsd_cor.db'
d20_c_rmsd = '/home/vito/PythonProjects/PythonProject/results/20_500_rmsd_cor.db'
d30_c_rmsd = '/home/vito/PythonProjects/PythonProject/results/30_500_rmsd_cor.db'


d10_soap = '/home/vito/PythonProjects/PythonProject/results/10_500_soap.db'
d20_soap = '/home/vito/PythonProjects/PythonProject/results/20_500_soap.db'
d30_soap = '/home/vito/PythonProjects/PythonProject/results/30_500_soap.db'

d10_c_soap = '/home/vito/PythonProjects/PythonProject/results/10_500_soap_cor.db'
d20_c_soap = '/home/vito/PythonProjects/PythonProject/results/20_500_soap_cor.db'
d30_c_soap = '/home/vito/PythonProjects/PythonProject/results/30_500_soap_cor.db'


def sql2df(sqlite_dir):
    con = sqlite3.connect(sqlite_dir)
    df_values = pd.read_sql('SELECT * FROM trial_values',con)
    best_id = df_values.loc[df_values['value'].idxmax(), 'trial_id']
    df_best = pd.read_sql(f'SELECT * FROM trial_params WHERE trial_id = {best_id}',con)
    df_best.drop('distribution_json', axis=1, inplace=True)
    return df_values, df_best


def sql2df_best(sqlite_dir, best_id):
    con = sqlite3.connect(sqlite_dir)
    df = pd.read_sql('SELECT * FROM trial_params',con)
    df = df[df['param_id'] == best_id]

    return df

def plot_step(df_dic, standard):
    first_value = []
    best_try = []
    plt.figure(figsize=(8, 6))
    plt.rc('font', size=14)
    fm.fontManager.addfont('/home/vito/miniconda3/envs/gnff_env/fonts/times.ttf')
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman']


    for k,v in df_dic.items():
        df = v[0]
        df['cummax'] = df['value'].transform('cummax')
        print(df.head())
        first_value.append(df['cummax'].min())
        best_try.append(df['value'].max())

        x = df['trial_id']
        y = df['value']
        # plt.scatter(x,y, marker='s')
        plt.step(x,df['cummax'], where='post', label=k)

    print('The best values', best_try)
    plt.axhline(y=standard, color='r', linestyle='--', label='mcGFNFF')
    plt.ylim(min(first_value),)
    plt.xlim(0,)
    plt.xlabel('Trials')
    plt.ylabel('Score')
    plt.legend()
    plt.show()



exp_gfnff_rmsd = 0.7741
exp_gfnff_soap  = 0.956

experiment_rmsd = {'d10': sql2df(d10_rmsd),
                    'd20':sql2df(d20_rmsd),
                    'd30':sql2df(d30_rmsd),
                   'd10_c':sql2df(d10_c_rmsd),
                   'd20_c':sql2df(d20_c_rmsd),
                   'd30_c':sql2df(d30_c_rmsd),}

experiment_soap = {'d10': sql2df(d10_soap),
                    'd20':sql2df(d20_soap),
                    'd30':sql2df(d30_soap),
                   'd10_c':sql2df(d10_c_soap),
                   'd20_c':sql2df(d20_c_soap),
                   'd30_c':sql2df(d30_c_soap),}

# plot_step(experiment_rmsd, exp_gfnff_rmsd)
# plot_step(experiment_soap, exp_gfnff_soap)

for k,v in experiment_soap.items():
    v[1]['experiment'] = k
    v[1]['param_value'] = v[1]['param_value'].round(4)
    print(v[1])

for k,v in experiment_rmsd.items():
    v[1]['experiment'] = k
    v[1]['param_value'] = v[1]['param_value'].round(4)
    print(v[1])




# print(tabulate(df.head(), headers='keys', tablefmt='psql'))
