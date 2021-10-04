import matplotlib
import pandas as pd
import os
import numpy as np
from astropy.table import Table
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(style="ticks", palette="pastel")

plt.rc('legend', fontsize=25)
result_dir = 'classification_results_food_chemical/results_food_chemical_10_folds_masked'
model_names = ['bert','roberta', 'biobert']
augmentation_settings=['none','aug','aug_bal']
fig, axs = plt.subplots(1,1, sharey=True, tight_layout=True)

augmentation_names={'none': 'non-augmented', 'aug': 'augmented-unbalanced', 'aug_bal':'augmented-balanced'}
metric_short_name={'precision':'precision', 'f1-score':'F1 score', 'recall':'recall'}
for model_idx, model_name in enumerate(model_names):

    results=[]
    for augmentation_setting in augmentation_settings:

        for fold_number in range(0,10):
            fold_results_file = f'{model_name}_{fold_number}_{augmentation_setting}.txt'

            results_path = os.path.join(result_dir, fold_results_file)
            if not os.path.isfile(results_path):
                print(f'Cannot find: {results_path}')
                continue
            model_result_df = Table.read(results_path, format='latex').to_pandas()
            model_result_df = model_result_df.set_index('col0')

            for label in ['0','1']:
                for metric in ['precision','recall','f1-score']:
                    results.append({'metric': f'{label}-{metric_short_name[metric]}', 'fold': fold_number, 'value': model_result_df.loc[metric,label],
                                    'dataset': augmentation_names[augmentation_setting]})
            results.append(
                {'metric': f'macro F1', 'fold': fold_number, 'value': model_result_df.loc['f1-score', 'macro avg'],
                 'dataset': augmentation_names[augmentation_setting]})

    result_df=pd.DataFrame(results)
    fig, axs = plt.subplots(1, 1, sharey=True, tight_layout=True)
    model_ax = axs
    model_ax.tick_params(axis='x', labelsize=25)
    model_ax.tick_params(axis='y', labelsize=25)


    ax = sns.boxplot(x="metric", y="value", ax=model_ax,
        hue="dataset", palette=[ "g", "m", 'c'],
        data=result_df)
    model_ax.set_ylabel('')
    model_ax.set_xlabel('')
    evaluation_title = model_name
    ax.legend(title=None)
    #model_ax.set_title(f'{evaluation_title}', fontdict={'fontsize': 25})
    print(model_name)
    print(result_df.groupby(['metric','dataset']).mean()['value'])
    plt.show()