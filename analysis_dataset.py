
import matplotlib.pyplot as plt
import pandas as pd
import os
import numpy as np
import seaborn as sns
from sklearn.feature_selection import mutual_info_classif

script_dir = os.path.dirname(__file__)
rel_path = "milknew.csv"
abs_file_path = os.path.join(script_dir,
                             rel_path)
data = pd.read_csv(abs_file_path)


def add_drops(data):
    gen_1 = np.random.RandomState(seed=0)
    gen_2 = np.random.RandomState(seed=1)
    gen_3 = np.random.RandomState(seed=42)
    number_of_drops = gen_1.randint(0,
                                    len(data)//4,
                                    1)
    drop_indx = gen_2.randint(0,
                              len(data),
                              number_of_drops)
    drop_column = gen_3.randint(0,
                                data.shape[1],
                                2)
    for column in drop_column:
        data.iloc[drop_indx, column] = [np.nan]*len(drop_indx)


class analysis_dataframe:

    def __init__(self, dataframe):
        self.X = dataframe.iloc[:, :-1]
        self.target = dataframe.iloc[:, -1]
        self.all_data = dataframe

    def drops(self):
        table_of_nans = self.all_data.isnull().sum()
        values_of_table = np.array(table_of_nans.values)
        if values_of_table.sum() == 0:
            print('Not drops in dataset')
        else:
            fig = plt.figure(figsize=(6, 10))
            fig.subplots_adjust(hspace=0.5)
            tables_with_drops = table_of_nans[table_of_nans != 0]
            columns_with_drops = tables_with_drops.index.values
            num_cols = len(tables_with_drops.index.values)
            for indx, column in enumerate(columns_with_drops):
                print(f'Column - {column}, '
                      f'Drops - {tables_with_drops[column]}')
                ax = fig.add_subplot(int(f'{num_cols}1{indx+1}'))
                sns.histplot(self.X,
                             x=column,
                             kde=True,
                             edgecolor='black',
                             ax=ax,
                             label=column)
                ax.legend()
            plt.show()
            return columns_with_drops

    def disbalance_classes(self):
        val_count = np.array(np.unique(self.target, return_counts=True))
        rates = [x/len(self.target) for x in val_count[1]]
        return dict(list(zip(val_count[0], rates)))

    def binary_features(self):
        features = []
        for column in self.X.columns:
            vals = np.array(self.X.loc[:, column].values)
            if len(np.unique(vals)) <= 2:
                features.append(column)
        return features
    
    def cat_features(self):
        return self.X.select_dtypes(include='object').columns.values

    def num_features(self):
        for_drop = np.hstack((self.cat_features(), self.binary_features()))
        return self.X.drop(columns=for_drop).columns.values

    def mutual_inf(self):
        features = self.X.columns
        values_of_cov = mutual_info_classif(self.X, self.target)
        table = pd.Series(values_of_cov, index=features)
        return table.sort_values(ascending=False)
    