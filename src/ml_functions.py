import copy
import numpy as np
import pandas as pd
import os
import re
from collections import Counter

from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
import tensorflow as tf
from sklearn.preprocessing import PowerTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

def load_data(data_folder = '../../data', identifier = 'normaloutput', file_type = 'excel', max_files = float('Inf'), **kwargs):
    filenames_all = os.listdir(data_folder)
    if file_type == 'excel':
        file_ext = 'xls.?'
        read_file = pd.read_excel
    elif file_type == 'csv':
        file_ext = 'csv'
        read_file = pd.read_csv
    else:
        print('File_type must be \'excel\' or \'csv\'.')
        return None
    filenames = [x for x in filenames_all if re.search(r'{}.*\.{}$'.format(identifier, file_ext), x)]
    for i, filename in enumerate(filenames):
        if i == 0:
            master_df = read_file('{}/{}'.format(data_folder, filename), **kwargs)
        elif i < max_files:
            new_df = read_file('{}/{}'.format(data_folder, filename), **kwargs)
            master_df = pd.concat([master_df, new_df])
    return master_df

def check_duplicates(df, index_col = None, use_index = False, threshold = 0.001, keep_duplicates = True):
    s = df.index if use_index else df[[index_col]]
    d = Counter(s)
    duplicates = [k for k, v in d.items() if v > 1]
    if duplicates:
        n_duplicates = len(duplicates)
        first = True
        result = []
        for i, v in enumerate(duplicates):
            if use_index:
                df1 = df.loc[v,:]
            else:
                df1 = df.loc[s == v,:]
            mask = (df1.apply(lambda x: abs(x - x[0])) < threshold).apply(all)
            if not all(mask):
                result.append(v)
        n_resolved = n_duplicates - len(result)
        print('{}/{} duplicates resolved'.format(n_resolved, n_duplicates))
        if result:
            if keep_duplicates:
                print('Resolve {} duplicates'.format(len(result)))
                return len(result), True, df.loc[result,:]
            else:
                print('{} unresolved duplicates dropped'.format(len(result)))
                return len(result), False, df[~df.index.duplicated(keep='first')]
        else:
            print('Duplicates resolved')
            return len(result), False, df[~df.index.duplicated(keep='first')]
    else:
        return 0, False, None

def clean_data(df, index_col = 'address_used', use_index = False):
    col = [x for x in df.columns if not(bool(re.match(r'Unnamed', x)))]
    df.address_used = df.address_used.apply(lambda x: re.search(r'\((.+)\)', x).group(1))
    n_duplicates, to_resolve_duplicates, result = check_duplicates(df, index_col, use_index)
    if n_duplicates:
        if to_resolve_duplicates:
            return result
        else:
            df = result
    df = df[col].set_index(index_col)
    for col in df.columns:
        if df[col].dtype == 'object':
            print('Non-numeric column found:', col)
            df[col] = pd.to_numeric(df[col], errors = 'coerce')
    return df

def filter_data(df, drop_na_method = 'any', col_name = 'total_number_of_transactions', lower_bound = 2):
    return df.loc[df[col_name] >= lower_bound,:].dropna(how = drop_na_method)

def scaling(data, scale_cols, scale_methods, default_scaler = 0):
    #data is a pandas dataframe
    #scale_cols is a list of lists of column names
    #scale_methods is a list of scaling methods to be applied to each item in scale_cols
    #Each scaling method is from sklearn.preprocessing, i.e. PowerTransformer(), StandardScaler(), MinMaxScaler()
    data1 = copy.deepcopy(data)
    scalers = []
    for cols, scale_method in zip(scale_cols, scale_methods):
        if cols:
            try:
                scaler = scale_method.fit(data1.loc[:,cols])
                data1.loc[:,cols] = pd.DataFrame(scaler.transform(data1.loc[:,cols]), index = data1.index, columns = cols)
            except:
                scaler = scale_methods[default_scaler].fit(data1.loc[:,cols])
                data1.loc[:,cols] = pd.DataFrame(scaler.transform(data1.loc[:,cols]), index = data1.index, columns = cols)
            scalers.append(scaler)
    return data1, scalers

def split_data(df, label_col, method, **kwargs):
    #df is a pandas dataframe
    #label_col is the column name of the label column in the df, i.e. 'ponzi'
    #method is either KFold' or StratifiedKFold
    #kwargs are the keyword arguments used in the KFold or StratifiedKFold objects, including n_splits (int), shuffle (bool), random_state (int)

    X = df.loc[:,df.columns != label_col]
    y = df.loc[:,label_col].astype(int)
    kf = method(**kwargs)
    
    df_list_list = []
    index_list_list = []

    for train_index, test_index in kf.split(X, y):
        #train_index, test_index = train_index.tolist(), test_index.tolist()
        #print(train_index)
        #print(test_index)
        X_train, X_test = X.iloc[train_index,:], X.iloc[test_index,:]
        y_train, y_test = y[train_index], y[test_index]

        df_list_list.append([X_train, X_test, y_train, y_test])
        index_list_list.append([train_index, test_index])
    
    return df_list_list, index_list_list