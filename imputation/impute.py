import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.metrics import hamming_loss # the fraction of labels that are incorrectly predicted
from sklearn.base import clone

def random_mask(df,mask_percentage,seed_):
    '''Randomly mask the dataframe with a given percentage of entries
        return the mask matrix for caluculating the prediction error
    '''
    np.random.seed(seed_)
    mask = np.random.choice([True,False],df.shape,p=[mask_percentage, 1-mask_percentage])
    masked_df = df.mask(mask)
    return masked_df,mask

def mean_mode_impute(df,num_cols_,cat_cols_):
    '''Impute the numerical columns with the mean and 
        categorical columns (oridinal encoded) with the mode
    '''
    masked_df, mask = random_mask(df,0.15,0)
    # get the columns with missing entries
    m = masked_df.isnull().sum()
    m = m[m>0]
    missing_cols = set(m.index)
    # split into numerical and categorical columns
    missing_num = missing_cols & set(num_cols_)
    missing_cat = missing_cols & set(cat_cols_)
    # impute mean for numerical columns and mode for categorical columns
    means = df[missing_num].mean()
    imputed_df = df.fillna(value=means)
    modes = df[missing_cat].mode().loc[0]
    imputed_df.fillna(value=modes,inplace=True)
    return imputed_df,mask,means,modes

def mean_mode_eval(df, num_cols_, cat_cols_):
    '''Evaluate the performance on masked entries'''
    imputed_df,mask,means,modes = mean_mode_impute(df,num_cols_,cat_cols_)
    # get the ground-truth values of the masked entries
    common_missing_idx = np.where((mask + df.isna().to_numpy(dtype=int)) == 2)
    intersection = np.zeros_like(mask,dtype=int)
    intersection[common_missing_idx] = 1
    flip_eval_entries = (mask - intersection) == 0
    g_truth_df = df.mask(flip_eval_entries)
    # calculate the validation error
    num_val_error = {}
    cat_val_error = {}
    for n in num_cols_:
        g_truth = g_truth_df[n]
        g_truth = g_truth[np.logical_not(np.isnan(g_truth))]
        num_val_error[n] = mean_squared_error(g_truth,means[n]*np.ones_like(g_truth))
    for c in cat_cols_:
        g_truth = g_truth_df[c]
        g_truth = g_truth[np.logical_not(np.isnan(g_truth))]
        cat_val_error[c] = hamming_loss(g_truth,modes[c]*np.ones_like(g_truth))
    return imputed_df, num_val_error,cat_val_error

def ml_train(df,num_cols_,cat_cols_,num_model,cat_model,impute):
    '''Use other features to predict the missing feature 
    with machine learning algorithms'''
    # we first impute the missing values with mean/mode 
    # in order to fully utilize the data for training
    imputed_df,mask,_,_ = mean_mode_impute(df,num_cols_,cat_cols_)
    # train the machine learning model on the naïvely imputed dataframe
    # Non-iterative: we only train the model once based on imputed_df. 
    # The values of previously imputed features are not updated.
    models = {}
    for n in num_cols_:
        reg = clone(num_model)
        reg.fit(imputed_df.drop(n,axis=1),imputed_df[n])
        if impute:
            imputed_df[n] = reg.predict(imputed_df.drop(n,axis=1))
        models[n] = reg
    for c in cat_cols_:
        clf = clone(cat_model)
        clf.fit(imputed_df.drop(c,axis=1),imputed_df[c])
        if impute:
            imputed_df[c] = clf.predict(imputed_df.drop(c,axis=1))
        models[c] = clf
    return imputed_df,mask,models

def ml_eval(df,num_cols_,cat_cols_,imputed_df,mask,models,impute):
    '''Impute the missing values and calculate the validation errors'''
    # get the ground-truth values of the masked entries
    common_missing_idx = np.where((mask + df.isna().to_numpy(dtype=int)) == 2)
    intersection = np.zeros_like(mask,dtype=int)
    intersection[common_missing_idx] = 1
    flip_eval_entries = (mask - intersection) == 0 # we want mask \setminus intersection NOT to be masked (i.e. False) and others to be masked (i.e. True)
    g_truth_df = df.mask(flip_eval_entries)
    # calculate the validation error
    num_val_error = {}
    cat_val_error = {}
    for n in num_cols_:
        g_truth = g_truth_df[n].to_numpy()
        indices = np.logical_not(np.isnan(g_truth))
        pred = models[n].predict(imputed_df.drop(n,axis=1))
        # impute the missing values with ML model predictions
        if impute:
            imputed_df[n] = pred
        num_val_error[n] = mean_squared_error(g_truth[indices],pred[indices])
    for c in cat_cols_:
        g_truth = g_truth_df[c].to_numpy()
        indices = np.logical_not(np.isnan(g_truth))
        pred = models[c].predict(imputed_df.drop(c,axis=1))
        # impute the missing values with ML model predictions
        if impute:
            imputed_df[c] = pred
        cat_val_error[c] = hamming_loss(g_truth[indices],pred[indices])
    return imputed_df,num_val_error,cat_val_error

def ml_all_steps(df,num_cols_,cat_cols_,num_model,cat_model):
    '''Combine ml_train and ml_eval into 1 function'''
    imputed_df,mask,models = ml_train(df,num_cols_,cat_cols_,num_model,cat_model,False)
    imputed_df,num_val_error,cat_val_error = ml_eval(df,num_cols_,cat_cols_,imputed_df,mask,models,True)
    return imputed_df,num_val_error,cat_val_error

def iter_ml_impute_eval(df, num_cols_,cat_cols_,num_model,cat_model, num_iter):
    # we first impute the missing values with mean/mode 
    # in order to fully utilize the data for training
    imputed_df,mask,_,_ = mean_mode_impute(df,num_cols_,cat_cols_)
    # We impute missing values by iterating through each feature
    # When impute the next feature, use the newly imputed values for previous features
    models = {}
    num_val_error_df = pd.DataFrame(columns=num_cols_)
    cat_val_error_df = pd.DataFrame(columns=cat_cols_)
    for i in range(num_iter):
        for n in num_cols_:
            if i == 0:
                reg = clone(num_model)
                reg.fit(imputed_df.drop(n,axis=1),imputed_df[n])
                models[n] = reg
                # update the imputed dataframe
                imputed_df[n] = reg.predict(imputed_df.drop(n,axis=1))
            else:
                models[n].fit(imputed_df.drop(n,axis=1),imputed_df[n])
                # update the imputed dataframe
                imputed_df[n] = models[n].predict(imputed_df.drop(n,axis=1))
        for c in cat_cols_:
            if i == 0:
                clf = clone(cat_model)
                clf.fit(imputed_df.drop(c,axis=1),imputed_df[c])
                models[c] = clf
                # update the imputed dataframe
                imputed_df[c] = clf.predict(imputed_df.drop(c,axis=1))
            else:
                models[c].fit(imputed_df.drop(c,axis=1),imputed_df[c])
                # update the imputed dataframe
                imputed_df[c] = models[c].predict(imputed_df.drop(c,axis=1))
        imputed_df,num_val_error,cat_val_error = ml_eval(df,num_cols_,cat_cols_,imputed_df,mask,models,False)
        num_val_error_df = num_val_error_df.append(pd.Series(num_val_error),ignore_index=True)
        cat_val_error_df = cat_val_error_df.append(pd.Series(cat_val_error),ignore_index=True)
    return imputed_df,num_val_error_df,cat_val_error_df