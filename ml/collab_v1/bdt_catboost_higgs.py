""" Process the MC higgs data to do a ML classification 


"""
import os,sys

# pylint: disable=g-bad-import-order
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

#some plotting routines
import plotting

# data processing
import data_higgs as dh

# #import the working methods
# import tensorflow as tf
# print("TensorFlow version ",tf.__version__)

# from tensorflow import keras
# print("TF Keras version ",tf.keras.__version__)

from catboost import CatBoostClassifier, Pool


#--------- globals

# Need to see a large portion of the data before we can build a layer, for
# example half of data n_batches_per_layer =  NBATCH_FRAC * NUM_EXAMPLES / BATCH_SIZE
BATCH_SIZE = 1000

# Seed value
# Apparently you may use different seed values at each stage
SEED_VALUE= 10001
# 1. Set `PYTHONHASHSEED` environment variable at a fixed value
import os
os.environ['PYTHONHASHSEED']=str(SEED_VALUE)
# 2. Set `python` built-in pseudo-random generator at a fixed value
import random
random.seed(SEED_VALUE)
# 3. Set `numpy` pseudo-random generator at a fixed value
np.random.seed(SEED_VALUE)
# # 4. Set `tensorflow` pseudo-random generator at a fixed value
# tf.random.set_seed(SEED_VALUE)

#-------- routines

def split_xy(rawdata):
    #split features and labels from data 
    #prepare the data => normalizations !   

    # split 
    data_y=rawdata['hlabel'] # labels only: 0.=bkg, 1.=sig
    data_x=rawdata.drop(['hlabel'], axis=1) # features only
    
    #now prepare the data
    mu = data_x.mean()
    s = data_x.std()
    dmax = data_x.max()
    dmin = data_x.min()

    # normal/standard rescaling 
    #data_x = (data_x - mu)/s

    # scaling to [-1,1] range
    #data_x = -1. + 2.*(data_x - dmin)/(dmax-dmin)

    # scaling to [0,1] range
    data_x = (data_x - dmin)/(dmax-dmin)


    return data_x,data_y

#-------- control

if __name__ == "__main__":
    # Training progress and eval results are shown as logging.INFO; so enables it.
 
    #needs to be done once if the h5 files need to be recreated...
    #dh.download_and_make_data()

    hdata=dh.load_data()
    data_fnames=hdata['feature_names'].to_numpy()[1:] #drop labels
    n_dims=data_fnames.shape[0]
    print ("Entries read {} with feature names {}".format(n_dims,data_fnames))

    x_trn,y_trn=split_xy(hdata['train']) # training sample, should split a fraction for testing
    x_train, x_test,y_train, y_test = train_test_split(x_trn,y_trn,test_size=0.1) # 10% split
    x_val,y_val=split_xy(hdata['valid']) # independent cross-valid sample

    print("Shapes train:{} and test:{}".format(x_train.shape,x_test.shape))

    #plot distributions
    plotting.plot_sig_bkg_from_np_arrays(x_train.to_numpy(),y_train.to_numpy(),data_fnames,logy=False)

    # ready the data for Catboost
    pool_train = Pool(data=x_train.to_numpy(),label=y_train.to_numpy(),feature_names=data_fnames.tolist())
    pool_test = Pool(data=x_test.to_numpy(),label=y_test.to_numpy(),feature_names=data_fnames.tolist())

    # CatBoost parameters 
    eval_metric = 'AUC' # see https://catboost.ai/docs/concepts/loss-functions-classification.html
    task_type = 'CPU'  # if GPU else 'CPU'
    max_number_of_trees = 100
    #eval_metric = 'Accuracy'
    bdt=CatBoostClassifier(
        verbose=True,
        task_type=task_type,
        loss_function='Logloss', # see values same as in eval_metric, e.g. CrossEntropy, Logloss is default
        iterations=max_number_of_trees,
        eval_metric=eval_metric,
        #learning_rate=0.1,
        #max_depth=6
        use_best_model=True,
        random_seed=SEED_VALUE,
        )   

    #now train the model
    bdt.fit(
            pool_train,
            #early_stopping_rounds=10,
            eval_set=pool_test,
            plot=True # works when called through Ipython/jupyter
        )


    print()
    #plot & print results like ROC and score distribution etc...
    y_score=bdt.predict_proba(x_val.to_numpy())[:,1]
    print("score shape {}",y_score.shape)
    plotting.plot_roc(y_val,y_score)
    plotting.plot_score(y_val,y_score)
    auc=roc_auc_score(y_val,y_score)
    print("AUC score: {}".format(auc))

