
"""
__file__

    train_model.py

__description__

    This file trains various models.

__author__

    Lei Xu < leixuast@gmail.com >

"""

import sys
import csv
import os
import cPickle
import numpy as np
import pandas as pd
import xgboost as xgb
import time
from datetime import timedelta
from scipy.sparse import hstack
## sklearn
from sklearn.base import BaseEstimator
from sklearn.datasets import load_svmlight_file, dump_svmlight_file
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.linear_model import RidgeClassifier, Lasso, LassoLars, ElasticNet
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import log_loss, f1_score
from sklearn.externals import joblib

## hyperopt
from hyperopt import hp
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials

## cutomized module
from model_library_config import feat_folders, feat_names, param_spaces, int_feat
from collections import Counter





global trial_counter
global log_handler

output_path = "./output"

### global params
## you can use bagging to stabilize the predictions
bootstrap_ratio = 1
bootstrap_replacement = False
bagging_size=  1
verbose_level = 1
n_runs = 5
n_folds = 5

#### warpper for hyperopt for logging the training reslut
#
def hyperopt_wrapper(param, feat_folder, feat_name):
    global trial_counter
    global log_handler
    trial_counter += 1

    # convert integer feat
    for f in int_feat:
        if param.has_key(f):
            param[f] = int(param[f])

    print("------------------------------------------------------------")
    print "Trial %d" % trial_counter

    print("        Model")
    print("              %s" % feat_name)
    print("        Param")
    for k,v in sorted(param.items()):
        print("              %s: %s" % (k,v))
    print("        Result")
    print("                    Run      Fold      Bag      f1_score     Shape")

    ## evaluate performance
    f1score_cv_mean, f1score_cv_std = hyperopt_obj(param, feat_folder, feat_name, trial_counter)

    ## log
    var_to_log = [
        "%d" % trial_counter,
        "%.6f" % f1score_cv_mean, 
        "%.6f" % f1score_cv_std
    ]
    for k,v in sorted(param.items()):
        var_to_log.append("%s" % v)
    writer.writerow(var_to_log)
    log_handler.flush()

    return {'loss': -f1score_cv_mean, 'attachments': {'std': f1score_cv_std}, 'status': STATUS_OK}
    

#### train CV and final model with a specified parameter setting
def hyperopt_obj(param, feat_folder, feat_name, trial_counter):

    f1score_cv = np.zeros((n_runs, n_folds), dtype=float)
    
    for run in range(1,n_runs+1):
        for fold in range(1,n_folds+1):
            rng = np.random.RandomState(2015 + 1000 * run + 10 * fold)

            #### all the path
            path = "%s/Run%d/Fold%d" % (feat_folder, run, fold)
            save_path = "%s/Run%d/Fold%d" % (output_path, run, fold)
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            save_model_path = "%s/Model/Run%d/Fold%d" % (output_path, run, fold)
            if not os.path.exists(save_model_path):
                os.makedirs(save_model_path)

            # feat
            #feat_train_path = "%s/train.feat" % path
            #feat_valid_path = "%s/valid.feat" % path
            # raw prediction path (rank)
            raw_pred_valid_path = "%s/valid.raw.pred.%s_[Id@%d].csv" % (save_path, feat_name, trial_counter)
            rank_pred_valid_path = "%s/valid.pred.%s_[Id@%d].csv" % (save_path, feat_name, trial_counter)

            # save trained model 
            cross_validation_model_path = "%s/%s_[Id@%d].pkl" % (save_model_path, feat_name, trial_counter)
            cross_validation_model_scaler_path = "%s/%s_[Id@%d]_standardscaler.pkl" % (save_model_path, feat_name, trial_counter)




            ## load feat
            with open("%s/train_tfidf_cosine_sim.feat.pkl" % (path), "rb") as f:
                  X_train_tfidf = cPickle.load(f)
            with open("%s/train_bow_cosine_sim.feat.pkl" % (path), "rb") as f:
                  X_train_bow = cPickle.load(f)
            with open("%s/train_header.feat.pkl" % (path), "rb") as f:
                  X_train_header = cPickle.load(f)
            with open("%s/train_tfidf_svd85_cosine_sim.feat.pkl"% (path), "rb") as f:
                  X_train_tfidf_svd85_cosine_sim = cPickle.load(f)


           
             
            #print "%s/valid_tfidf_cosine_sim.feat.pkl" % (path)
            with open("%s/valid_tfidf_cosine_sim.feat.pkl" % (path), "rb") as f:
                  X_valid_tfidf = cPickle.load(f)
            with open("%s/valid_bow_cosine_sim.feat.pkl" % (path), "rb") as f:
                  X_valid_bow = cPickle.load(f)
            with open("%s/valid_header.feat.pkl" % (path), "rb") as f:
                  X_valid_header = cPickle.load(f)
            with open("%s/valid_tfidf_svd85_cosine_sim.feat.pkl"% (path), "rb") as f:
                  X_valid_tfidf_svd85_cosine_sim = cPickle.load(f)





            with open("%s/train_label_scalar.pkl" % (path), "rb") as f:
                  labels_train = cPickle.load(f)
            with open("%s/valid_label_scalar.pkl" % (path), "rb") as f:
                  labels_valid = cPickle.load(f)
            #print X_valid_tfidf.shape
            #print X_valid_bow.shape
            #X_train = np.hstack([X_train_tfidf, X_train_bow])
            #X_valid = np.hstack([X_valid_tfidf, X_valid_bow])
            #X_train = np.hstack([X_train_tfidf, X_train_bow])#, X_train_tfidf_svd85_cosine_sim])#, X_train_header])
            #X_valid = np.hstack([X_valid_tfidf, X_valid_bow])#, X_valid_tfidf_svd85_cosine_sim])#, X_valid_header])
            X_train = X_train_header
            X_valid = X_valid_header


            #X_train, labels_train = load_svmlight_file(feat_train_path)
            #X_valid, labels_valid = load_svmlight_file(feat_valid_path)
            #X_train = X_train.tocsr()
            #X_valid = X_valid.tocsr()
            Y_valid = labels_valid
            #print Y_valid.shape
            #print X_valid.shape
            numTrain = X_train.shape[0]
            numValid = X_valid.shape[0]
            print X_train.shape, X_valid.shape
            ##############
            ## Training ##
            ##############
            ## you can use bagging to stabilize the predictions
            preds_bagging = np.zeros((numValid, bagging_size), dtype=float)
            for n in range(bagging_size):
                if bootstrap_replacement:
                    sampleSize = int(numTrain*bootstrap_ratio)
                    index_base = rng.randint(numTrain, size=sampleSize)
                    index_meta = [i for i in range(numTrain) if i not in index_base]
                else:
                    randnum = rng.uniform(size=numTrain)
                    index_base = [i for i in range(numTrain) if randnum[i] < bootstrap_ratio]
                    index_meta = [i for i in range(numTrain) if randnum[i] >= bootstrap_ratio]
                
                if param.has_key("booster"):
                    #print X_valid.shape
                    #print labels_valid.shape
                    dvalid_base = xgb.DMatrix(X_valid, label=labels_valid)
                    dtrain_base = xgb.DMatrix(X_train[index_base], label=labels_train[index_base])
                        
                    watchlist = []
                    if verbose_level >= 2:
                        watchlist  = [(dtrain_base, 'train'), (dvalid_base, 'valid')]
                    
                ## various models
                if param["task"] in ["softmax"]:
                    ## classification  xgboost
                    # "Model@clf_xgb_linear"
                    clf = xgb.train(param, dtrain_base, param['num_round'], watchlist)
                    pred_proba = clf.predict(dvalid_base)
                    pred_label = np.argmax(pred_proba, axis=1)
                elif param['task'] == "reg_skl_rf":
                    ## classification with sklearn random forest regressor
                    clf = RandomForestClassifier(n_estimators=param['n_estimators'],
                                               max_features=param['max_features'],
                                               n_jobs=param['n_jobs'],
                                               random_state=param['random_state'])
                    clf.fit(X_train[index_base], labels_train[index_base])
                    pred = clf.predict(X_valid)
                    pred_proba = clf.predict_proba(X_valid)
                   

                elif param['task'] == "reg_skl_etr":
                    ## classification with sklearn extra trees regressor
                    clf = ExtraTreesClassifier(n_estimators=param['n_estimators'],
                                              max_features=param['max_features'],
                                              n_jobs=param['n_jobs'],
                                              random_state=param['random_state'])
                    clf.fit(X_train[index_base], labels_train[index_base])
                    pred_proba = clf.predict_proba(X_valid)[:,1]

                elif param['task'] == "reg_skl_gbm":
                    ## classification with sklearn gradient boosting regressor
                    clf = GradientBoostingClassifier(n_estimators=param['n_estimators'],
                                                    max_features=param['max_features'],
                                                    learning_rate=param['learning_rate'],
                                                    max_depth=param['max_depth'],
                                                    subsample=param['subsample'],
                                                    random_state=param['random_state'])
                    clf.fit(X_train.toarray()[index_base], labels_train[index_base])
                    pred_proba = clf.predict_proba(X_valid.toarray())[:,1]
                    

                elif param['task'] == "clf_skl_lr":
                    ## classification with sklearn logistic regression
                    clf = LogisticRegression(penalty="l2", dual=True, tol=1e-5,
                                            C=param['C'], fit_intercept=True, intercept_scaling=1.0,
                                            class_weight='auto', random_state=param['random_state'])
                    clf.fit(X_train[index_base], labels_train[index_base])
                    pred_proba = clf.predict_proba(X_valid)[:,1]

                elif param['task'] == "clf_skl_lr_l1":
                    ## classification with sklearn logistic regression with l1 penalty lasso
                    clf = LogisticRegression(penalty="l1", dual=True, tol=1e-5, solver ="liblinear",
                                            C=param['C'], fit_intercept=True, intercept_scaling=1.0,
                                            class_weight='auto', random_state=param['random_state'])
                    clf.fit(X_train[index_base], labels_train[index_base])
                    pred_proba = clf.predict_proba(X_valid)[:,1]


                elif param['task'] == "reg_skl_svr":
                    ## classification with sklearn support vector regression
                    X_train, X_valid = X_train.toarray(), X_valid.toarray()
                    scaler = StandardScaler()
                    X_train[index_base] = scaler.fit_transform(X_train[index_base])
                    # save standard scaler
                    joblib.dump(scalar, cross_validation_model_scaler_path)
                    X_valid = scaler.transform(X_valid)
                    clf = SVC(C=param['C'], gamma=param['gamma'], 
                                            degree=param['degree'], kernel=param['kernel'], probability=True)
                    clf.fit(X_train[index_base], labels_train[index_base])
                    pred_proba = clf.predict_proba(X_valid)[:,1]
                   
                elif param['task'] == "reg_skl_ridge":
                    ## classification with sklearn ridge regression
                    clf = RidgeClassifier(alpha=param["alpha"], normalize=True)
                    clf.fit(X_train[index_base], labels_train[index_base])
                    pred_proba = clf.predict(X_valid)

                elif param['task'] == "reg_keras_dnn":
                    ## binary classification with keras' deep neural networks
                    model = Sequential()
                    ## input layer
                    model.add(Dropout(param["input_dropout"]))
                    ## hidden layers
                    first = True
                    hidden_layers = param['hidden_layers']
                    while hidden_layers > 0:
                        if first:
                            dim = X_train.shape[1]
                            first = False
                        else:
                            dim = param["hidden_units"]
                        model.add(Dense(dim, param["hidden_units"], init='glorot_uniform'))
                        if param["batch_norm"]:
                            model.add(BatchNormalization((param["hidden_units"],)))
                        if param["hidden_activation"] == "prelu":
                            model.add(PReLU((param["hidden_units"],)))
                        else:
                            model.add(Activation(param['hidden_activation']))
                        model.add(Dropout(param["hidden_dropout"]))
                        hidden_layers -= 1

                    ## output layer
                    model.add(Dense(param["hidden_units"], 1, init='glorot_uniform'))
                    model.add(Activation('linear'))

                    ## loss
                    model.compile(loss='binary_crossentropy', optimizer="adam", metrics=['accuracy'])

                    ## to array
                    X_train = X_train.toarray()
                    X_valid = X_valid.toarray()

                    ## scale
                    scaler = StandardScaler()
                    X_train[index_base] = scaler.fit_transform(X_train[index_base])
                    X_valid = scaler.transform(X_valid)
                    # save standard scaler
                    joblib.dump(scalar, cross_validation_model_scaler_path)

                    ## train
                    model.fit(X_train[index_base], labels_train[index_base]+1,
                                nb_epoch=param['nb_epoch'], batch_size=param['batch_size'],
                                validation_split=0, verbose=0)

                    ##prediction
                    # pred = model.predict(X_valid, verbose=0)
                    # pred.shape = (X_valid.shape[0],)
                    # for each sample return a single value which is a probability of the positive prediction.
                    pred_proba = model.predict_proba(X_valid, verbose=0)


                ## this bagging iteration
                #print len(pred_proba)
                #temp = np.asarray(pred_proba)
                #print temp.shape
                #print pred.shape
                #print pred[0]
                #print pred_proba.shape
                #print pred_proba[0]
                #print (temp[:,0,:])[:, 1]
                #print [pred[i] for i in range(pred.shape[0])]
                
                #preds_bagging[:,n] = pred
                #pred_raw = np.mean(preds_bagging[:,:(n+1)], axis=1)
                #pred_class_bagging = proba2class(pred_raw)              
                #f1score_valid = log_loss(Y_valid, pred_class_bagging)


                #print pred_proba.shape
                #f1score_valid = log_loss(Y_valid, pred_proba, labels=range(pred_proba.shape[1]))
                f1score_valid = f1_score(Y_valid, pred_label, labels=range(pred_proba.shape[1]),average='weighted' )

                if (n+1) != bagging_size:
                    print("              {:>3}   {:>3}   {:>3}   {:>6}   {} x {}".format(
                                run, fold, n+1, np.round(f1score_valid,6), X_train.shape[0], X_train.shape[1]))
                else:
                    print("                    {:>3}       {:>3}      {:>3}    {:>8}  {} x {}".format(
                                run, fold, n+1, np.round(f1score_valid,6), X_train.shape[0], X_train.shape[1]))

                ### saving the training results on the training folds
                ### no bagging here
                ### saving the trained model


                # save the retrained classifer result
                if param['task'] == "reg_keras_dnn":
                    model.save(cross_validation_model_path)  # creates a HDF5 file 
                else:
                    with open(cross_validation_model_path, "wb") as f:
                        cPickle.dump(clf, f, -1)
        
            f1score_cv[run-1,fold-1] = f1score_valid
            ## save this prediction
            #print raw_pred_valid_path
            #dfPred = pd.DataFrame({"target": Y_valid, "prediction": pred_raw})
            #dfPred.to_csv(raw_pred_valid_path, index=False, header=True,
            #             columns=["target", "prediction"])
            
            ## save this prediction
            #dfPred = pd.DataFrame({"target": Y_valid, "prediction": pred_class_bagging})
            #dfPred.to_csv(rank_pred_valid_path, index=False, header=True,
            #             columns=["target", "prediction"])
    f1score_cv_mean = np.mean(f1score_cv)
    f1score_cv_std = np.std(f1score_cv)
    if verbose_level >= 1:
        print("              Mean: %.6f" % f1score_cv_mean)
        print("              Std: %.6f" % f1score_cv_std)

    print ("----time elapsed----\n", str(timedelta(seconds=time.time() - start_time)))



 
    ####################
    #### Retraining ####
    ####################
    #### all the path
    path = "%s/All" % (feat_folder)
    save_path = "%s/All" % output_path
    subm_path = "%s/Subm" % output_path
    save_model_path = "%s/Model/All" % output_path
    if not os.path.exists(save_model_path):
        os.makedirs(save_model_path)

    if not os.path.exists(save_path):
        os.makedirs(save_path)
    if not os.path.exists(subm_path):
        os.makedirs(subm_path)

    raw_pred_test_path = "%s/test.raw.pred.%s_[Id@%d].csv1" % (save_path, feat_name, trial_counter)
    rank_pred_test_path = "%s/test.pred.%s_[Id@%d].csv" % (save_path, feat_name, trial_counter)

    # save trained model
    cross_validation_model_path = "%s/%s_[Id@%d].pkl" % (save_model_path, feat_name, trial_counter)
    cross_validation_model_scaler_path = "%s/%s_[Id@%d]_standardscaler.pkl" % (save_model_path, feat_name, trial_counter)

    ## load feat
    with open("%s/train_tfidf_cosine_sim.feat.pkl" % (path), "rb") as f:
          X_train_tfidf = cPickle.load(f)
    with open("%s/train_bow_cosine_sim.feat.pkl" % (path), "rb") as f:
          X_train_bow = cPickle.load(f)

    with open("%s/test_tfidf_cosine_sim.feat.pkl" % (path), "rb") as f:
          X_test_tfidf = cPickle.load(f)
    with open("%s/test_bow_cosine_sim.feat.pkl" % (path), "rb") as f:
          X_test_bow = cPickle.load(f)


    
    df_train = pd.read_csv('./data/train.csv')
    labels_train = df_train['cate_id'].values

    labels_test = [0]*X_test_bow.shape[0]
    X_train = np.hstack([X_train_tfidf, X_train_bow])
    X_test = np.hstack([X_test_tfidf, X_test_bow])

    #Y_valid = labels_valid
    numTrain = X_train.shape[0]
    numTest = X_test.shape[0]


    # feat
    #feat_train_path = "%s/train.feat" % path
    # save trained model
    all_training_model_path = "%s/%s_[Id@%d].pkl" % (save_model_path, feat_name, trial_counter)
    #all_training_model_scaler_path = "%s/%s_[Id@%d]_standardscaler.pkl" % (save_model_path, feat_name, trial_counter)

    #### load data
    #X_train, labels_train = load_svmlight_file(feat_train_path)
    #X_train = X_train.tocsr()
    #numTrain = X_train.shape[0]
      
    ## bagging
    #preds_bagging = np.zeros((numTest, bagging_size), dtype=float)
    run = 5
    fold =5
    for n in range(bagging_size):
        if bootstrap_replacement:
            sampleSize = int(numTrain*bootstrap_ratio)
            index_base = rng.randint(numTrain, size=sampleSize)
            index_meta = [i for i in range(numTrain) if i not in index_base]
        else:
            rng = np.random.RandomState(2015 + 1000 * run + 10 * fold)
            randnum = rng.uniform(size=numTrain)
            index_base = [i for i in range(numTrain) if randnum[i] < bootstrap_ratio]
            index_meta = [i for i in range(numTrain) if randnum[i] >= bootstrap_ratio]
 
        if param.has_key("booster"):
            
            dtrain = xgb.DMatrix(X_train[index_base], label=labels_train[index_base])
            dtest = xgb.DMatrix(X_test, label=labels_test)
   
            watchlist = []
            if verbose_level >= 2:
                watchlist  = [(dtrain, 'train')]
                    
        ## train
        if param["task"] in ["softmax"]:
            clf = xgb.train(param, dtrain, param['num_round'], watchlist)
            pred_prob = clf.predict(dtest)
            pred_label = np.argmax(pred_prob, axis=1)
            #error_rate = np.sum(pred != test_Y) / test_Y.shape[0]


        elif param['task'] == "reg_skl_rf":
            ## random forest regressor
            clf = RandomForestClassifier(n_estimators=param['n_estimators'],
                                       max_features=param['max_features'],
                                       n_jobs=param['n_jobs'],
                                       random_state=param['random_state'])
            clf.fit(X_train[index_base], labels_train[index_base])

        elif param['task'] == "reg_skl_etr":
            ## extra trees regressor
            clf = ExtraTreesClassifier(n_estimators=param['n_estimators'],
                                      max_features=param['max_features'],
                                      n_jobs=param['n_jobs'],
                                      random_state=param['random_state'])
            clf.fit(X_train[index_base], labels_train[index_base])

        elif param['task'] == "reg_skl_gbm":
            ## gradient boosting regressor
            clf = GradientBoostingClassifier(n_estimators=param['n_estimators'],
                                            max_features=param['max_features'],
                                            learning_rate=param['learning_rate'],
                                            max_depth=param['max_depth'],
                                            subsample=param['subsample'],
                                            random_state=param['random_state'])
            clf.fit(X_train.toarray()[index_base], labels_train[index_base])



        elif param['task'] == "clf_skl_lr":
            clf = LogisticRegression(penalty="l2", dual=True, tol=1e-5,
                                    C=param['C'], fit_intercept=True, intercept_scaling=1.0,
                                    class_weight='auto', random_state=param['random_state'])
            clf.fit(X_train[index_base], labels_train[index_base])

    # save the retrained classifer result
    #no bagging used for retraining 
    ## write
        # read data
    output = open('cate_label_map1.txt', 'rb')
    obj_dict = cPickle.load(output)


    df_test = pd.read_csv('./data/test.csv')
    pred_cate = [obj_dict[x] for x in pred_label]
    output = pd.DataFrame({"Judgements": df_test["Judgements"], "Area.of.Law": pred_cate})    
    output.to_csv(raw_pred_test_path, index=False)   
     
    print ("----time elapsed----\n", str(timedelta(seconds=time.time() - start_time)))
    return f1score_cv_mean, f1score_cv_std





    
####################
## Model Buliding ##
####################

def check_model(models, feat_name):
#    print "check_model models:", models
#    print "feat_name", feat_name
    if models == "all":
        return True
    for model in models:
        if model in feat_name:
            return True
    return False


if __name__ == "__main__":
    start_time = time.time()
    
    specified_models = sys.argv[1:]
    print specified_models
    '''
    if len(specified_models) == 0:
        print("You have to specify which model to train.\n")
        print("Usage: python ./train_model_library_lsa.py model1 model2 model3 ...\n")
        print("Example: python ./train_model_library_lsa.py reg_skl_ridge reg_skl_lasso reg_skl_svr\n")
        print("See model_library_lsa.py for a list of available models (i.e., Model@model_name)")
        sys.exit()



    '''
    feat_name = sys.argv[1]# "Model@reg_xgb_tree" 
    feat_folder = './output/feature'
    log_path = "%s/Log" % output_path
    print log_path
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    #for feat_name, feat_folder in zip(feat_names, feat_folders):
    for ii in [0]: 
        #if not check_model(specified_models, feat_name):
        #    continue
        #print feat_name, param_spaces
        print feat_name
        param_space = param_spaces[feat_name]
        #"""
        
        log_file = "%s/%s_hyperopt.log_new" % (log_path, feat_name)
        log_handler = open( log_file, 'wb' )
        writer = csv.writer( log_handler )
        headers = [ 'trial_counter', 'f1score_mean', 'f1score_std' ]
        for k,v in sorted(param_space.items()):
            headers.append(k)
        writer.writerow( headers )
        log_handler.flush()
        
        print("************************************************************")
        print("Search for the best params")
        #global trial_counter
        trial_counter = 0
        trials = Trials()
        objective = lambda p: hyperopt_wrapper(p, feat_folder, feat_name)
        best_params = fmin(objective, param_space, algo=tpe.suggest,
                           trials=trials, max_evals=param_space["max_evals"])
        for f in int_feat:
            if best_params.has_key(f):
                best_params[f] = int(best_params[f])
        print("************************************************************")
        print("Best params")
        for k,v in best_params.items():
            print "        %s: %s" % (k,v)
        trial_f1scores = -np.asarray(trials.losses(), dtype=float)
        best_f1score_mean = max(trial_f1scores)
        ind = np.where(trial_f1scores == best_f1score_mean)[0][0]
        best_f1score_std = trials.trial_attachments(trials.trials[ind])['std']
        print("F1 Score stats")
        print("        Mean: %.6f\n        Std: %.6f" % (best_f1score_mean, best_f1score_std))
