# -*- coding: <coding> -*-
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics.pairwise import cosine_similarity, pairwise_distances
from sklearn.datasets import dump_svmlight_file
import cPickle

import nltk
from nlp_utils import * 
import os,sys
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from scipy.sparse import hstack, vstack

#from sklearn.model_selection import StratifiedKFold

from sklearn.cross_validation import StratifiedKFold


def get_sample_indices_by_area(dfTrain, additional_key=None):
	""" 
		return a dict with
		key: (additional_key, median_relevance)
		val: list of sample indices
	"""
	dfTrain["sample_index"] = range(dfTrain.shape[0])
	group_key = ["cate_id"]
	#if additional_key != None:
	#	group_key.insert(0, additional_key)
	agg = dfTrain.groupby(group_key, as_index=False).apply(lambda x: list(x["sample_index"]))
	d = dict(agg)
	dfTrain = dfTrain.drop("sample_index", axis=1)
	return d



#####################
## Helper function ##
#####################
## compute cosine similarity
def cosine_sim(x, y):
    try:
        d = cosine_similarity(x, y)
        d = d[0][0]
    except:
        print x
        print y
        d = 0.
    return d


quantiles_range = np.arange(0, 1.5, 0.5)
stats_func = [ np.mean, np.std ]
stats_feat_num = len(quantiles_range) + len(stats_func)

## generate distance stats feat
n_classes = 41
def generate_dist_stats_feat(metric, X_train, ids_train, X_test, ids_test, indices_dict, qids_test=None):
    if metric == "cosine":
        stats_feat = 0 * np.ones((len(ids_test), stats_feat_num*n_classes), dtype=float)
        sim = 1. - pairwise_distances(X_test, X_train, metric=metric, n_jobs=1)
    elif metric == "euclidean":
        stats_feat = -1 * np.ones((len(ids_test), stats_feat_num*n_classes), dtype=float)
        sim = pairwise_distances(X_test, X_train, metric=metric, n_jobs=1)
    for i in range(len(ids_test)):
    #for i in [0]: 
        id = ids_test[i]
        #print id
        if qids_test is not None:
            qid = qids_test[i]
        for j in range(n_classes):
        #for j in [0]:
            key = (qid, j) if qids_test is not None else j
            #print 'key=', key
            if indices_dict.has_key(key):
                inds = indices_dict[key]
                # exclude this sample itself from the list of indices
                inds = [ ind for ind in inds if id != ids_train[ind] ]
                sim_tmp = sim[i][inds]
                if len(sim_tmp) != 0:
                    feat = [ func(sim_tmp) for func in stats_func ]
                    ## quantile
                    sim_tmp = pd.Series(sim_tmp)
                    quantiles = sim_tmp.quantile(quantiles_range)
                    feat = np.hstack((feat, quantiles))
                    stats_feat[i,j*stats_feat_num:(j+1)*stats_feat_num] = feat
                    #print  'key=', key, feat
    return stats_feat



if __name__ == "__main__":
    
    feat_folder = './output/feature'
    
    df_all = pd.read_csv('./data/Interview_Mapping.csv')
    #num_all = df_all.shape[0],
    #df_all["index"] = np.arange(num_all)
    
    
    
    df_train = df_all.loc[df_all['Area.of.Law'] !='To be Tested']
    num_train = df_train.shape[0]
    df_train["id"] = np.arange(num_train)
    
    df_test = df_all.loc[df_all['Area.of.Law'] =='To be Tested']
    num_test = df_test.shape[0] 
    df_test["id"] = np.arange(num_test) + num_train

    df_test.to_csv('./data/test.csv', index=False)


    df_train.groupby(df_train['Area.of.Law']).value_counts()
    
    mlb = MultiLabelBinarizer()
    Y = mlb.fit_transform([[x] for x in df_train['Area.of.Law'].values])
    #df_train['Y_cate'] = 0 #np.zeros((df_train.shape[0])) 
    xx = []
    for j, t in enumerate(Y):
       xx.append([i for i, x in enumerate(t) if x][0])
       
    df_train['cate_id'] = xx
    #factorize
    d ={}

    '''
    for i in range(num_train):
        if df_train['Area.of.Law'].iloc[i] in d:
            pass
        if df_train['Area.of.Law'].iloc[i] not in d:
            d[df_train['Area.of.Law'].iloc[i]] = df_train['cate_id'].iloc[i]
    '''
    for i in range(num_train):
        if df_train['cate_id'].iloc[i] in d:
            pass
        if df_train['cate_id'].iloc[i] not in d:
            d[df_train['cate_id'].iloc[i]] = df_train['Area.of.Law'].iloc[i]

    print d
    print d.values()
    output = open('cate_label_map.txt', 'ab+')
    cPickle.dump(d, output)
    output.close()

    # read data
    output = open('cate_label_map.txt', 'rb')
    obj_dict = cPickle.load(output) 
    print obj_dict
    pred_cate = [obj_dict[x] for x in df_train['cate_id']]

    print [[pred_cate[i], df_train["Area.of.Law"].iloc[i]] for i in range(num_train)]
    print 

    '''
    n_runs = 5
    n_folds = 5
    skf = [0]*n_runs
    
    for run in range(n_runs):
        random_seed = 2015 + 1000 * (run+1)
        skf[run] = StratifiedKFold(df_train['cate_id'], n_folds=n_folds,shuffle=True, random_state=random_seed)
        #skf_result[run] = skf.split(Y, df_train['cate_id'])
        for fold, (trainInd, validInd) in enumerate(skf[run]):
            print("================================")
            print("Index for run: %s, fold: %s" % (run+1, fold+1))
            print("Train (num = %s)" % len(trainInd))
            print(trainInd[:10])
            print("Valid (num = %s)" % len(validInd))
            print(validInd[:10])
    

    with open("./data/stratifiedKFold.pkl", "wb") as f:
        cPickle.dump(skf, f, -1)
    
    
    #df_train.iloc[:,4:] = Y
    
    df_train.to_csv('./data/train.csv', index=False)
    df_train['cate_id'].to_csv('./data/train_check.csv', index=False)
    
    with open("./data/stratifiedKFold.pkl", "rb") as f:
        skf = cPickle.load(f)
    
    
    
    print("For cross-validation...")
    for run in range(n_runs):
        print run
        for fold, (trainInd, validInd) in enumerate(skf[run]):
            print("Run: %d, Fold: %d" % (run+1, fold+1))
            path = "%s/Run%d/Fold%d" % (feat_folder, run+1, fold+1)
            if not os.path.exists(path):
                os.makedirs(path)
            df_train2 = df_train.iloc[trainInd].copy()
            df_valid = df_train.iloc[validInd].copy()
            print df_train2.shape, df_valid.shape
    
            # clean text
            print "clean text"
            x_train2  = np.array(map(read_clean_text, df_train2['Judgements']))
            x_valid  = np.array(map(read_clean_text, df_valid['Judgements']))
            print "tfidf"
            vec_tfidf = getTFV()
            x_vec_tfidf_train = vec_tfidf.fit_transform(x_train2)
            vec_tfidf.vocabulary_
            x_vec_tfidf_valid = vec_tfidf.transform(x_valid)
            print "bow"        
            vec_bow = getBOW()
            x_vec_bow_train = vec_bow.fit_transform(x_train2)
            x_vec_bow_valid = vec_bow.transform(x_valid)
            
            with open("%s/train_tfidf.feat.pkl" % (path), "wb") as f:
                cPickle.dump(x_vec_tfidf_train, f, -1)        
            with open("%s/valid_tfidf.feat.pkl" % (path), "wb") as f:
                cPickle.dump(x_vec_tfidf_valid, f, -1)        

            with open("%s/train_bow.feat.pkl" % (path), "wb") as f:
                cPickle.dump(x_vec_bow_train, f, -1)
            with open("%s/valid_bow.feat.pkl" % (path), "wb") as f:
                cPickle.dump(x_vec_bow_valid, f, -1)
            
            print "tfidf cosine sim"
            # compute cosine similarity
            # for tfidf
            area_indices_dict = get_sample_indices_by_area(df_train2)
            cosine_sim_stats_feat_tfidf_by_area_train = generate_dist_stats_feat("cosine", x_vec_tfidf_train, df_train2["id"].values,
                                              x_vec_tfidf_train, df_train2["id"].values, area_indices_dict)

            cosine_sim_stats_feat_tfidf_by_area_valid = generate_dist_stats_feat("cosine", x_vec_tfidf_train, df_train2["id"].values,
                                              x_vec_tfidf_valid, df_valid["id"].values, area_indices_dict)
            print "bow cosine sim"
                    
            # for bow
            cosine_sim_stats_feat_bow_by_area_train = generate_dist_stats_feat("cosine", x_vec_bow_train, df_train2["id"].values,
                                              x_vec_bow_train, df_train2["id"].values, area_indices_dict)
 
            cosine_sim_stats_feat_bow_by_area_valid = generate_dist_stats_feat("cosine", x_vec_bow_train, df_train2["id"].values,
                                              x_vec_bow_valid, df_valid["id"].values, area_indices_dict)
            
            with open("%s/train_tfidf_cosine_sim.feat.pkl" % (path), "wb") as f:
                cPickle.dump(cosine_sim_stats_feat_tfidf_by_area_train, f, -1)
            with open("%s/valid_tfidf_cosine_sim.feat.pkl" % (path), "wb") as f:
                cPickle.dump(cosine_sim_stats_feat_tfidf_by_area_valid, f, -1)
 
            with open("%s/train_bow_cosine_sim.feat.pkl" % (path), "wb") as f:
                cPickle.dump(cosine_sim_stats_feat_bow_by_area_train, f, -1)
            
            with open("%s/valid_bow_cosine_sim.feat.pkl" % (path), "wb") as f:
                cPickle.dump(cosine_sim_stats_feat_bow_by_area_valid, f, -1)
        


    print ("Done.") 

    
    
    # generate labels for each fold
    print ("Generateing labels for each Run/Fold")
    for run in range(n_runs):
        print run
        for fold, (trainInd, validInd) in enumerate(skf[run]):
    
            print("Run: %d, Fold: %d" % (run+1, fold+1))
            path = "%s/Run%d/Fold%d" % (feat_folder, run+1, fold+1)
            if not os.path.exists(path):
                os.makedirs(path)
            Y_train2 = Y[trainInd]
            Y_valid = Y[validInd]
    
        print Y_train2.shape, Y_valid.shape
        with open("%s/train_label.pkl" % (path), "wb") as f:
            cPickle.dump(Y_train2, f, -1)
        with open("%s/valid_label.pkl" % (path), "wb") as f:
            cPickle.dump(Y_valid, f, -1)

        Y_train2 = df_train['cate_id'].iloc[trainInd].values
        Y_valid = df_train['cate_id'].iloc[validInd].values
        print Y_train2
        #print trainInd
        #print df_train['cate_id'].values
        print Y_train2.shape, Y_valid.shape
        with open("%s/train_label_scalar.pkl" % (path), "wb") as f:
            cPickle.dump(Y_train2, f, -1)
        with open("%s/valid_label_scalar.pkl" % (path), "wb") as f:
            cPickle.dump(Y_valid, f, -1)
    print("Done.")






    print ("generating features for testing sets.")

    path = "%s/All" % (feat_folder)
    if not os.path.exists(path):
        os.makedirs(path)

    ## generating features for testing sets
    # clean text
    print "clean text"
    x_train  = np.array(map(read_clean_text, df_train['Judgements']))
    x_test  = np.array(map(read_clean_text, df_test['Judgements']))
    print "tfidf"
    vec_tfidf = getTFV()
    x_vec_tfidf_train = vec_tfidf.fit_transform(x_train)
    vec_tfidf.vocabulary_
    x_vec_tfidf_test = vec_tfidf.transform(x_test)
    print "bow"
    vec_bow = getBOW()
    x_vec_bow_train = vec_bow.fit_transform(x_train)
    x_vec_bow_test = vec_bow.transform(x_test)

    with open("%s/train_tfidf.feat.pkl" % (path), "wb") as f:
        cPickle.dump(x_vec_tfidf_train, f, -1)
    with open("%s/test_tfidf.feat.pkl" % (path), "wb") as f:
        cPickle.dump(x_vec_tfidf_test, f, -1)

    with open("%s/train_bow.feat.pkl" % (path), "wb") as f:
        cPickle.dump(x_vec_bow_train, f, -1)
    with open("%s/test_bow.feat.pkl" % (path), "wb") as f:
        cPickle.dump(x_vec_bow_test, f, -1)

    print "tfidf cosine sim"
    # compute cosine similarity
    # for tfidf
    area_indices_dict = get_sample_indices_by_area(df_train)
    cosine_sim_stats_feat_tfidf_by_area_train = generate_dist_stats_feat("cosine", x_vec_tfidf_train, df_train["id"].values,
                                      x_vec_tfidf_train, df_train["id"].values, area_indices_dict)

    cosine_sim_stats_feat_tfidf_by_area_test = generate_dist_stats_feat("cosine", x_vec_tfidf_train, df_train["id"].values,
                                      x_vec_tfidf_test, df_test["id"].values, area_indices_dict)
    print "bow cosine sim"

    # for bow
    cosine_sim_stats_feat_bow_by_area_train = generate_dist_stats_feat("cosine", x_vec_bow_train, df_train["id"].values,
                                      x_vec_bow_train, df_train["id"].values, area_indices_dict)

    cosine_sim_stats_feat_bow_by_area_test = generate_dist_stats_feat("cosine", x_vec_bow_train, df_train["id"].values,
                                      x_vec_bow_test, df_test["id"].values, area_indices_dict)

    with open("%s/train_tfidf_cosine_sim.feat.pkl" % (path), "wb") as f:
        cPickle.dump(cosine_sim_stats_feat_tfidf_by_area_train, f, -1)
    with open("%s/test_tfidf_cosine_sim.feat.pkl" % (path), "wb") as f:
        cPickle.dump(cosine_sim_stats_feat_tfidf_by_area_test, f, -1)

    with open("%s/train_bow_cosine_sim.feat.pkl" % (path), "wb") as f:
        cPickle.dump(cosine_sim_stats_feat_bow_by_area_train, f, -1)
    with open("%s/test_bow_cosine_sim.feat.pkl" % (path), "wb") as f:
        cPickle.dump(cosine_sim_stats_feat_bow_by_area_test, f, -1)






'''

