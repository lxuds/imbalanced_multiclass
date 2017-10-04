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
from sklearn.decomposition import TruncatedSVD

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

n_classes = 31
n_runs=5
df_train = pd.read_csv('./data/train.csv')
df_test = pd.read_csv('./data/test.csv')
num_train = df_train.shape[0]
num_test = df_test.shape[0]
#df = df_train.append(df_test, ignore_index=True)

#########################
# generate features

print("For cross-validation...")


with open("./data/stratifiedKFold.pkl", "rb") as f:
    skf = cPickle.load(f)

# generate SVD
svd_n_components = range(20, 150, 5)
feat_folder = './output/feature'
for n_components in svd_n_components:
    print 'n_components=', n_components
    for run in range(n_runs):
        for fold, (trainInd, validInd) in enumerate(skf[run]):
            print("Run: %d, Fold: %d" % (run+1, fold+1))
            path = "%s/Run%d/Fold%d" % (feat_folder, run+1, fold+1)
            df_train2 = df_train.iloc[trainInd].copy()
            df_valid = df_train.iloc[validInd].copy()
            print df_train2.shape, df_valid.shape

            with open("%s/train_tfidf.feat.pkl" % (path), "rb") as f:
                x_vec_tfidf_train = cPickle.load(f)        
            with open("%s/valid_tfidf.feat.pkl" % (path), "rb") as f:
                x_vec_tfidf_valid = cPickle.load(f)       
            with open("%s/train_bow.feat.pkl" % (path), "rb") as f:
                x_vec_bow_train = cPickle.load(f)
            with open("%s/valid_bow.feat.pkl" % (path), "rb") as f:
                x_vec_bow_valid = cPickle.load(f)           

            print "generate SVD"


            svd_tfidf = TruncatedSVD(n_components=n_components, n_iter=15)
            svd_tfidf.fit(x_vec_tfidf_train)
            x_tfidf_svd_train = svd_tfidf.transform(x_vec_tfidf_train)
            x_tfidf_svd_valid = svd_tfidf.transform(x_vec_tfidf_valid)
     
            svd_bow = TruncatedSVD(n_components=n_components, n_iter=15)
            svd_bow.fit(x_vec_bow_train)
            x_bow_svd_train = svd_bow.transform(x_vec_bow_train)
            x_bow_svd_valid = svd_bow.transform(x_vec_bow_valid)


            print "generate tfidf SVD cosine sim"
            # compute SVD cosine similarity
            # for tfidf
            area_indices_dict = get_sample_indices_by_area(df_train2)
            cosine_sim_stats_feat_tfidf_svd_by_area_train = generate_dist_stats_feat("cosine", x_tfidf_svd_train, df_train2["id"].values,
                                              x_tfidf_svd_train, df_train2["id"].values, area_indices_dict)

            cosine_sim_stats_feat_tfidf_svd_by_area_valid = generate_dist_stats_feat("cosine", x_tfidf_svd_train, df_train2["id"].values,
                                              x_tfidf_svd_valid, df_valid["id"].values, area_indices_dict)
            print "generate bow SVD cosine sim"
                    
            # for bow
            cosine_sim_stats_feat_bow_svd_by_area_train = generate_dist_stats_feat("cosine", x_bow_svd_train, df_train2["id"].values,
                                              x_bow_svd_train, df_train2["id"].values, area_indices_dict)
 
            cosine_sim_stats_feat_bow_svd_by_area_valid = generate_dist_stats_feat("cosine", x_bow_svd_train, df_train2["id"].values,
                                              x_bow_svd_valid, df_valid["id"].values, area_indices_dict)
            
            with open("%s/train_tfidf_svd%d_cosine_sim.feat.pkl" % (path, n_components), "wb") as f:
                cPickle.dump(cosine_sim_stats_feat_tfidf_svd_by_area_train, f, -1)
            with open("%s/valid_tfidf_svd%d_cosine_sim.feat.pkl" % (path,n_components), "wb") as f:
                cPickle.dump(cosine_sim_stats_feat_tfidf_svd_by_area_valid, f, -1)
 
            with open("%s/train_bow_svd%d_cosine_sim.feat.pkl" % (path, n_components), "wb") as f:
                cPickle.dump(cosine_sim_stats_feat_bow_svd_by_area_train, f, -1)
            
            with open("%s/valid_bow_svd%d_cosine_sim.feat.pkl" % (path,n_components), "wb") as f:
                cPickle.dump(cosine_sim_stats_feat_bow_svd_by_area_valid, f, -1)
        


    print ("Done.") 
    
print ("==================")
print ("generating SVD features for testing sets.")

path = "%s/All" % (feat_folder)

with open("%s/train_tfidf.feat.pkl" % (path), "rb") as f:
    x_vec_tfidf_train = cPickle.load(f)
with open("%s/test_tfidf.feat.pkl" % (path), "rb") as f:
    x_vec_tfidf_test = cPickle.load(f)

with open("%s/train_bow.feat.pkl" % (path), "rb") as f:
    x_vec_bow_train = cPickle.load(f)
with open("%s/test_bow.feat.pkl" % (path), "rb") as f:
    x_vec_bow_test = cPickle.load(f)

# generate SVD 
for n_components in svd_n_components:

    svd_tfidf = TruncatedSVD(n_components=n_components, n_iter=15)
    svd_tfidf.fit(x_vec_tfidf_train)
    x_tfidf_svd_train = svd_tfidf.transform(x_vec_tfidf_train)
    x_tfidf_svd_test = svd_tfidf.transform(x_vec_tfidf_test)

    svd_bow = TruncatedSVD(n_components=n_components, n_iter=15)
    svd_bow.fit(x_vec_bow_train)
    x_bow_svd_train = svd_bow.transform(x_vec_bow_train)
    x_bow_svd_test = svd_bow.transform(x_vec_bow_test)


    print "tfidf SVD cosine sim"
    # compute cosine similarity
    # for tfidf SVD
    area_indices_dict = get_sample_indices_by_area(df_train)
    cosine_sim_stats_feat_tfidf_svd_by_area_train = generate_dist_stats_feat("cosine", x_tfidf_svd_train, df_train["id"].values,
                                      x_tfidf_svd_train, df_train["id"].values, area_indices_dict)

    cosine_sim_stats_feat_tfidf_svd_by_area_test = generate_dist_stats_feat("cosine", x_tfidf_svd_train, df_train["id"].values,
                                      x_tfidf_svd_test, df_test["id"].values, area_indices_dict)
    print "bow SVD cosine sim"
 
    # for bow
    cosine_sim_stats_feat_bow_svd_by_area_train = generate_dist_stats_feat("cosine", x_bow_svd_train, df_train["id"].values,
                                      x_bow_svd_train, df_train["id"].values, area_indices_dict)
 
    cosine_sim_stats_feat_bow_svd_by_area_test = generate_dist_stats_feat("cosine", x_bow_svd_train, df_train["id"].values,
                                      x_bow_svd_test, df_test["id"].values, area_indices_dict)

    with open("%s/train_tfidf_svd%d_cosine_sim.feat.pkl" % (path, n_components), "wb") as f:
        cPickle.dump(cosine_sim_stats_feat_tfidf_svd_by_area_train, f, -1)
    with open("%s/test_tfidf_svd%d_cosine_sim.feat.pkl" % (path,n_components), "wb") as f:
        cPickle.dump(cosine_sim_stats_feat_tfidf_svd_by_area_test, f, -1)

    with open("%s/train_bow_svd%d_cosine_sim.feat.pkl" % (path,n_components), "wb") as f:
        cPickle.dump(cosine_sim_stats_feat_bow_svd_by_area_train, f, -1)
    with open("%s/test_bow_svd%d_cosine_sim.feat.pkl" % (path,n_components), "wb") as f:
        cPickle.dump(cosine_sim_stats_feat_bow_svd_by_area_test, f, -1)


    
    
    
    
