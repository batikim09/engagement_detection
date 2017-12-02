import numpy as np
np.random.seed(1337)
import itertools
import scipy.stats as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectFromModel

def one_hot_vector(num_classes, *y_s):
    
    result = []
    #expand class to class + 1, "class1,class2,class3" --> "class1,class2,class3,fake"
    for y in y_s:
        result.append(to_categorical(y, num_classes=num_classes))
    
    if len(result) == 1:
        return result[0]

    return result

def compose_idx(args_train_idx, args_test_idx, args_valid_idx, args_ignore_idx, args_adopt_idx, args_kf_idx):
    train_idx = []
    test_idx = []
    valid_idx = []
    ignore_idx = []
    adopt_idx = []
    kf_idx = []
    if args_train_idx:
        if ',' in args_train_idx:
            train_idx = args_train_idx.split(',')
        elif ':' in args_train_idx:
            indice = args_train_idx.split(':')
            for idx in range(int(indice[0]), int(indice[1]) + 1, +1):
                train_idx.append(idx)
        else:
            train_idx = args_train_idx.split(",")

    if args_test_idx:
        if ',' in args_test_idx:
            test_idx = args_test_idx.split(',')
        elif ':' in args_test_idx:
            indice = args_test_idx.split(':')
            for idx in range(int(indice[0]), int(indice[1]) + 1, +1):
                test_idx.append(idx)
        else:
            test_idx = args_test_idx.split(",")

    if args_ignore_idx:
        if ',' in args_ignore_idx:
            ignore_idx = args_ignore_idx.split(',')
        elif ':' in args_ignore_idx:
            indice = args_ignore_idx.split(':')
            for idx in range(int(indice[0]), int(indice[1]) + 1, +1):
                ignore_idx.append(idx)
        else:
            ignore_idx = args_ignore_idx.split(",")

    if args_valid_idx:
        if ',' in args_valid_idx:
            valid_idx = args_valid_idx.split(',')
        elif ':' in args_valid_idx:
            indice = args_valid_idx.split(':')
            for idx in range(int(indice[0]), int(indice[1]) + 1, +1):
                valid_idx.append(idx)
        else:
            valid_idx = args_valid_idx.split(",")
    if args_adopt_idx:
            if ',' in args_adopt_idx:
                adopt_idx = args_adopt_idx.split(',')
            elif ':' in args_adopt_idx:
                indice = args_adopt_idx.split(':')
                for idx in range(int(indice[0]), int(indice[1]) + 1, +1):
                    adopt_idx.append(idx)
            else:
                adopt_idx = args_adopt_idx.split(",")
    if args_kf_idx:
        kf_idx = args_kf_idx.split(",")
    kf_idx = set(kf_idx)

    return train_idx, test_idx, valid_idx, ignore_idx, adopt_idx, kf_idx


def log(log, logger = None):
    if logger is not None:
        logger.write(log + "\n")
    
    print(log)    

def feature_analysis(train_csv, train_lab, args, log_writer):
    X = train_csv
    Y = train_lab[:, (args.class_idx, args.window_idx)]
    Y = Y.astype(int)

    if args.spearmanr:
        method = st.spearmanr
    else:
        method = st.kendalltau

    corr_feat = np.zeros(X.shape[1])

    for f in range(X.shape[1]):
        if args.pairwise:
            X_p, Y_p = transform_pairwise(X[:, f], Y)
            cor, p_value = method(X_p, Y_p)
        else:
            cor, p_value = method(X[:, f], Y[:, 0])

        corr_feat[f] = cor
        log("feature:\t%d\tcor:\t%.3f\tp-value:\t%.3f" %(f, cor, p_value), log_writer)
    
    log("feature cor: %s" %(str(corr_feat)), log_writer)

def compose_data_set(test_idx, valid_idx, train_idx, train_csv, train_lab, start_indice, end_indice):
    

    test_indice = []
    valid_indice = []
    remove_indice = []


    for cid in test_idx:
        print("cross-validation test: ", cid)
        start_idx = start_indice[int(cid)]
        end_idx = end_indice[int(cid)]

        if start_idx == 0 and end_idx == 0:
            continue

        for idx in range(int(start_idx), int(end_idx), + 1):
            test_indice.append(idx)
            remove_indice.append(idx)

    for cid in valid_idx:
        print("cross-validation valid: ", cid)
        start_idx = start_indice[int(cid)]
        end_idx = end_indice[int(cid)]

        if start_idx == 0 and end_idx == 0:
            continue

        for idx in range(int(start_idx), int(end_idx), + 1):
            remove_indice.append(idx)
            valid_indice.append(idx)

    if len(train_idx):
        train_indice = []
        for cid in train_idx:
            print("cross-validation train: ", cid)
            start_idx = start_indice[int(cid)]
            end_idx = end_indice[int(cid)]

            if start_idx == 0 and end_idx == 0:
                continue

            for idx in range(int(start_idx), int(end_idx), + 1):
                train_indice.append(idx)

        X_train = train_csv[train_indice]  
        Y_train = train_lab[train_indice]
    else:
        X_train = np.delete(train_csv, remove_indice, axis=0)
        Y_train = np.delete(train_lab, remove_indice, axis=0)

    #test set
    X_test = train_csv[test_indice]  
    Y_test = train_lab[test_indice]

    #valid set
    if len(valid_indice) == 0:
        X_valid = X_test
        Y_valid = Y_test
    else:
        X_valid = train_csv[valid_indice]  
        Y_valid = train_lab[valid_indice]
    
    print('train shape: ', X_train.shape)
    print('test shape: ', X_test.shape)
    print('validatation shape: ', X_valid.shape)

    return X_train, X_test, X_valid, Y_train, Y_test, Y_valid

def compose_multi_data_set(test_idx, valid_idx, train_idx, train_csv, train_lab, start_indice, end_indice):
    

    test_indice = []
    valid_indice = []
    remove_indice = []


    for cid in test_idx:
        print("cross-validation test: ", cid)
        start_idx = start_indice[int(cid)]
        end_idx = end_indice[int(cid)]

        if start_idx == 0 and end_idx == 0:
            continue

        for idx in range(int(start_idx), int(end_idx), + 1):
            test_indice.append(idx)
            remove_indice.append(idx)

    for cid in valid_idx:
        print("cross-validation valid: ", cid)
        start_idx = start_indice[int(cid)]
        end_idx = end_indice[int(cid)]

        if start_idx == 0 and end_idx == 0:
            continue

        for idx in range(int(start_idx), int(end_idx), + 1):
            remove_indice.append(idx)
            valid_indice.append(idx)

    if len(train_idx):
        train_indice = []
        for cid in train_idx:
            print("cross-validation train: ", cid)
            start_idx = start_indice[int(cid)]
            end_idx = end_indice[int(cid)]

            if start_idx == 0 and end_idx == 0:
                continue

            for idx in range(int(start_idx), int(end_idx), + 1):
                train_indice.append(idx)

        X_train = []
        for s_train_csv in train_csv:
            X_train.append(s_train_csv[train_indice])

        Y_train = train_lab[train_indice]
    else:
        X_train = []
        for s_train_csv in train_csv:
            X_train.append(np.delete(s_train_csv, remove_indice, axis=0))

        Y_train = np.delete(train_lab, remove_indice, axis=0)

    #test set
    X_test = []
    for s_train_csv in train_csv:
        X_test.append(s_train_csv[test_indice])
        
    Y_test = train_lab[test_indice]
    
    #valid set
    if len(valid_indice) == 0:
        X_valid = X_test
            
        Y_valid = Y_test
    else:
        X_valid = []
        for s_train_csv in train_csv:
            X_valid.append(s_train_csv[valid_indice])
                   
        Y_valid = train_lab[valid_indice]
    
    for idx in range(len(X_train)):
        print('train shape: ', X_train[idx].shape)
        print('test shape: ', X_test[idx].shape)
        print('validatation shape: ', X_valid[idx].shape)

    return X_train, X_test, X_valid, Y_train, Y_test, Y_valid

def feature_selection(X_train, X_test, X_valid, Y_train, Y_test, Y_valid,args, log_writer):

    if args.fs_by_valid:
        X_temp = X_valid
        Y_temp = Y_valid
    else:
        X_temp = X_train
        Y_temp = Y_train
    
    if args.pairwise:
        log("Pairwise transform", log_writer)
        X_temp, Y_temp = transform_pairwise(X_temp, Y_temp)

    if args.fs_by_rf:
        log("selected feature by RF", log_writer)
        selected_feat_indice = select_feature_idx_by_RF(X_temp, Y_temp, args, log_writer)
    else:
        log("selected feature by correlation", log_writer)
        selected_feat_indice = select_feature_by_cor(X_temp, Y_temp, args, log_writer)

    X_train = X_train[:,selected_feat_indice]
    X_test = X_test[:,selected_feat_indice]
    X_valid = X_valid[:,selected_feat_indice]

    log("selected feature dim: %d" %(len(selected_feat_indice)), log_writer)
    log("selected feature indice: %s" % str(selected_feat_indice), log_writer)

    return X_train, X_test, X_valid

def select_feature_by_cor(X_temp, Y_temp, args, log_writer):

    corr_feat = np.zeros(X_temp.shape[1])
    if args.spearmanr:
        method = st.spearmanr
    else:
        method = st.kendalltau

    for f in range(X_temp.shape[1]):
        cor, p_value = method(X_temp[:, f], Y_temp[:])

        corr_feat[f] = 1. - np.absolute(cor)
            
    log("feature : %s" %(str(corr_feat)), log_writer)

    #sort and return indice
    selected_feat_indice = np.argsort(corr_feat)

    return selected_feat_indice[0:args.max_feat]
    

def select_feature_idx_by_RF(X_temp, Y_temp, args, log_writer):
    clf = RandomForestClassifier(n_estimators=args.rf_n_estimator, random_state=0, n_jobs=-1)

    # Train the classifier
    clf.fit(X_temp, Y_temp)

    for feature in zip([x for x in range(X_temp.shape[1])], clf.feature_importances_):
        log(str(feature), log_writer)

    corr_feat = 1. - clf.feature_importances_
    selected_feat_indice = np.argsort(corr_feat)

    return selected_feat_indice[0:args.max_feat]

    #sfm = SelectFromModel(clf, threshold=args.rf_threshold)
    # Train the selector
    #sfm.fit(X_temp, Y_temp)
    #return sfm.get_support(indices=True)


def transform_pairwise_multi_data(X_s, y):

    result_X_s = []
    for X in X_s:
        paired_X, paired_y = transform_pairwise(X, y)
        result_X_s.append(paired_X)

    return result_X_s, paired_y

def transform_pairwise(X, y):
    """Transforms data into pairs with balanced labels for ranking

    Transforms a n-class ranking problem into a two-class classification
    problem. Subclasses implementing particular strategies for choosing
    pairs should override this method.

    In this method, all pairs are choosen, except for those that have the
    same target value. The output is an array of balanced classes, i.e.
    there are the same number of -1 as +1

    Parameters
    ----------
    X : array, shape (n_samples, n_features)
        The data
    y : array, shape (n_samples,) or (n_samples, 2)
        Target labels. If it's a 2D array, the second column represents
        the grouping of samples, i.e., samples with different groups will
        not be considered.

    Returns
    -------
    X_trans : array, shape (k, n_feaures)
        Data as pairs
    y_trans : array, shape (k,)
        Output class labels, where classes have values {-1, +1}
    """
    X_new = []
    y_new = []
    y = np.asarray(y)
    if y.ndim == 1:
        y = np.c_[y, np.ones(y.shape[0])]
    comb = itertools.combinations(range(X.shape[0]), 2)
    for k, (i, j) in enumerate(comb):
        if y[i, 0] == y[j, 0] or y[i, 1] != y[j, 1]:
            # skip if same target or different group
            continue
        X_new.append(X[i] - X[j])
        y_new.append(np.sign(y[i, 0] - y[j, 0]))
        # output balanced classes
        if y_new[-1] != (-1) ** k:
            y_new[-1] = - y_new[-1]
            X_new[-1] = - X_new[-1]
    return np.asarray(X_new), np.asarray(y_new).ravel()
