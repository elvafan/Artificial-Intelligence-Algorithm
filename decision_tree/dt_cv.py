# version 1.1
from typing import List

from numpy.core.records import array

import dt_global
from dt_core import *


def cv_pre_prune(folds: List, value_list: List[float]) -> (List[float], List[float]):
    """
    Determines the best parameter value for pre-pruning via cross validation.

    Returns two lists: the training accuracy list and the validation accuracy list.

    :param folds: folds for cross validation
    :type folds: List[List[List[Any]]]
    :param value_list: a list of parameter values
    :type value_list: List[float]
    :return: the training accuracy list and the validation accuracy list
    :rtype: List[float], List[float]
    """  
    num_para = len(value_list)
    training_accuracy_list = [0]*num_para
    validation_accuracy_list = [0]*num_para
    features = dt_global.feature_names[:-1]
    for i in range(len(folds)):
        validation_set = folds[i]
        training_set = []
        for j in range(len(folds)):
            if i != j:
                training_set = training_set + folds[j]
        #build decision tree using training set
        root = learn_dt(training_set,features)
        for index in range(len(value_list)):
            depth = value_list[index]
            pa = get_prediction_accuracy(root, training_set,max_depth=depth)
            training_accuracy_list[index] += pa
            pa = get_prediction_accuracy(root, validation_set,max_depth=depth)
            validation_accuracy_list[index] += pa
    num_fold = len(folds)
    training_accuracy_list = [i/num_fold for i in training_accuracy_list]
    validation_accuracy_list = [i/num_fold for i in validation_accuracy_list]
    return training_accuracy_list, validation_accuracy_list


def cv_post_prune(folds: List, value_list: List[float]) -> (List[float], List[float]):
    """
    Determines the best parameter value for post-pruning via cross validation.

    Returns two lists: the training accuracy list and the validation accuracy list.

    :param folds: folds for cross validation
    :type folds: List[List[List[Any]]]
    :param value_list: a list of parameter values
    :type value_list: List[float]
    :return: the training accuracy list and the validation accuracy list
    :rtype: List[float], List[float]
    """ 
    num_para = len(value_list)
    training_accuracy_list = [0]*num_para
    validation_accuracy_list = [0]*num_para
    features = dt_global.feature_names[:-1]
    for i in range(len(folds)):
        validation_set = folds[i]
        training_set = []
        for j in range(len(folds)):
            if i != j:
                training_set = training_set + folds[j]
        #build decision tree using training set
        root = learn_dt(training_set,features)
        for index in range(len(value_list)):
            minnum = value_list[index]
            pa = get_prediction_accuracy(root, training_set,min_num_examples=minnum)
            training_accuracy_list[index] += pa
            pa = get_prediction_accuracy(root, validation_set,min_num_examples=minnum)
            validation_accuracy_list[index] += pa
    num_fold = len(folds)
    training_accuracy_list = [i/num_fold for i in training_accuracy_list]
    validation_accuracy_list = [i/num_fold for i in validation_accuracy_list]
    return training_accuracy_list, validation_accuracy_list

