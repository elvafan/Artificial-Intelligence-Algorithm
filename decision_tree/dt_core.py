# version 1.1
import math
from typing import List
from anytree import Node

import dt_global 

global_name = 0
def get_name():
    global global_name
    global_name += 1
    return str(global_name)



def get_splits(examples: List, feature: str) -> List[float]:
    """
    Given some examples and a feature, returns a list of potential split point values for the feature.
    
    :param examples: a set of examples
    :type examples: List[List[Any]]
    :param feature: a feature
    :type feature: str
    :return: a list of potential split point values 
    :rtype: List[float]
    """ 
    #initial a list of potential split point values
    potential_split_point = []
    #find the feature index
    try:
        feature_index = dt_global.feature_names.index(feature)
    except ValueError as e:
        print(repr(e))
        raise
    #Sort the instances according to the real-valued feature.
    examples = sorted(examples, key = lambda x:x[feature_index])
    #print(examples)
    for i in range(len(examples)-1):
        x = examples[i][feature_index]
        y = examples[i+1][feature_index]
        #print('x',x,'y',y)
        if math.isclose(x,y,abs_tol=1e-5):
            continue
        else:
            #print('diff')
            #find all Lx that have value x
            Lx = [examples[i][dt_global.label_index]]
            for j in range(i-1, -1,-1):
                if examples[j][feature_index] == x:
                    Lx.append(examples[j][feature_index])
                else:
                    break
            #find all Ly that have value y
            Ly = [examples[i+1][dt_global.label_index]]
            for j in range(i+2, len(examples)):
                if examples[j][feature_index] == y:
                    Ly.append(examples[j][feature_index])
                else:
                    break
            #If there exists a label a ∈ LX and a label b ∈ LY such that a 6= b, then (X + Y )/2 is a possible split point.
            flag = 0
            #print(Lx)
            #print(Ly)
            for a in Lx:
                if flag == 1:
                    break
                for b in Ly:
                    if a != b:
                        #print('a',a,'b',b)
                        potential_split_point.append((x+y)/2)
                        flag = 1
                        #print('split  x',x,'y',y)
                        break
    return potential_split_point


def calculate_I(examples:List)->float:
    lst_label =[]
    label_counter = []
    for ex in examples:
        if ex[dt_global.label_index] in lst_label:
            label_counter[lst_label.index(ex[dt_global.label_index])] += 1
        else:
            lst_label.append(ex[dt_global.label_index])
            label_counter.append(1)
    total = 0
    for i in label_counter:            
        pi = i/len(examples)
        #print('pi',i,len(examples))
        total += pi * math.log2(pi)
    total = 0 - total
    #print('total',total)
    return total

def choose_feature_split(examples: List, features: List[str]) -> (str, float):
    """
    Given some examples and some features,
    returns a feature and a split point value with the max expected information gain.

    If there are no valid split points for the remaining features, return None and -1.

    Tie breaking rules:
    (1) With multiple split points, choose the one with the smallest value. 
    (2) With multiple features with the same info gain, choose the first feature in the list.

    :param examples: a set of examples
    :type examples: List[List[Any]]    
    :param features: a set of features
    :type features: List[str]
    :return: the best feature and the best split value
    :rtype: str, float
    """  
    Hbefore = calculate_I(examples)
    chosen_feature = None
    chosen_split_point = -1
    max_gain = 0
    total_num = len(examples)
    for f in features:
        potential_split_point = get_splits(examples, f)
        temp_sp = -1
        temp_max = 0
        for sp in potential_split_point:
            left, right = split_examples(examples,f,sp)
            EIafter = (len(left)/total_num)*calculate_I(left) + (len(right)/total_num)*calculate_I(right)
            gain = Hbefore -  EIafter
            #print(gain)
            if gain > temp_max:
                temp_max = gain
                temp_sp = sp
            if math.isclose(gain,max_gain,abs_tol=1e-5):
                if sp < temp_sp:
                    temp_sp = sp
        if temp_max > max_gain:
            chosen_feature = f
            chosen_split_point = temp_sp
            max_gain = temp_max
            if math.isclose(gain,max_gain,abs_tol=1e-5):
                if dt_global.feature_names.index(f) < dt_global.feature_names.index(chosen_feature):
                        chosen_feature = f
                        chosen_split_point = temp_sp
    return chosen_feature,chosen_split_point

def split_examples(examples: List, feature: str, split: float) -> (List, List):
    """
    Given some examples, a feature, and a split point,
    splits examples into two lists and return the two lists of examples.

    The first list of examples have their feature value <= split point.
    The second list of examples have their feature value > split point.

    :param examples: a set of examples
    :type examples: List[List[Any]]
    :param feature: a feature
    :type feature: str
    :param split: the split point
    :type split: float
    :return: two lists of examples split by the feature split
    :rtype: List[List[Any]], List[List[Any]]
    """
    first_list = []
    second_list = []
    #find the feature index
    try:
        feature_index = dt_global.feature_names.index(feature)
    except ValueError:
        print('feature not present split_example', feature)
    for ex in examples:
        if math.isclose(ex[feature_index],split,abs_tol=1e-5) or ex[feature_index] < split:
            first_list.append(ex)
        elif ex[feature_index] > split:
            second_list.append(ex)
    return first_list,second_list

def majority(examples):
    max_count = 0
    max_label = -1
    lst_label = []
    for ex in examples:
        lst_label.append(ex[dt_global.label_index])
    for label in lst_label:
        count = lst_label.count(label)
        if count > max_count:
            max_count = count
            max_label = label
        if count == max_count:
            if label < max_label:
                max_label = label
    return max_label

def same_class(examples):
    if len(examples) == 0:
        return False
    else:
        label = examples[0][dt_global.label_index]
        for ex in examples:
            if label != ex[dt_global.label_index]:
                return False
        return True

def split_node(cur_node: Node, examples: List, features: List[str], max_depth=math.inf):
    """
    Given a tree with cur_node as the root, some examples, some features, and the max depth,
    grows a tree to classify the examples using the features by using binary splits.

    If cur_node is at max_depth, makes cur_node a leaf node with majority decision and return.

    This function is recursive.

    :param cur_node: current node
    :type cur_node: Node
    :param examples: a set of examples
    :type examples: List[List[Any]]
    :param features: a set of features
    :type features: List[str]
    :param max_depth: the maximum depth of the tree
    :type max_depth: int
    """ 
    # all of the examples belong to the same class
    if same_class(examples):
        cur_node.feature = None
        cur_node.split = None
        cur_node.decision = examples[0][dt_global.label_index]
    # no features left
    elif len(features) == 0:
        cur_node.feature = None
        cur_node.split = None
        cur_node.decision = majority(examples)
    elif len(examples) == 0:
        cur_node.feature = None
        cur_node.split = None
        cur_node.decision = cur_node.parent.majority
    elif cur_node.depth == max_depth:
        cur_node.feature = None
        cur_node.split = None
        cur_node.decision = majority(examples)
    else:
        cur_node.majority = majority(examples)
        left, right = split_examples(examples, cur_node.feature,cur_node.split)
        f, sp = choose_feature_split(left,features)
        leftnode = Node(name= get_name(), parent = cur_node, feature = f,split = sp,num_examples = len(left),decision = None)
        split_node(leftnode,left,features,max_depth)
        f, sp = choose_feature_split(right,features)
        rightnode = Node(name= get_name(), parent = cur_node, feature = f,split = sp,num_examples = len(right),decision = None)
        split_node(rightnode,right,features,max_depth)



def learn_dt(examples: List, features: List[str], max_depth=math.inf) -> Node:
    """
    Given some examples, some features, and the max depth,
    creates the root of a decision tree, and
    calls split_node to grow the tree to classify the examples using the features, and
    returns the root node.

    This function is a wrapper for split_node.

    Tie breaking rule:
    If there is a tie for majority voting, always return the label with the smallest value.

    :param examples: a set of examples
    :type examples: List[List[Any]]
    :param features: a set of features
    :type features: List[str]
    :param max_depth: the max depth of the tree
    :type max_depth: int, default math.inf
    :return: the root of the tree
    :rtype: Node
    """ 
    f, sp = choose_feature_split(examples,features)
    root = Node(name = 'root', parent = None,feature = f, split = sp, num_examples = len(examples),decision = None)
    split_node(root, examples, features,max_depth)
    return root




def predict(cur_node: Node, example, max_depth=math.inf, \
    min_num_examples=0) -> int:
    """
    Given a tree with cur_node as its root, an example, and optionally a max depth,
    returns a prediction for the example based on the tree.

    If max_depth is provided and we haven't reached a leaf node at the max depth, 
    return the majority decision at this node.

    If min_num_examples is provided and the number of examples at the node is less than min_num_examples, 
    return the majority decision at this node.
    
    This function is recursive.

    Tie breaking rule:
    If there is a tie for majority voting, always return the label with the smallest value.

    :param cur_node: cur_node of a decision tree
    :type cur_node: Node
    :param example: one example
    :type example: List[Any]
    :param max_depth: the max depth
    :type max_depth: int, default math.inf
    :param min_num_examples: the minimum number of examples at a node
    :type min_num_examples: int, default 0
    :return: the decision for the given example
    :rtype: int
    """ 
    while cur_node:
        if cur_node.decision != None:
            return  cur_node.decision
        elif cur_node.depth == max_depth:
            return cur_node.majority
        elif cur_node.num_examples < min_num_examples:
            return cur_node.majority
        else:
            feature_index = dt_global.feature_names.index(cur_node.feature)
            value = example[feature_index]
            if value <= cur_node.split:
                cur_node = cur_node.children[0]
            else:
                cur_node = cur_node.children[1]


def get_prediction_accuracy(cur_node: Node, examples: List, max_depth=math.inf, \
    min_num_examples=0) -> float:
    """
    Given a tree with cur_node as the root, some examples, 
    and optionally the max depth or the min_num_examples, 
    returns the accuracy by predicting the examples using the tree.

    The tree may be pruned by max_depth or min_num_examples.

    :param cur_node: cur_node of the decision tree
    :type cur_node: Node
    :param examples: the set of examples. 
    :type examples: List[List[Any]]
    :param max_depth: the max depth
    :type max_depth: int, default math.inf
    :param min_num_examples: the minimum number of examples at a node
    :type min_num_examples: int, default 0
    :return: the prediction accuracy for the examples based on the cur_node
    :rtype: float
    """ 
    total = len(examples)
    hit = 0
    for ex in examples:
        prediction = predict(cur_node,ex,max_depth,min_num_examples)
        if prediction == ex[dt_global.label_index]:
            hit += 1
    return hit/total


def post_prune(cur_node: Node, min_num_examples: float):
    """
    Given a tree with cur_node as the root, and the minimum number of examples,
    post prunes the tree using the minimum number of examples criterion.

    This function is recursive.

    Let leaf parents denote all the nodes that only have leaf nodes as its descendants. 
    Go through all the leaf parents.
    If the number of examples at a leaf parent is smaller than the pre-defined value,
    convert the leaf parent into a leaf node.
    Repeat until the number of examples at every leaf parent is greater than
    or equal to the pre-defined value of the minimum number of examples.

    :param cur_node: the current node
    :type cur_node: Node
    :param min_num_examples: the minimum number of examples
    :type min_num_examples: float
    """
    if cur_node.num_examples < min_num_examples:
        if cur_node.decision == None:
            cur_node.feature = None
            cur_node.split = None
            cur_node.decision = cur_node.majority
            cur_node.children = []
    else:
        if cur_node.children != []:
            post_prune(cur_node.children[0],min_num_examples)
            post_prune(cur_node.children[1],min_num_examples)



