
# decision tree classifier and decision tree regressor from scratch

import numpy as np


class DecisionTreeNode():

    def __init__(self, split_rule = None):

        self.split_rule = split_rule

        self.isLeaf = True
        self.leftChild = None
        self.rightChild = None
        self.info = None


class DecisionTreeClassifier():

    def __init__(self, gain_criterion, split_strategy, max_depth, min_samples_split, min_samples_leaf):
        
        self.gain_criterion = gain_criterion
        self.split_strategy = split_strategy
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf

        self.rootNode = DecisionTreeNode()
        self.dataToSplit = None
        self.treeManufacturingFinish = False

    def fit(self, X, y):
        
        # Make tree using build tree function// self.root = self.build_tree(data_to_split, depth, ...)

        pass

    def _build_tree(self):

        # find split for the data
        # create node
        # split data for left and right child node
        # node.left = build tree
        # node.right = build tree
        # return node
        
        pass

    def predict(self):
        pass

    
