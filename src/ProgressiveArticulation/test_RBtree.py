import numpy as np
import pickle as pickle
from src.ProgressiveArticulation.rule_based_tree import RuleBasedTree

def test_BK1_RBtree():
    load_path = '/home/kemal/Programming/Python/Articulation/data/progressive/RBTrees/'
    file_name = load_path + 'BK1' + '_rbtree.pickle'
    tree = None
    with open(file_name, 'rb') as f:
        tree = pickle.load(f)
    print(tree.classify(np.array([0, 0])))
    print(tree.classify(np.array([19.9, 39.9])))
    print(tree.classify(np.array([21.8, 40.1])))
    print(tree.classify(np.array([40.09, 0])))
    print(tree.classify(np.array([67, 20])))
    print(tree.classify(np.array([900, 900])))


if __name__ == '__main__':
    test_BK1_RBtree()


