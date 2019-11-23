"""
Get examples for XGBoost
"""

import numpy as np
import xgboost as xgb


PARAMETERS = {'max_depth': 2,
              'eta': 0.5,
              'lambda': 1,
              'gamma': 0,
              'min_child_weight': 0}

def example_1(objective):
    """
    Simple data
    """

    parameters = PARAMETERS.copy()
    parameters['objective'] = objective

    if objective == 'reg:squarederror':
        X_train = np.array([[0],
                            [1],
                            [2]])
        y_train = np.array([0, 1, 2])
        X_test = np.array([[-1],
                           [0],
                           [1],
                           [2],
                           [3]])
        y_test = np.array([-1, 0, 1, 2, 3])
    elif objective == 'binary:logistic':
        X_train = np.array([[0],
                            [1]])
        y_train = np.array([0, 1])
        X_test = np.array([[0],
                           [1],
                           [np.nan]])
        y_test = np.array([0, 1, 0])
    elif objective == 'multi:softprob':
        X_train = np.array([[0],
                            [1],
                            [2]])
        y_train = np.array([0, 1, 2])
        X_test = np.array([[0],
                           [1],
                           [2],
                           [np.nan]])
        y_test = np.array([0, 1, 2, 0])
        parameters['num_class'] = 3
    else:
        raise ValueError('Not a recognized objective')

    data_train = xgb.DMatrix(X_train, y_train)
    data_test = xgb.DMatrix(X_test, y_test)

    bst = xgb.train(parameters,
                    data_train,
                    3)

    dump = bst.get_dump(with_stats=True)
    for tree in dump:
        print(tree)

    print(bst.predict(data_test))


if __name__ == '__main__':
    example_1('reg:squarederror')
    example_1('binary:logistic')
    example_1('multi:softprob')
