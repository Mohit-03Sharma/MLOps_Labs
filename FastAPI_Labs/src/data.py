from sklearn.datasets import load_wine


def load_dataset():
    
    wine = load_wine()
    X = wine.data
    y = wine.target
    feature_names = list(wine.feature_names)
    target_names = list(wine.target_names)
    return X, y, feature_names, target_names