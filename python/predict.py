import numpy as np

def predict_proba(features, weights):
    e = np.exp(features.dot(weights))
    probs = e / (1 + e) > 0.5
    return probs

def predict_classes(features, weights):
    probs = predict_proba(features, weights)
    return probs > 0.5

def evaluate(features, weights, target):
    preds = predict_classes(features, weights)
    print('Accuracy: ', np.mean(target == preds), '\n')

