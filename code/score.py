def f1(tp, tn, fp, fn):
    return {
        "Precision": tp/(tp+fp),
        "Recall": tp/(tp+fn),
        "F1-score": 2*tp/(2*tp+fp+fn),
        "Accuracy": (tp + tn)/(tp + tn + fp + fn)
    }