import numpy as np
import sklearn


def get_class_prob(predictions_dict, metric_name="Metric"):
    class_prob = predictions_dict.get("class_prob")
    if class_prob is None:
        raise ValueError("{} is only applicable to classification tasks".format(metric_name))
    return class_prob


def get_labels(predictions_list, metric_name="Metric"):
    labels_list = []
    for predictions_dict in predictions_list:
        class_prob = get_class_prob(predictions_dict, metric_name)
        labels_list.append(max(class_prob, key=class_prob.get))
    return np.array(labels_list)


def get_probs(predictions_list, metric_name="Metric"):
    probs_list = []
    for predictions_dict in predictions_list:
        class_prob = get_class_prob(predictions_dict, metric_name)
        probs_list.append([class_prob[key] for key in sorted(class_prob.keys())])
    return np.array(probs_list)


def f1_score(y_true, y_pred, average="macro", **kwargs):
    y_true = get_labels(y_true, metric_name="F1 score")
    y_pred = get_labels(y_pred, metric_name="F1 score")
    labels = sorted(set(y_true) | set(y_pred))
    return sklearn.metrics.f1_score(y_true, y_pred, labels=labels, average=average, **kwargs)


def auc(y_true, y_pred, average="macro", **kwargs):
    y_true = get_probs(y_true, metric_name="AUC")
    y_pred = get_probs(y_pred, metric_name="AUC")
    return sklearn.metrics.roc_auc_score(y_true, y_pred, average=average, **kwargs)


def classification_report(y_true, y_pred, **kwargs):
    y_true = get_labels(y_true, metric_name="Classification report")
    y_pred = get_labels(y_pred, metric_name="Classification report")
    return sklearn.metrics.classification_report(y_true, y_pred, **kwargs)


METRICS_DICT = {
    "f1_score": f1_score,
    "auc": auc,
    "classification_report": classification_report,
}


def calculate_metrics(metrics_list, y_true, y_pred):
    metrics_res = []
    for metric in metrics_list:
        if isinstance(metric, str):
            metric_fn = METRICS_DICT.get(metric)
            if metric_fn is None:
                raise KeyError("Unknown metric name {}".format(metric))
        elif callable(metric):
            metric_fn = metric
        else:
            raise ValueError("Unknown metric type")
        metrics_res.append(metric_fn(y_true, y_pred))
    return metrics_res
