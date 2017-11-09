"""Contains metric functions."""

import numpy as np
import pandas as pd
from sklearn import metrics


__all__ = ["f1_score", "auc", "classification_report", "confusion_matrix", "calculate_metrics"]


def get_class_prob(predictions_dict):
    """Get true and predicted targets from predictions_dict.

    Parameters
    ----------
    predictions_dict : dict
        Dict of model predictions. Must contain "target_true" and "target_pred" keys with corresponding
        dict values of the form class_label : probability.

    Returns
    -------
    true_dict : dict
        True target - value for "target_true" key from the predictions_dict.
    pred_dict : dict
        Predicted target - value for "target_pred" key from the predictions_dict.
    """
    true_dict = predictions_dict.get("target_true")
    pred_dict = predictions_dict.get("target_pred")
    if true_dict is None or pred_dict is None:
        raise ValueError("Each element of predictions list must be a dict with target_true and target_pred keys")
    return true_dict, pred_dict


def get_labels(predictions_list):
    """Get true and predicted class labels from predictions_list.

    Parameters
    ----------
    predictions_list : list
        List, containing dicts of model predictions. Each dict must contain "target_true" and "target_pred" keys
        with corresponding dict values of the form class_label : probability.

    Returns
    -------
    true_labels : 1-D ndarray
        True class labels.
    pred_labels : 1-D ndarray
        Predicted class labels.
    """
    true_labels = []
    pred_labels = []
    for predictions_dict in predictions_list:
        true_dict, pred_dict = get_class_prob(predictions_dict)
        true_labels.append(max(true_dict, key=true_dict.get))
        pred_labels.append(max(pred_dict, key=pred_dict.get))
    return np.array(true_labels), np.array(pred_labels)


def get_probs(predictions_list):
    """Get true and predicted class probabilities from predictions_list.

    Parameters
    ----------
    predictions_list : list
        List, containing dicts of model predictions. Each dict must contain "target_true" and "target_pred" keys
        with corresponding dict values of the form class_label : probability.

    Returns
    -------
    true_probs : 2-D ndarray
        True class probabilities.
    pred_probs : 2-D ndarray
        Predicted class probabilities.
    """
    true_probs = []
    pred_probs = []
    for predictions_dict in predictions_list:
        true_dict, pred_dict = get_class_prob(predictions_dict)
        true_probs.append([true_dict[key] for key in sorted(true_dict.keys())])
        pred_probs.append([pred_dict[key] for key in sorted(pred_dict.keys())])
    return np.array(true_probs), np.array(pred_probs)


def f1_score(predictions_list, average="macro", **kwargs):
    """Compute the F1 score.

    Parameters
    ----------
    predictions_list : list
        List, containing dicts of model predictions. Each dict must contain "target_true" and "target_pred" keys
        with corresponding dict values of the form class_label : probability.
    average : str or None
        sklearn.metrics.f1_score average parameter, which determines the type of averaging performed on the data.
    **kwargs : misc
        Other sklearn.metrics.f1_score keyword arguments.

    Returns
    -------
    f1_score : float or array of floats
        F1 score for each class or weighted average of the F1 scores.
    """
    true_labels, pred_labels = get_labels(predictions_list)
    unique_labels = sorted(set(true_labels) | set(pred_labels))
    return metrics.f1_score(true_labels, pred_labels, labels=unique_labels, average=average, **kwargs)


def auc(predictions_list, average="macro", **kwargs):
    """Compute area under the ROC curve.

    Parameters
    ----------
    predictions_list : list
        List, containing dicts of model predictions. Each dict must contain "target_true" and "target_pred" keys
        with corresponding dict values of the form class_label : probability.
    average : str or None
        sklearn.metrics.roc_auc_score average parameter, which determines the type of averaging performed on the data.
    **kwargs : misc
        Other sklearn.metrics.roc_auc_score keyword arguments.

    Returns
    -------
    auc : float or array of floats
        AUC for each class or weighted average of the AUCs.
    """
    return metrics.roc_auc_score(*get_probs(predictions_list), average=average, **kwargs)


def classification_report(predictions_list, **kwargs):
    """Build a text report showing the main classification metrics.

    Parameters
    ----------
    predictions_list : list
        List, containing dicts of model predictions. Each dict must contain "target_true" and "target_pred" keys
        with corresponding dict values of the form class_label : probability.
    **kwargs : misc
        Other sklearn.metrics.classification_report keyword arguments.

    Returns
    -------
    report : string
        Text summary of the precision, recall and F1 score for each class.
    """
    return metrics.classification_report(*get_labels(predictions_list), **kwargs)


def confusion_matrix(predictions_list, margins=True, **kwargs):
    """Compute confusion matrix from correct and estimated targets.

    Parameters
    ----------
    predictions_list : list
        List, containing dicts of model predictions. Each dict must contain "target_true" and "target_pred" keys
        with corresponding dict values of the form class_label : probability.
    margins : bool
        Specifies whether to add row and column margins with subtotals.
    **kwargs : misc
        Other pandas.crosstab keyword arguments.

    Returns
    -------
    crosstab : DataFrame
        Classifier confusion matrix.
    """
    true, pred = get_labels(predictions_list)
    true = pd.Series(true, name="True")
    pred = pd.Series(pred, name="Pred")
    return pd.crosstab(pred, true, margins=margins, **kwargs)


METRICS_DICT = {
    "f1_score": f1_score,
    "auc": auc,
    "classification_report": classification_report,
    "confusion_matrix": confusion_matrix,
}


def calculate_metrics(metrics_list, predictions_list):
    """Calculate metrics from metrics_list.

    Parameters
    ----------
    metrics_list : list
        List of metrics. Each element can be either str with metric name or callable.
    predictions_list : list
        List, containing dicts of model predictions. Each dict must contain "target_true" and "target_pred" keys
        with corresponding dict values of the form class_label : probability.

    Returns
    -------
    metrics_res : list
        List containing results for every metric in metrics_list.
    """
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
        metrics_res.append(metric_fn(predictions_list))
    return metrics_res
