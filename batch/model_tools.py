import numpy as np
import matplotlib.pyplot as plt
import time

import dataset as ds


def preprocess(data_source):
    '''
    Preprocess for fft_inception model
    '''
    return (data_source.p
            .load(src)
            .load_labels(DIR_ECG + 'REFERENCE.csv')
            .drop_noise()
            .augment_fs([('delta', {'loc': 250}),
                         ('delta', {'loc': 350}),
                         ('none', {})])
            .segment(3000, 3000, pad=True))


def show_loss(batch, model_name):
    '''
    Show loss and metric for train and validation parts
    '''
    model_comp = batch.get_model_by_name(model_name)
    model, hist, code, _ = model_comp

    metrics = ['train_loss',
               'train_metric',
               'val_loss',
               'val_metric']

    values = np.mean([hist[k] for k in metrics], axis=1)
    print('train_loss: {:.4f}   train_metric: {:.4f}   val_loss: {:.4f}   val_metric: {:.4f}'
          .format(*values))

    hist['train_loss'] = []
    hist['train_metric'] = []
    hist['val_loss'] = []
    hist['val_metric'] = []
    return metrics, values


def LearningRateSheduler(batch, model_name, epoch, lr_s):
    '''
    Schedule learning rate
    '''
    model_comp = batch.get_model_by_name(model_name)
    model, hist, code, _ = model_comp
    if epoch in lr_s[0]:
        new_lr = lr_s[1][lr_s[0].index(epoch)]
        opt = Adam(lr=new_lr)
        model.compile(optimizer=opt, loss="categorical_crossentropy")


def train_model(dataset, model_name, preprocess,
                nb_epoch, batch_size, callback_list=[],
                lr_schedule = None,
                val_split=None, prefetch=0):
    if val_split is not None:
        dataset.cv_split(val_split)
    else:
        dataset.cv_split(1)

    pp = preprocess(dataset.train).train_on_batch(model_name)
    ppt = preprocess(dataset.test).validate_on_batch(model_name)

    hist = []

    for epoch in range(nb_epoch):
        startTime = time.time()
        if lr_schedule is not None:
            LearningRateSheduler(dataset.next_batch(1), model_name, epoch, lr_schedule)
        b1 = pp.next_batch(batch_size=batch_size,
                           shuffle=True, n_epochs=1, prefetch=prefetch)
        b2 = ppt.next_batch(batch_size=batch_size,
                            shuffle=True, n_epochs=1, prefetch=prefetch)
        for callback in callback_list:
            callback(b2, model_name)
        print('Epoch {0}/{1}   finished in {2:.1f}s'.format(epoch + 1, nb_epoch, time.time() - startTime))
        metrics, values = show_loss(dataset.next_batch(1), model_name)
        hist.append(values)
    hist = np.array(hist).T
    return {k: v for (k, v) in zip(metrics, hist)}
