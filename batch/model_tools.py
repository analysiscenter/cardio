import time
import numpy as np
import matplotlib.pyplot as plt


def preprocess(dataset, scr, dir_ecg):
    '''
    Preprocess for fft_inception model
    '''
    return (dataset.p
            .load(src)
            .load_labels(dir_ecg + 'REFERENCE.csv')
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
    hist = model_comp[1]

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


def learning_rate_sheduler(batch, model_name, epoch, lr_s):
    '''
    Schedule learning rate
    '''
    model_comp = batch.get_model_by_name(model_name)
    model = model_comp[0]
    if epoch in lr_s[0]:
        new_lr = lr_s[1][lr_s[0].index(epoch)]
        opt = Adam(lr=new_lr)
        model.compile(optimizer=opt, loss="categorical_crossentropy")


def train_model(dataset, model_name, preprocess,#pylint: disable=too-many-arguments
                nb_epoch, batch_size, callback_list=None,
                lr_schedule=None,
                val_split=None, prefetch=0):
    '''
    Train model
    '''
    if val_split is not None:
        dataset.cv_split(val_split)
    else:
        dataset.cv_split(1)

    ppt = preprocess(dataset.train).train_on_batch(model_name)
    ppv = preprocess(dataset.test).validate_on_batch(model_name)

    hist = []

    for epoch in range(nb_epoch):
        start_time = time.time()
        if lr_schedule is not None:
            LearningRateSheduler(dataset.next_batch(1), model_name, epoch, lr_schedule)
        ppt.next_batch(batch_size=batch_size,
                      shuffle=True, n_epochs=1, prefetch=prefetch)
        ppv.next_batch(batch_size=batch_size,
                       shuffle=True, n_epochs=1, prefetch=prefetch)
        if callback_list is not None:
            for callback in callback_list:
                callback(b2, model_name)
        print('Epoch {0}/{1}   finished in {2:.1f}s'.format(epoch + 1, nb_epoch, time.time() - start_time))
        metrics, values = show_loss(dataset.next_batch(1), model_name)
        hist.append(values)
    hist = np.array(hist).T
    return {k: v for (k, v) in zip(metrics, hist)}
