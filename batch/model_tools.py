import numpy as np
import matplotlib.pyplot as plt
from keras.models import model_from_yaml

def save_model(model, fname):
    '''
    Save model layers and weights
    '''
    model.save_weights(fname)
    yaml_string = model.to_yaml()
    fout = open(fname + ".layers", "w")
    fout.write(yaml_string)
    fout.close()

def load_model(fname):
    '''
    Load model layers and weights
    '''
    fin = open(fname + ".layers", "r")
    yaml_string  = fin.read()
    fin.close()
    model = model_from_yaml(yaml_string )
    model.load_weights(fname)
    return model

def spectral_envelope(x, window):
    '''
    tf signal spectral envelope
    '''
    import tensorflow as tf
    z = tf.map_fn(tf.transpose, tf.cast(x, dtype=tf.complex64))
    fft = tf.abs(tf.fft(z))
    log_fft = tf.log(fft)
    fft_log_fft = tf.fft(tf.cast(log_fft, dtype=tf.complex64))

    wfft_log_fft = tf.multiply(fft_log_fft, window)
    res = tf.cast(tf.exp(tf.ifft(wfft_log_fft)), dtype=tf.float32)
    half = int(res.get_shape().as_list()[-1] / 2)
    return tf.map_fn(tf.transpose, res)[:, :half, :]

def fft(x):
    '''
    tf fft 
    '''
    import tensorflow as tf
    z = tf.map_fn(tf.transpose, tf.cast(x, dtype=tf.complex64))
    z2 = tf.cast(tf.abs(tf.fft(z)), dtype=tf.float32)
    return tf.map_fn(tf.transpose, z2)


def arrhythmia_prediction(signal, models, plot=True, frame=3000,
                          t_step=500, thresh=0.7, artm_pos=0): 
    '''
    Divides signal into frames of length frame with time step equal to t_step. 
    Returns probability of arrythmia for each frame according to each model in models.
    Supports plotting of signal and frames with detected arrythmia.
    '''
    step = t_step - 1
    pred_thr = 0.7
    votes = []
    if plot:
        plt.plot(signal)
        last_p = 0

    for start in range(0, len(signal) - frame, step):
        segment = signal[start: start + frame]
        for model in models:
            pred = model.predict(np.array([segment[:, np.newaxis]]))[0]
            votes.append(pred)
            if plot and pred[artm_pos] > thresh:
                plt.axvspan(max(start, last_p), start + frame, alpha=0.3, color='red', lw=0)
                last_p = start + frame
    votes = np.array(votes)
    print('Probability of arrhythmia %.2f' % np.mean(votes[:, 0]))
    if plot:
        plt.show()
    return votes
