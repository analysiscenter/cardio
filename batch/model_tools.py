import numpy as np
import matplotlib.pyplot as plt
from keras.models import model_from_yaml

def spectrum1D(x, kernel_list):
	'''
	Convolve x with kennels in kernel_list 
	'''
    layers = []
    input_shape = x.get_shape().as_list()
    x_2d = tf.expand_dims(x, -2)
    for kernel in kernel_list:
        conv = tf.nn.conv2d(x_2d, kernel, strides=[1, 1, 1, 1], padding='SAME')
        layers.append(conv) 
    output = tf.concat(layers, axis=-2)
    return output
	
def corrcoef(a, b, axis=1):
	'''
	Computes correlation coeffitient between corresponding 1-D slices
	of arrays a and b along given axis
	'''
    mda = tf.nn.moments(a, axes=[axis])
    mdb = tf.nn.moments(b, axes=[axis])
    res = tf.reduce_mean(tf.multiply(a - tf.expand_dims(mda[0], dim=axis), 
                                    b - tf.expand_dims(mdb[0], dim=axis)), axis=axis)
    res = tf.divide(res, tf.multiply(tf.sqrt(mda[1]), tf.sqrt(mdb[1])))
    return tf.expand_dims(res, dim=axis)
	
def corrmatrix(x):
	'''
	Computes correlation matrix along first axis.
	X is a 4D tensor of dims batch_size + width + height + channels
	Returns 4D tensor of dims batch_size + height*channels + height + channels
	'''
    dims = x.get_shape().as_list()
    coef = []
    for r1 in range(dims[2]):
        for r2 in range(dims[3]):
            coef.append(corrcoef(x, tf.expand_dims(tf.expand_dims(x[:, :, r1, r2], dim=2), dim=3)))
    return tf.concat(coef, axis=1)

class ScaledConv1D(Layer):
	'''
	Keras trainable layer computes spectrogram of 1D signal. 
	Input is [batch_size, signal_length, channels]
	Output is [batch_size, signal_length, number_of_scales, filters]
	kernel_size is a length of generation function (wavelet)
	scales is a list of scales at which signal is concolved with kernel
	filter is a number of kernels.
	'''
    def __init__(self, filters, kernel_size, scales, 
                 activation=None, use_bias=True, **kwargs):
        self.kernel_size = kernel_size
        self.output_dim = filters
        self.scales = scales
        self.use_bias = use_bias
        if activation is None:
            self.activation = 'relu'
        else:
            self.activation = activation
        super(ScaledConv1D, self).__init__(**kwargs)

    def build(self, input_shape):
        self.kernel = self.add_weight(shape=(self.kernel_size, input_shape[-1], self.output_dim),
                                      initializer='uniform', name='kernel',
                                      trainable=True)       
        if self.use_bias:
            self.bias = self.add_weight(shape=(self.output_dim, ),
                                        initializer='uniform',
                                        name='bias')
        else:
            self.bias = None
        super(ScaledConv1D, self).build(input_shape)

    def call(self, x):
        scaled_kernels_2d = []
        kernel_shape = self.kernel.get_shape().as_list()
        for scale in self.scales:
            f_conv = tf.cast(tf.image.resize_images(self.kernel, 
                                                    [scale, kernel_shape[1]]), dtype=tf.float32)
            f_conv = tf.nn.l2_normalize(f_conv, dim=-1)
            scaled_kernels_2d.append(tf.expand_dims(f_conv, 1))
        output = spectrum1D(x, scaled_kernels_2d)
        if self.use_bias:
            output = tf.nn.bias_add(output, self.bias)
        if self.activation == 'linear':
            return output
        elif self.activation == 'relu':
            return tf.nn.relu(output)
        elif self.activation == 'sigmoid':
            return tf.nn.sigmoid(output)
        elif self.activation == 'tanh':
            return tf.nn.tanh(output)
        else:
            raise NotImplementedError("Only linear, relu, sigmoid, tanh activations are available")

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], len(self.scales), self.output_dim)

class AxisMaxPooling(Layer):
	'''
	Max pooling along given axis. Default is last axis.
	'''
    def __init__(self, axis=-1, **kwargs):
        self.pool_ax = axis
        super(AxisMaxPooling, self).__init__(**kwargs)

    def call(self, inputs):
        return K.max(inputs, axis=self.pool_ax)
    
    def build(self, input_shape):
        super(AxisMaxPooling, self).build(input_shape)
    
    def compute_output_shape(self, input_shape):
        return tuple(np.delete(input_shape, self.pool_ax))

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

def spectrum(kernel, input, scales):
	'''
	tf convolution of input with kernel resized to given scales
	'''
    layers = []
    for scale in scales:
        f_conv = tf.cast(tf.image.resize_images(kernel, [scale, 1]), dtype=tf.float32)
        layers.append(tf.nn.conv1d(input, f_conv, stride=1, padding='SAME'))
    return tf.concat(layers, axis=-1)

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
