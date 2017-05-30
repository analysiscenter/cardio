'''
Model gives 97% f1 score for A vs N
'''
def fft_call(x):
    length = x.get_shape().as_list()[1]
    fft_1 = Lambda(fft)(x)
    crop_1 = Lambda(crop, arguments={'a': 2, 'b': int(length / 3)})(fft_1)    
    conv_1 = Convolution1D(4, 4, activation='relu')(crop_1)
    mp_1 = MaxPooling1D()(conv_1)
    conv_2 = Convolution1D(8, 4, activation='relu')(mp_1)
    mp_2 = MaxPooling1D()(conv_2)
    conv_3 = Convolution1D(16, 4, activation='relu')(mp_2)
    mp_3 = GlobalMaxPooling1D()(conv_3)
    
    conv_1a = Convolution1D(4, 4, activation='relu')(x)
    mp_1a = MaxPooling1D()(conv_1a)
    conv_2a = Convolution1D(8, 4, activation='relu')(mp_1a)
    mp_2a = MaxPooling1D()(conv_2a)
    conv_3a = Convolution1D(16, 4, activation='relu')(mp_2a)
    mp_3a = GlobalMaxPooling1D()(conv_3a)
    
    output = merge([mp_3, mp_3a], mode='concat', concat_axis=-1)    
    return output

input = Input(trainX[0].shape, name='user_input')

conv_1 = Convolution1D(4, 4, activation='relu')(input)
mp_1 = MaxPooling1D()(conv_1)
conv_2 = Convolution1D(8, 4, activation='relu')(mp_1)
mp_2 = MaxPooling1D()(conv_2)
conv_3 = Convolution1D(16, 4, activation='relu')(mp_2)
mp_3 = MaxPooling1D()(conv_3)
conv_4 = Convolution1D(32, 4, activation='relu')(mp_3)

fft_1 = fft_call(conv_4)

fc_1 = Dense(8, init='uniform', activation='relu')(fft_1)
drop = Dropout(0.3)(fc_1)

fc_2 = Dense(2, init='uniform', activation='softmax')(drop)

model = Model(
        input=input,
        output=fc_2)
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=['fmeasure'])
model.summary()