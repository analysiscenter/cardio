'''
Model creates 2d fft spectrogram and analyses it with inception layers. 
Gives above 80% f1 score for A, N, O classes
'''
def Inception2D(x, dim, nb_filters, s1, s2):
    conv_1 = Convolution2D(dim, 1, 1, activation='relu', border_mode='same')(x)
    
    conv_2 = Convolution2D(dim, 1, 1, activation='relu', border_mode='same')(x)
    conv_2a = Convolution2D(nb_filters, s1, s1, activation='relu', border_mode='same')(conv_2)
    
    conv_3 = Convolution2D(dim, 1, 1, activation='relu', border_mode='same')(x)
    conv_3a = Convolution2D(nb_filters, s2, s2, activation='relu', border_mode='same')(conv_3)
    
    pool = MaxPooling2D(strides = (1, 1), border_mode='same')(x)
    conv_4 = Convolution2D(nb_filters, 1, 1, activation='relu', border_mode='same')(pool)
    
    concat = merge([conv_1, conv_2a, conv_3a, conv_4], mode='concat', concat_axis=-1)    
    return concat

input = Input(trainX[0].shape, name='user_input')

conv_1 = Convolution1D(4, 4, activation='relu')(input)
mp_1 = MaxPooling1D()(conv_1)
conv_2 = Convolution1D(8, 4, activation='relu')(mp_1)
mp_2 = MaxPooling1D()(conv_2)
conv_3 = Convolution1D(16, 4, activation='relu')(mp_2)
mp_3 = MaxPooling1D()(conv_3)
conv_4 = Convolution1D(32, 4, activation='relu')(mp_3)

fft_1 = Lambda(fft)(conv_4)
length = fft_1.get_shape().as_list()[1]
crop_1 = Lambda(crop, arguments={'a': 2, 'b': int(length / 3)})(fft_1)  

shape_1d = crop_1.get_shape().as_list()[1:]
shape_1d.append(1)
to2d = Reshape(shape_1d)(crop_1)

incept_1 = Inception2D(to2d, 4, 4, 3, 5)
mp2d_1 = MaxPooling2D(pool_size=(4, 2))(incept_1)

incept_2 = Inception2D(mp2d_1, 4, 8, 3, 5)
mp2d_2 = MaxPooling2D(pool_size=(4, 2))(incept_2)

incept_3 = Inception2D(mp2d_2, 4, 12, 3, 3)

pool = GlobalMaxPooling2D()(incept_3)

fc_1 = Dense(8, init='uniform', activation='relu')(pool)
drop = Dropout(0.2)(fc_1)

fc_2 = Dense(2, init='uniform', activation='softmax')(drop)

model = Model(
        input=input,
        output=fc_2)
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=['fmeasure'])
model.summary()