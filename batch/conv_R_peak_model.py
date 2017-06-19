'''
модель после 200 итераций (батч=2000) на 7 слое дает неплохой датчик Р-пиков
'''

input = Input(trainX[0].shape, name='user_input')

conv_1 = Convolution1D(4, 4, activation='relu')(input)
mp_1 = MaxPooling1D()(conv_1)
conv_2 = Convolution1D(8, 4, activation='relu')(mp_1)
mp_2 = MaxPooling1D()(conv_2)
conv_3 = Convolution1D(16, 4, activation='relu')(mp_2)
mp_3 = MaxPooling1D()(conv_3)
conv_4 = Convolution1D(3, 4, activation='relu')(mp_3)


conv2_1 = Convolution1D(8, 4, activation='relu')(conv_4)
mp2_1 = MaxPooling1D()(conv2_1)
conv2_2 = Convolution1D(16, 4, activation='relu')(mp2_1)
mp2_2 = MaxPooling1D()(conv2_2)
conv2_3 = Convolution1D(32, 4, activation='relu')(mp2_2)
mp2_3 = MaxPooling1D()(conv2_3)
conv2_4 = Convolution1D(3, 4, activation='relu')(mp2_3)


conv3_1 = Convolution1D(16, 4, activation='relu')(conv2_4)
mp3_1 = MaxPooling1D()(conv3_1)
conv3_2 = Convolution1D(32, 4, activation='relu')(mp3_1)
conv3_4 = Convolution1D(3, 4, activation='relu')(conv3_2)

pool = Flatten()(conv3_4))

fc_1 = Dense(16, kernel_initializer="uniform", activation='relu')(pool)

drop = Dropout(0.2)(fc_1)

fc_2 = Dense(2, kernel_initializer="uniform", activation='softmax')(drop)

model = Model(
        inputs=input,
        outputs=fc_2)
model.compile(loss="categorical_crossentropy", optimizer="adam")
model.summary()