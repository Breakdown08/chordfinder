from keras import layers, Model, Input
from keras.regularizers import l2
from keras.layers import LayerNormalization


def model_Conv2D(n_classes, n_mels, frames):
    input_shape = (n_mels, frames, 1)
    inputs = Input(shape=input_shape)
    x = LayerNormalization(axis=1, name='batch_norm')(inputs)
    
    x = layers.Conv2D(8, kernel_size=(7, 7), activation='tanh', padding='same', name='conv2d_tanh')(x)
    x = layers.MaxPooling2D(pool_size=(2, 2), padding='same', name='max_pool_2d_1')(x)
    
    x = layers.Conv2D(16, kernel_size=(5, 5), activation='relu', padding='same', name='conv2d_relu_1')(x)
    x = layers.MaxPooling2D(pool_size=(2, 2), padding='same', name='max_pool_2d_2')(x)
    
    x = layers.Conv2D(16, kernel_size=(3, 3), activation='relu', padding='same', name='conv2d_relu_2')(x)
    x = layers.MaxPooling2D(pool_size=(2, 2), padding='same', name='max_pool_2d_3')(x)
    
    x = layers.Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same', name='conv2d_relu_3')(x)
    x = layers.MaxPooling2D(pool_size=(2, 2), padding='same', name='max_pool_2d_4')(x)
    
    x = layers.Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same', name='conv2d_relu_4')(x)
    x = layers.Flatten(name='flatten')(x)
    x = layers.Dropout(rate=0.2, name='dropout')(x)
    x = layers.Dense(64, activation='relu', activity_regularizer=l2(0.001), name='dense')(x)
    outputs = layers.Dense(n_classes, activation='softmax', name='softmax')(x)
    
    model = Model(inputs=inputs, outputs=outputs, name='2d_convolution')
    
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model
