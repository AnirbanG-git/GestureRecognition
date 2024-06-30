from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv3D, Activation, BatchNormalization, MaxPooling3D, Dropout, Flatten, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import he_normal
from tensorflow.keras.regularizers import L2

def build_model(input_shape, num_labels, learning_rate):
    model = Sequential()
    model.add(Conv3D(16, (3, 3, 3), padding='same',
             kernel_initializer=he_normal(), input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling3D(pool_size=(2, 2, 2)))
    #model.add(Dropout(0.25))

    model.add(Conv3D(32, (3, 3, 3), padding='same', kernel_initializer=he_normal()))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling3D(pool_size=(2, 2, 2)))
    #model.add(Dropout(0.25))

    model.add(Conv3D(64, (3, 3, 3), padding='same', kernel_initializer=he_normal()))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling3D(pool_size=(2, 2, 2)))
    #model.add(Dropout(0.25))
    
    model.add(Conv3D(128, (3, 3, 3), padding='same', kernel_initializer=he_normal()))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling3D(pool_size=(2, 2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    
    model.add(Dense(128, activation='relu', kernel_initializer=he_normal(), kernel_regularizer=L2(0.1)))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))

    model.add(Dense(128, activation='relu', kernel_initializer=he_normal(), kernel_regularizer=L2(0.1)))
    model.add(BatchNormalization())
    model.add(Dropout(0.25))
    
    model.add(Dense(num_labels, activation='softmax'))

    optimiser = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimiser, loss='categorical_crossentropy', metrics=['categorical_accuracy'])
    print(model.summary())  

    return model
