from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Conv2D, ConvLSTM2D, MaxPooling2D, Activation, BatchNormalization, Dropout, Flatten, Dense, GlobalAveragePooling2D, TimeDistributed, GRU
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.regularizers import l2


def build_model(input_shape, num_labels, learning_rate):
    # Define the video input
    video_input = Input(shape=input_shape)

    # Define the CNN base model
    cnn = Sequential()
    cnn.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
    cnn.add(BatchNormalization())
    cnn.add(MaxPooling2D((2, 2)))

    cnn.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    cnn.add(BatchNormalization())
    cnn.add(MaxPooling2D((4, 4)))

    cnn.add(Flatten())
    cnn.add(Dense(128, activation='relu', kernel_regularizer=l2(0.01)))  # Increased regularization
    cnn.add(Dropout(0.6))  # Increased dropout rate

    # Apply the CNN base to each frame of the video input
    encoded_frames = TimeDistributed(cnn)(video_input)

    # GRU layer
    gru1 = GRU(256, kernel_regularizer=l2(0.01))(encoded_frames)  # Added regularization
    dropout1 = Dropout(0.6)(gru1)  # Increased dropout rate

    # Fully connected layers
    hidden_layer = Dense(256, activation="relu", kernel_regularizer=l2(0.01))(dropout1)  # Added regularization
    dropout2 = Dropout(0.6)(hidden_layer)  # Increased dropout rate

    # Output layer
    outputs = Dense(num_labels, activation="softmax")(dropout2)  # Adjust the number of units according to num_labels if necessary

    # Create the model
    model = Model(inputs=video_input, outputs=outputs)
    
    optimiser = SGD(learning_rate=learning_rate, momentum=0.9)
    model.compile(optimizer=optimiser, loss='categorical_crossentropy', metrics=['categorical_accuracy'])
    
    print(model.summary())

    return model
