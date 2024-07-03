from tensorflow.keras.applications import MobileNet
from tensorflow.keras.layers import Input, TimeDistributed, BatchNormalization, MaxPooling2D, Flatten, GRU, Dense, Dropout
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.regularizers import l2

def build_model(input_shape, num_labels, learning_rate):
    # Define the video input
    video_input = Input(shape=input_shape)

    # Load the MobileNet model with pre-trained weights and exclude the top layer
    mobilenet_transfer = MobileNet(weights='imagenet', include_top=False, input_shape=input_shape[1:])
    
    # Set the MobileNet layers as non-trainable
    for layer in mobilenet_transfer.layers:
        layer.trainable = False

    # Define the CNN base model as a Sequential model
    cnn = Sequential()
    cnn.add(mobilenet_transfer)
    cnn.add(BatchNormalization())
    cnn.add(MaxPooling2D((2, 2)))
    cnn.add(Flatten())
    cnn.add(Dense(128, activation='relu', kernel_regularizer=l2(0.01)))
    cnn.add(Dropout(0.5))

    # Apply the CNN base to each frame of the video input
    encoded_frames = TimeDistributed(cnn)(video_input)

    # GRU layer
    gru_out = GRU(128)(encoded_frames)
    dropout1 = Dropout(0.5)(gru_out)

    # Fully connected layers
    hidden_layer = Dense(128, activation="relu", kernel_regularizer=l2(0.01))(dropout1)
    dropout2 = Dropout(0.5)(hidden_layer)

    # Output layer
    outputs = Dense(num_labels, activation="softmax")(dropout2)

    # Create the model
    model = Model(inputs=video_input, outputs=outputs)
    
    optimiser = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimiser, loss='categorical_crossentropy', metrics=['categorical_accuracy'])
    
    print(model.summary())

    return model