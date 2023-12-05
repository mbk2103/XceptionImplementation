# model_builder.py
from tensorflow.keras import layers, models

class ModelBuilder:
  @staticmethod
  def build_model(input_shape, num_classes):
    model = models.Sequential()

    # Entry Flow
    model.add(layers.Conv2D(32, (3, 3), strides=(2, 2), activation='relu', input_shape=input_shape, padding='same'))
    model.add(layers.Conv2D(64, (3, 3), strides=(2, 2), activation='relu', padding='same')) 
    model.add(layers.SeparableConv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(layers.MaxPooling2D((2, 2)))

    # Middle Flow
    for _ in range(8):
      model.add(XceptionBlock(128))

    # Exit Flow
    model.add(layers.SeparableConv2D(256, (3, 3), activation='relu', padding='same'))
    model.add(layers.GlobalAveragePooling2D())

    # Fully connected layer
    model.add(layers.Dense(num_classes, activation='softmax'))

    return model