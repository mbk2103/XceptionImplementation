# xception_block.py
from tensorflow.keras import layers

class XceptionBlock(layers.Layer):
  def __init__(self, filters):
    super(XceptionBlock, self).__init__()

    self.sep_conv1 = layers.SeparableConv2D(filters, (3, 3), activation='relu', padding='same')
    self.sep_conv2 = layers.SeparableConv2D(filters, (3, 3), activation='relu', padding='same')
    self.sep_conv3 = layers.SeparableConv2D(filters, (3, 3), activation='relu', padding='same')
    self.add_residual = layers.Add()
  
  def call(self, inputs):
    x = self.sep_conv1(inputs)
    x = self.sep_conv2(x)
    x = self.sep_conv3(x)
    return self.add_residual([inputs, x])