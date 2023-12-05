# main.py
from data_loader import DataLoader
from data_preprocessor import DataPreprocessor
from model_builder import ModelBuilder, XceptionBlock
from trainer import Trainer

def main():
  x_train, y_train, x_val, y_val, x_test, y_test = DataLoader.load_data()
  x_train, x_val, x_test = DataPreprocessor.preprocess_data(x_train, x_val, x_test)

  input_shape = x_train.shape[1:]
  num_classes = 10

  model = ModelBuilder.build_model(input_shape, num_classes)

  optimizer = 'adam'
  loss_function = 'sparse_categorical_crossentropy'
  metrics = ['accuracy']

  trainer = Trainer(model, optimizer, loss_function, metrics)

  epochs = 5
  batch_size = 32

  history = trainer.train_model(x_train, y_train, x_val, y_val, epochs, batch_size)

  test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=2)
  print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

if __name__ == "__main__":
    main()
