# trainer.py
class Trainer:
    def __init__(self, model, optimizer, loss_function, metrics):
    self.model = model
    self.model.compile(optimizer = optimizer, loss=loss_function, metrics=metrics)

  def train_model(self, x_train, y_train, x_val, y_val, epochs, batch_size):
    history = self.model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(x_val, y_val), verbose=2)
    return history