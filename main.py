from wildfire.wildfire import Model

model = Model()
# model.train(epochs=10)
model.load_weights()
model.evaluate()
model.test_and_show_results()
