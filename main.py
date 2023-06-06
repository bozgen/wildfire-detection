from wildfire.wildfire import Model
from wildfire.preprocessing import contrast, mass_flip


# PREPROCESSING
# mass_flip(1)


# PREDICTION
model = Model()
# model.train(epochs=10)
model.load_weights()
model.evaluate()
model.test_and_show_first_ten()
