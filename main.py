from wildfire.wildfire import Model
from wildfire.preprocessing import contrast, mass_flip
import os

# PREPROCESSING  ( read README.md to learn how it works.)
# mass_flip(1)


# PREDICTION
model = Model()
# model.train(epochs=10)
model.load_weights()
# model.evaluate()
model.test_and_show_first_ten()
model.display_single(os.path.join('test_images', 'fire_test_1.jpg'))
