# Wildfire Detection

## Get started

in your virtual environment
```
python -m pip install -r requirements.txt
```
---

## Prediction / Training
Use **main.py** file in the project root folder to run the project.
The model class uses **data** folder to train the model and **test_images** folder to test the images.

You can add whatever image you want to test_images folder to predict it.

---

You can also use the functions below to test and display the result for one single image.
+  *predict_single* to test one single image and print the result

+  *display_single* to test and display the result of one single image.

---

## Preprocessing

Currently there is no pipeline where you can preprocess the images and add them to the dataset automatically. So it is done manually.

You can see an example of preprocessing in the **main.py** file too.

Preprocessing functions use the images in the **predata** folder. According to the processing function, new images are saved with new names in the same folder.

Then you can move the images you want to the data folder manually to include them in the dataset.

You can also import functions from preprocessing module like below and use them wherever you want

```
from wildfire.preprocessing import contrast, mass_flip, resize_in_place
```