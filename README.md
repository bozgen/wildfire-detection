# Wildfire Detection

## Get started


### Prediction / Training
Use **main.py** file in the project root folder to run the project.
The model class uses **data** folder to train the model and **test_images** folder to test the images.



### Preprocessing

Currently there is no pipeline where you can preprocess the images and add them to the dataset automatically. So it is done manually.

You can see an example of preprocessing in the **main.py** file too.

Preprocessing functions use the images in the **predata** folder. According to the processing function, new images are saved with new names in the same folder.

Then you can move the images you want to the data folder manually to include them in the dataset.