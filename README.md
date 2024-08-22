# Quantification of _Bacillus coagulans_ in Phase-Contrast Microscopy Images using a CNN-based Algorithm
## 1 Abstract
Due to their resistance against environmental stress, like heat and high pressure, products made from _Bacillus_ endospores are becoming increasingly popular. Therefore, a method is needed to quantify them quickly and reliably during a running bioprocess. The determination of colony forming units, which is used as the standard method, is elaborate, time-consuming and consumes a lot of material. Because of their strong self-reflection in phase-contrast, endospores can be differentiated particularly well from vegetative cells in phase-contrast microscopy images. To analyze these images, a CNN-based algorithm was developed that can classify between the cell types vegetative cells, forespores and endospores and achieves an average precision of 96.4 %. In order to find the regions of interest, a pre-processing step was integrated which is able to segment 98.82 % of the relevant image objects.


## 2 Installation
### 2.1 Installation using VSC
1. Open VSC and if necessary open a new window...
2. Start > Clone Git Repository
3. Enter Link in the automatically opened search bar
```
https://github.com/TCI-LUH/CNN-Bacillus-Analyzer.git
```
4. Choose destination for the repository
5. Wait till repository is cloned

### 2.2 Create Environment in VSC
1. Press Strg+Shift+P in VSC to open the Command Palette
2. Search for "Python: Create Environment" and select it
3. Choose Conda and select Python 3.10
4. Packages will automatically be added through environment.yml file

### 2.3 Starting the scripts
The packages will automatically be installed from the environment.yml file. To start the scripts by open the code and pressing start or typing in  the VSC terminal:
```python
python .\'name_of_script'.py
```

## 3 How to use
### 3.1 Adding new images to the database
1. Add images to folder _data_ in a subfolder
2. Open _dataProcessing.py_
3. Set add_to_database = True
```python
### Settings ###
add_to_database = True
process_labeled_images = False
random_image_selection = False
create_dataset_split = False
```
4. Enter destination of new images in folder_path (line 22)
5. Run script. Code automatically searches images for region of interest (ROI) and adds their location to database
### 3.2 Label newly added ROIs
1. Run GraphicalUserInterface by opening the script and click RUN or by typing in the terminal
```python
python .\GraphicalUserInterface.py
```
2. Open folder or files and press _Start Labeling_ or by pressing Space
3. Enter class number by pressing button or pressing the number

### 3.3 Create dataset
1. Open _dataProcessing.py_
2. Change settings and set *process_labeled_images* to True
```python
### Settings ###
add_to_database = False
process_labeled_images = True
random_image_selection = False
create_dataset_split = False
```
3. Add folder path for all desired images from data
4. Change dataset name in *file_name* (line 71)
4. Add foder path in *ImagePreprocess.process_labeled_images_folder()*
5. Run the scipt. 

### 3.4 Normalize dataset
1. Open _dataProcessing.py_
2. Change settings and set *random_image_selection* to True
```python
### Settings ###
add_to_database = False
process_labeled_images = False
random_image_selection = True
create_dataset_split = False
```
3. Set *num_images_to_copy* to a specific number
```python
# Specify the number of random images you want to copy from each subfolder
    num_images_to_copy = 1171
```
4. Enter destination of testregion folder in *source_folder* (line 110)
5. Run the script. If the entered number is too high, the programm will not sample the images.
### 3.5 Create validation and test dataset
1. Open *dataProcessing.py*
2. Change settings and set *create_dataset_split* to True
```python
### Settings ###
add_to_database = False
process_labeled_images = False
random_image_selection = False
create_dataset_split = True
```

### 3.6 Train Network
1. Open *neuralNetwork_Tests.py*
2. Choose parameter.
3. Run script.


## 4 Functions/Scripts
### 4.1 dataProcessing.py
**add to database**: Edit new images and add the ROIs to the database. The ROIs can now be labeled in the GraphicalUserInterface. <br />
**process_labeled_images**: Creation of the data set without validation and test split. Data is loaded from the database. Warning: Excess subfolders must be removed for training. <br />
**random_images_selection**: To normalize the data set so that all classes contain the same number of images <br />
**create_dataset_split**: Creates a training, validation and test data set from the complete data set. Warning: Folder must not have more subfolders than the required number of classes 
### 4.2 GraphicalUserInterface.py 
GUI for labeling ROIs from database
### 4.3 neuralNetwork_Test.py
Script for training the networks. The structure of the script makes it possible to test several parameters without starting a new run.


## 5 Developer
- Main Developer: Paul Heskamp, M.Sc.
- Co-Developer: Laura Niemeyer, M.Sc.


***
