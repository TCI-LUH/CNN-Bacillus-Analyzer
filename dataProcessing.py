import os
from sqliteDatabase import LabeledRegionsOfInterestDB as LabeledROIDB
from imagePreprocess import ImagePreprocess
import shutil
import random

### Settings ###
add_to_database = True
process_labeled_images = False
random_image_selection = False
create_dataset_split = False

#process labeled images settings for ResNet Training
color_images = False

# Directory
file_directory = os.path.dirname(os.path.abspath(__file__))

####ALREADY IN DATABASE####
if add_to_database:
    # folder_path = os.path.join(file_directory, "data", "Bilder B.coagulans", "DOE1Test")
    # folder_path = os.path.join(file_directory, "data", "Bilder B.coagulans", "Sample1")
    # folder_path = os.path.join(
    #     file_directory, "data", "Bilder B.coagulans", "VegZellenBCoa", "Sample 1"
    # )
    # folder_path = os.path.join(
    #     file_directory, "data", "Bilder B.coagulans", "VegZellenBCoa", "Sample 4"
    # )
    folder_path = os.path.join("data", "Bilder BLicheniformis", "vegZellen", "Sample 3")
    ls_regions_of_interest = ImagePreprocess.preprocess_images_folder(
        folder_path, ".bmp"
    )

    # ####INSERT LIST IN DATABASE####
    LabeledROIDB("LabelDatabase.db").insert_roi_list(ls_regions_of_interest)
    LabeledROIDB("LabelDatabase.db").close()

###PROCESS LABELED IMAGES####
# Modify save folder after creating (remove or combine subfolders)
if process_labeled_images:
    folder_path1 = os.path.join(
        file_directory, "data", "Bilder B.coagulans", "DOE1Test"
    )
    folder_path2 = os.path.join(file_directory, "data", "Bilder B.coagulans", "Sample1")
    folder_path3 = os.path.join(
        file_directory, "data", "Bilder B.coagulans", "VegZellenBCoa", "Sample 1"
    )
    folder_path4 = os.path.join(
        file_directory, "data", "Bilder B.coagulans", "VegZellenBCoa", "Sample 4"
    )
    # folder_path5 = (
    #   file_directory, "data", "folder", "subfolder", "subsubfolder"
    # )

    arr_image_size = [
        # 16,
        # 20,
        # 25,
        # 30,
        32,
        # 35,
        # 40,
        # 45,
        # 50,
        # 50,
        # 55,
        # 64,
    ]
    for image_size in arr_image_size:
        # name for dataset ### Change here ######
        file_name = str(image_size) + "_5C_all_color"

        save_directory = os.path.join(
            file_directory ,
            "testregions" ,
            file_name ,
        )
        if not os.path.exists(save_directory): ### Add folder_path
            ImagePreprocess.process_labeled_images_folder(
                folder_path1, save_directory, image_size, color_images
            )
            ImagePreprocess.process_labeled_images_folder(
                folder_path2, save_directory, image_size, color_images
            )
            ImagePreprocess.process_labeled_images_folder(
                folder_path3, save_directory, image_size, color_images
            )
            ImagePreprocess.process_labeled_images_folder(
                folder_path4, save_directory, image_size, color_images
            )
            #
            # ImagePreprocess.process_labeled_images_folder(
            #   folder_path5, save_directory, image_size, color_images
            # )
        else:
            print(f"Folder with name: {save_directory} already exists")

###RANDOM IMAGE SELECTION####
if random_image_selection:
    # Specify the seed for random number generation
    random_seed = 123

    # Specify the number of random images you want to copy from each subfolder
    num_images_to_copy = 1171

    # Set the random seed
    random.seed(random_seed)

    # Specify the source folder containing the subfolders
    source_folder = os.path.join(
        file_directory, "testregions", "32_newDataset_DOE1_Sample1_VegZellenBCoa1+2_3classes"
    )

    folder_name = "reduced_dataset" + str(num_images_to_copy)
    # Specify the destination folder where you want to copy the images
    destination_folder = os.path.join(file_directory, "testregions", folder_name)

    # Create the destination folder if it doesn't exist
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)

    # Iterate over subfolders 0, 1, 2, 3, 4
    for subfolder_name in ["0", "1", "2"]:
        subfolder_source = os.path.join(source_folder, subfolder_name)
        subfolder_destination = os.path.join(destination_folder, subfolder_name)

        # Create the subfolder in the destination folder
        if not os.path.exists(subfolder_destination):
            os.makedirs(subfolder_destination)

        # Get a list of all .png files in the subfolder
        png_files = [f for f in os.listdir(subfolder_source) if f.endswith(".png")]

        # Check if the number of images to copy is greater than the available images in this subfolder
        if num_images_to_copy > len(png_files):
            print(f"Warning: Not enough .png images in subfolder {subfolder_name}.")
        else:
            # Choose random .png files to copy
            random_images = random.sample(png_files, num_images_to_copy)

            # Copy the selected random images to the corresponding subfolder in the destination folder
            for image in random_images:
                source_path = os.path.join(subfolder_source, image)
                destination_path = os.path.join(subfolder_destination, image)
                shutil.copyfile(source_path, destination_path)

    print(
        f"{num_images_to_copy} random .png images from each subfolder copied to the destination folder."
    )

if create_dataset_split:
    # Specify the seed for random number generation
    random_seed = 123

    # Specify the split for each set
    trainings_val_split = 0.1
    trainings_test_split = 0.1

    # Set the random seed
    random.seed(random_seed)

    # Specify the source folder containing the subfolders
    folder_name = "32_5C_all_color"
    source_folder = os.path.join(
        file_directory, "testregions", folder_name
    )

    val_folder_name = "Val_" + folder_name
    test_folder_name = "Test_" + folder_name
    # Specify the destination folder where you want to copy the images
    destination_val_folder = os.path.join(file_directory, "testregions", val_folder_name)
    destination_test_folder = os.path.join(file_directory, "testregions", test_folder_name)

    # Create the destination folder if it doesn't exist
    if not os.path.exists(destination_val_folder):
        os.makedirs(destination_val_folder)

    if not os.path.exists(destination_test_folder):
        os.makedirs(destination_test_folder)

    # Iterate over subfolders 0, 1, 2, 3, 4
    for subfolder_name in ["0", "1", "2", "3", "4"]:
        subfolder_source = os.path.join(source_folder, subfolder_name)
        subfolder_val_destination = os.path.join(destination_val_folder, subfolder_name)
        subfolder_test_destination = os.path.join(destination_test_folder, subfolder_name)

        # Create the subfolder in the destination folder
        if not os.path.exists(subfolder_val_destination):
            os.makedirs(subfolder_val_destination)

        if not os.path.exists(subfolder_test_destination):
            os.makedirs(subfolder_test_destination)

        # Get a list of all .png files in the subfolder
        png_files = [f for f in os.listdir(subfolder_source) if f.endswith(".png")]
        num_all_img = len(png_files)

        #######VALIDATION#############
        val_num_images_to_copy = round(num_all_img * trainings_val_split)
        # Check if the number of images to copy is greater than the available images in this subfolder
        if val_num_images_to_copy > len(png_files):
            print(f"Warning: Not enough .png images in subfolder {subfolder_name} for VALIDATION DATASET.")
        else:
            # Choose random .png files to copy
            val_random_images = random.sample(png_files, val_num_images_to_copy)

            # Copy the selected random images to the corresponding subfolder in the destination folder
            for image in val_random_images:
                source_path = os.path.join(subfolder_source, image)
                destination_path = os.path.join(subfolder_val_destination, image)
                shutil.move(source_path, destination_path)


        #########TEST################
        #create new png file list
        train_num_images_to_copy = round(num_all_img * trainings_test_split)
        test_png_files = [g for g in os.listdir(subfolder_source) if g.endswith(".png")]
        if train_num_images_to_copy > len(test_png_files):
            print(f"Warning: Not enough .png images in subfolder {subfolder_name} for TEST DATASET.")
        else:
            # Choose random .png files to copy
            test_random_images = random.sample(test_png_files, train_num_images_to_copy)

            # Copy the selected random images to the corresponding subfolder in the destination folder
            for image in test_random_images:
                source_path = os.path.join(subfolder_source, image)
                destination_path = os.path.join(subfolder_test_destination, image)
                shutil.move(source_path, destination_path)

    print("validation and test dataset was created")