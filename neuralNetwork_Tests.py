import os
import matplotlib.pyplot as plt
import numpy as np
from tensorflow import keras
from keras import layers
from keras.preprocessing.image import ImageDataGenerator
import time
import pandas as pd
from neuralNetwork import NeuralNetwork


img_height = 32
img_width = 32
num_classes = 3

epochs = 20
batch_size = 8
seed = 123

num_nets_training = 50


testing = False
# Testing parameters
if testing:
    epochs = 1
    num_nets_training = 1

# Data directory, png images
file_directory = os.path.dirname(os.path.abspath(__file__))

date_test = "3C20epo_ResNets"

arr_dataset_tests = [
     "32_3C_all_5",
]

arr_augmentation_tests = [
    
    #"hvflip",
    "hvflip+90",
    #"negative_control",
    #"rot90",
    #"whShift_0.1_r",
    #"whShift_0.1_n",                                                                                                                                                                                                                                                                                                                                          "rot20",
    #"whShift_0.2_n",
    #"brightness_0.9_1.1",
    #"brightness_0.8_1.2",
    #"combination_all",
]

arr_optimizer_tests = [
    "adam",
    # "nadam",
    # "SGD",
    # "RMSprop",
]

arr_layer_tests = [
    # "small",
    # "minimal",
    # "115k",
    # "263k",
    "527k",
    #"ResNet101V2",     #IMPORTANT: Set color_mode in generator to "rgb" (3x for training, validation and test)
    #"ResNet50V2",      #IMPORTANT: Set color_mode in generator to "rgb" (3x for training, validation and test)
]

for dataset in arr_dataset_tests:
    
    save_directory = os.path.join(file_directory, "testregions", dataset)
    val_directory = os.path.join(file_directory, "testregions", "Val_" + dataset)
    test_directory = os.path.join(file_directory, "testregions", "Test_" + dataset)

    for data_augmentation_setting in arr_augmentation_tests:
       
        # no data augmentation for validation data
        validation_datagen = ImageDataGenerator(
            rescale=1.0 / 255.0
        )

        # Data augmentation and preprocessing
        # negative control
        if data_augmentation_setting == "negative_control":
            train_datagen = ImageDataGenerator(
                rescale=1.0 / 255.0,
            )
        # only horizontal + vertical flip
        elif data_augmentation_setting == "hvflip":
            train_datagen = ImageDataGenerator(
                rescale=1.0 / 255.0,
                horizontal_flip=True,
                vertical_flip=True,
            )
        # only rotation, horizontal + vertical flip + random degrees of rotation in between
        elif data_augmentation_setting == "hvflip+90":
            train_datagen = ImageDataGenerator(
                rescale=1.0 / 255.0,
                horizontal_flip=True,
                vertical_flip=True,
                rotation_range=90,
            )
        # only rotation
        elif data_augmentation_setting == "rot90":
            train_datagen = ImageDataGenerator(
                rescale=1.0 / 255.0,
                rotation_range=90,
            )

        elif data_augmentation_setting == "rot20":
            train_datagen = ImageDataGenerator(
                rescale=1.0 / 255.0,
                rotation_range=20,
            )

        # only width and height shift
        elif data_augmentation_setting == "whShift_0.05":
            train_datagen = ImageDataGenerator(
                rescale=1.0 / 255.0,
                width_shift_range=0.05,  # Adjust this value as needed (percentage of image width)
                height_shift_range=0.05,  # Adjust this value as needed (percentage of image height)
            )

        elif data_augmentation_setting == "whShift_0.1_r":
            train_datagen = ImageDataGenerator(
                rescale=1.0 / 255.0,
                fill_mode="reflect",
                width_shift_range=0.1,  # Adjust this value as needed (percentage of image width)
                height_shift_range=0.1,  # Adjust this value as needed (percentage of image height)
            )
        
        elif data_augmentation_setting == "whShift_0.1_n":
            train_datagen = ImageDataGenerator(
                rescale=1.0 / 255.0,
                fill_mode="nearest",
                width_shift_range=0.1,  # Adjust this value as needed (percentage of image width)
                height_shift_range=0.1,  # Adjust this value as needed (percentage of image height)
            )

        elif data_augmentation_setting == "whShift_0.15":
            train_datagen = ImageDataGenerator(
                rescale=1.0 / 255.0,
                width_shift_range=0.15,  # Adjust this value as needed (percentage of image width)
                height_shift_range=0.15,  # Adjust this value as needed (percentage of image height)
            )

        elif data_augmentation_setting == "whShift_0.2_n":
            train_datagen = ImageDataGenerator(
                rescale=1.0 / 255.0,
                fill_mode="nearest",
                width_shift_range=0.2,  # Adjust this value as needed (percentage of image width)
                height_shift_range=0.2,  # Adjust this value as needed (percentage of image height)
            )

        elif data_augmentation_setting == "whShift_0.2_r":
            train_datagen = ImageDataGenerator(
                rescale=1.0 / 255.0,
                fill_mode="reflect",
                width_shift_range=0.3,  # Adjust this value as needed (percentage of image width)
                height_shift_range=0.3,  # Adjust this value as needed (percentage of image height)
            )

        elif data_augmentation_setting == "whShift_0.4":
            train_datagen = ImageDataGenerator(
                rescale=1.0 / 255.0,
                fill_mode="reflect",
                width_shift_range=0.4,  # Adjust this value as needed (percentage of image width)
                height_shift_range=0.4,  # Adjust this value as needed (percentage of image height)
            )

        # only brightness
        elif data_augmentation_setting == "brightness_0.95_1.05":
            train_datagen = ImageDataGenerator(
                rescale=1.0 / 255.0,
                brightness_range=[0.95, 1.05],
            )

        elif data_augmentation_setting == "brightness_0.9_1.1":
            train_datagen = ImageDataGenerator(
                rescale=1.0 / 255.0,
                brightness_range=[0.9, 1.1],
            )

        elif data_augmentation_setting == "brightness_0.85_1.15":
            train_datagen = ImageDataGenerator(
                rescale=1.0 / 255.0,
                brightness_range=[0.85, 1.15],
            )
        elif data_augmentation_setting == "brightness_0.8_1.2":
            train_datagen = ImageDataGenerator(
                rescale=1.0 / 255.0,
                brightness_range=[0.8, 1.2],
            )

        elif data_augmentation_setting == "brightness_0.75_1.25":
            train_datagen = ImageDataGenerator(
                rescale=1.0 / 255.0,
                brightness_range=[0.75, 1.25],
            )

        elif data_augmentation_setting == "brightness_0.7_1.3":
            train_datagen = ImageDataGenerator(
                rescale=1.0 / 255.0,
                brightness_range=[0.7, 1.3],
            )

        # everything combined
        elif data_augmentation_setting == "combination_all":
            train_datagen = ImageDataGenerator(
                rescale=1.0 / 255.0,
                horizontal_flip=True,
                vertical_flip=True,
                rotation_range=20,
                brightness_range=[0.9, 1.1],
                fill_mode="nearest",
                width_shift_range=0.1,  # Adjust this value as needed (percentage of image width)
                height_shift_range=0.1,  # Adjust this value as needed (percentage of image height)
                # width_shift_range=0.05,
                # height_shift_range=0.05,
                #brightness_range=[0.7, 1.3],
            )
        elif data_augmentation_setting == "combined2":
            train_datagen = ImageDataGenerator(
                rescale=1.0 / 255.0,
                horizontal_flip=True,
                vertical_flip=True,
                rotation_range=90,
                # width_shift_range=0.1,
                # height_shift_range=0.1,
                # brightness_range=[0.9, 1.1],
            )


        test_datagen = ImageDataGenerator(
            rescale=1.0 / 255.0
        )

        train_generator = train_datagen.flow_from_directory(
            save_directory,
            color_mode="grayscale", #rgb
            target_size=(img_height, img_width),
            interpolation="bicubic",
            batch_size=batch_size,
            class_mode="sparse",
            shuffle=True,
            seed=seed,
        )

        validation_generator = validation_datagen.flow_from_directory(
            val_directory,
            target_size=(img_height, img_width),
            interpolation="bicubic",
            batch_size=batch_size,
            color_mode="grayscale", #rgb
            class_mode="sparse",
            shuffle=False,  # Do not shuffle validation data for better evaluation
            seed=seed,
        )

        test_generator = test_datagen.flow_from_directory(
            test_directory,
            target_size=(img_height, img_width),
            interpolation="bicubic",
            batch_size=batch_size,
            color_mode="grayscale", #rgb
            class_mode="sparse",
            shuffle=False,  # Do not shuffle test data for better evaluation
            seed=seed,
        )

        for optimizer in arr_optimizer_tests:
            for layer_configuration in arr_layer_tests:
                # Initialize n to count the number of training runs
                n = 0

                # Create an empty DataFrame to store results
                results_df = pd.DataFrame(
                    columns=["Relative path", "Test Accuracy", "Test Loss"]
                )

                # Naming convention
                name_test = (
                    data_augmentation_setting
                    + "_"
                    + optimizer
                    + "_"
                    + layer_configuration
                    + "_"
                    + dataset
                )

                # Results directory
                relative_results_directory = os.path.join(
                    "training_results",
                    date_test + "_" + name_test,
                )
                while n < num_nets_training:
                    print(f"Training run {n+1} of {num_nets_training}")
                    print(
                        f"Current time and date: {time.strftime('%H:%M:%S %d.%m.%Y')}"
                    )
                    print(f"data_augmentation_setting: {data_augmentation_setting}")
                    print(f"optimizer: {optimizer}")
                    print(f"layer_configuration: {layer_configuration}")
                    print(f"dataset: {dataset}")
                    # Results directory
                    current_date = time.strftime("%Y.%m.%d")
                    current_time = time.strftime("%H%M")
                    folder_name = current_date + "_" + current_time + "_run" + str(n)
                    results_path = os.path.join(
                        file_directory, relative_results_directory, folder_name
                    )
                    model, history = NeuralNetwork.neural_network_training(
                        epochs,
                        img_height,
                        img_width,
                        num_classes,
                        train_generator,
                        validation_generator,
                        layer_configuration=layer_configuration,
                        optimizer=optimizer,
                        data_augmentation=data_augmentation_setting
                    )
                    NeuralNetwork.save_training_history_to_folder(history, results_path)
                    NeuralNetwork.generate_trainings_history_plot(history, results_path)
                    NeuralNetwork.save_model_to_folder(model, results_path)

                    # Call the modified function to plot true and false prediction examples for all classes
                    NeuralNetwork.generate_true_and_false_prediction_examples(
                        test_generator, model, results_path
                    )

                    # Evaluate the model on test data
                    loss, accuracy = model.evaluate(test_generator, verbose=0)
                    print(
                        f"Test Loss: {loss:.4f}, Test Accuracy: {accuracy:.4f}"
                    )

                    # Generate predictions for the test data
                    print(f"Predict labels")
                    predicted_labels = model.predict(test_generator).argmax(
                        axis=-1
                    )
                    true_labels = test_generator.classes

                    NeuralNetwork.generate_classification_report(
                        true_labels,
                        predicted_labels,
                        list(test_generator.class_indices.keys()),
                        "Classification_Report.xlsx",
                        results_path,
                    )

                    # Plot the confusion matrix for the test data
                    NeuralNetwork.generate_confusion_matrix(
                        true_labels, predicted_labels, results_path
                    )
                    # German Translation
                    NeuralNetwork.generate_confusion_matrix(
                        true_labels,
                        predicted_labels,
                        results_path,
                        "Confusion_Matrix_DE",
                        translate=True,
                    )

                    new_data = {
                        "Relative path": os.path.join(
                            relative_results_directory, folder_name
                        ),
                        "Test Accuracy": accuracy,
                        "Test Loss": loss,
                    }

                    results_df = pd.concat(
                        [results_df, pd.DataFrame([new_data])], ignore_index=True
                    )
                    # Memory management
                    # Clear model and history
                    del model
                    del history
                    n += 1

                # Save the DataFrame to an Excel file
                results_excel_path = os.path.join(
                    file_directory,
                    relative_results_directory,
                    "results" + "_" + name_test + ".xlsx",
                )
                results_df.to_excel(results_excel_path, index=False)

                print(f"Results saved to {results_excel_path}")
