import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix, classification_report
import time
import pandas as pd
from keras.applications import ResNet50V2, ResNet101V2, ResNet152V2

img_height = 32
img_width = 32


class NeuralNetwork:
    def neural_network_training(
        epochs,
        img_height,
        img_width,
        num_classes,
        train_generator,
        validation_generator,
        layer_configuration="small",
        optimizer="adam",
        data_augmentation = "negative_control",
    ):
        """Trains a CNN to classify images into 5 classes"""
    
        # Define CNN model
        if layer_configuration == "small":
            model = keras.Sequential(
                [
                    layers.Input(shape=(img_height, img_width, 1)),
                    layers.Conv2D(32, 3, padding="same", activation="relu"),
                    layers.Conv2D(32, 3, padding="same", activation="relu"),
                    layers.MaxPooling2D(2),
                    layers.Flatten(),
                    layers.Dropout(0.25),
                    layers.Dense(num_classes),
                ]
            )
        elif layer_configuration == "minimal":
            model = keras.Sequential(
                [
                    layers.Input(shape=(img_height, img_width, 1)),
                    layers.Conv2D(32, 3, padding="same", activation="relu"),
                    layers.MaxPooling2D(2),
                    layers.Flatten(),
                    layers.Dropout(0.25),
                    layers.Dense(num_classes),
                ]
            )

        elif layer_configuration == "115k":
            model = keras.Sequential(
                [
                    layers.Input(shape=(img_height, img_width, 1)),
                    keras.layers.Conv2D(64, 3, activation="relu", padding="same"),
                    keras.layers.MaxPooling2D(2),
                    keras.layers.Conv2D(128, 3, activation="relu", padding="same"),
                    keras.layers.MaxPooling2D(2),
                    keras.layers.Flatten(),
                    keras.layers.Dropout(0.25),
                    keras.layers.Dense(num_classes),
                ]
            )

        elif layer_configuration == "263k":
            model = keras.Sequential(
                [
                    layers.Input(shape=(img_height, img_width, 1)),
                    keras.layers.Conv2D(64, 3, activation="relu", padding="same"),
                    keras.layers.MaxPooling2D(2),
                    keras.layers.Conv2D(128, 3, activation="relu", padding="same"),
                    keras.layers.Conv2D(128, 3, activation="relu", padding="same"),
                    keras.layers.MaxPooling2D(2),
                    keras.layers.Flatten(),
                    keras.layers.Dropout(0.25),
                    keras.layers.Dense(num_classes),
                ]
            )
        elif layer_configuration == "527k":
            model = keras.Sequential(
            [
                layers.Input(shape=(img_height, img_width, 1)),
                keras.layers.Conv2D(64, 3, activation="relu", padding="same"),
                keras.layers.MaxPooling2D(2),
                keras.layers.Conv2D(128, 3, activation="relu", padding="same"),
                keras.layers.Conv2D(128, 3, activation="relu", padding="same"),
                keras.layers.MaxPooling2D(2),
                keras.layers.Conv2D(128, 3, activation="relu", padding="same"),
                keras.layers.Conv2D(128, 3, activation="relu", padding="same"),
                keras.layers.MaxPooling2D(2),
                keras.layers.Flatten(),
                keras.layers.Dropout(0.25),
                keras.layers.Dense(num_classes),
            ]
            )

        elif layer_configuration == "mnist":
            model = keras.Sequential(
                [
                    layers.Input(shape=(img_height, img_width, 1)),
                    keras.layers.Conv2D(64, 7, activation="relu", padding="same"),
                    keras.layers.MaxPooling2D(2),
                    keras.layers.Conv2D(128, 3, activation="relu", padding="same"),
                    keras.layers.Conv2D(128, 3, activation="relu", padding="same"),
                    keras.layers.MaxPooling2D(2),
                    # keras.layers.Conv2D(256, 3, activation="relu", padding="same"),
                    # keras.layers.Conv2D(256, 3, activation="relu", padding="same"),
                    # keras.layers.MaxPooling2D(2),
                    keras.layers.Flatten(),
                    keras.layers.Dense(64, activation="relu"),
                    keras.layers.Dropout(0.25),
                    keras.layers.Dense(num_classes),
                ]
            )

        elif layer_configuration == "ResNet50V2":
            
            _weights = "imagenet"
            trainable = False
            
            base_model = ResNet50V2(weights=_weights, include_top=False, input_shape=(img_height, img_width,3), classes=num_classes)

            #base_model should not be trainable because it's pre-trained
            base_model.trainable = trainable
            x = base_model.output
            x = layers.Flatten()(x)
            #x = keras.layers.Dense(1024, activation="relu")(x)
            x = keras.layers.Dropout(0.2)(x)  #Dense(1024, activation="relu")(x)   
            predictions = layers.Dense(num_classes, activation="relu")(x)
            # Create the model
            model = keras.Model(inputs=base_model.input, outputs=predictions)

        elif layer_configuration == "ResNet101V2":
            
            weights = "imagenet"
            trainable = False
            
            base_model = ResNet101V2(weights=weights, include_top=False, input_shape=(img_height, img_width,3), classes=num_classes)

            #base_model should not be trainable because it's pre-trained
            base_model.trainable = trainable
            x = base_model.output
            x = layers.Flatten()(x)
            #x = keras.layers.Dense(1024, activation="relu")(x)
            x = keras.layers.Dropout(0.2)(x)  #Dense(1024, activation="relu")(x)   
            predictions = layers.Dense(num_classes, activation="relu")(x)
            # Create the model
            model = keras.Model(inputs=base_model.input, outputs=predictions)
        
        # Print model architecture
        # model.summary()
        # Compile the model
        if optimizer == "adam":
            model.compile(
                optimizer=keras.optimizers.Adam(),
                loss=[keras.losses.SparseCategoricalCrossentropy(from_logits=True)],
                metrics=["accuracy"],
            )
        elif optimizer == "nadam":
            model.compile(
                optimizer=keras.optimizers.Nadam(),
                loss=[keras.losses.SparseCategoricalCrossentropy(from_logits=True)],
                metrics=["accuracy"],
            )
        elif optimizer == "SGD":
            model.compile(
                optimizer=keras.optimizers.SGD(),
                loss=[keras.losses.SparseCategoricalCrossentropy(from_logits=True)],
                metrics=["accuracy"],
            )
        elif optimizer == "RMSprop":
            model.compile(
                optimizer=keras.optimizers.RMSprop(),
                loss=[keras.losses.SparseCategoricalCrossentropy(from_logits=True)],
                metrics=["accuracy"],
            )

        # Train the model
        start_time = time.time()
        #callback for early stopping
        #callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)

        history = model.fit(
            train_generator,
            epochs=epochs,
            verbose=1,
            validation_data=validation_generator,
            #callbacks=[callback]
        )
        end_time = time.time()

        # Calculate total training time
        total_training_time = end_time - start_time
        print(f"Total Training Time: {total_training_time:.2f} seconds")
        return model, history

    def save_training_history_to_folder(history, directory):
        """Save the training history to a folder as an .xsls file, create the folder if it does not exist"""
        # Convert the training history dictionary to a DataFrame
        history_df = pd.DataFrame(history.history)

        # Check if the folder exists, and if not, create it
        if not os.path.exists(directory):
            os.makedirs(directory)
        history_df.to_excel(
            os.path.join(
                directory,
                "training_history.xlsx",
            ),
            index=False,
        )

    def generate_trainings_history_plot(history, directory):
        # Plot training history
        plt.figure(figsize=(8, 6))
        plt.plot(history.history["accuracy"], label="Training Accuracy")
        plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
        plt.plot(history.history["loss"], label="Training Loss")
        plt.plot(history.history["val_loss"], label="Validation Loss")
        # plt.title("Training and Validation Accuracy/Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy/Loss")
        plt.legend()
        # plt.show()

        NeuralNetwork.save_plot_to_folder(plt, "Training_History", directory)

    def generate_true_and_false_prediction_examples(generator, model, directory):
        """Plot true and false predictions"""
        img_height = 32
        img_width = 32
        num_classes = len(generator.class_indices)
        class_names = list(generator.class_indices.keys())

        # Create dictionaries to store lists of true and false example images and labels for each class
        true_examples_per_class = {class_name: [] for class_name in class_names}
        false_examples_per_class = {class_name: [] for class_name in class_names}

        # Generate predictions for all images
        all_images = []
        all_true_classes = []
        all_predicted_classes = []

        for i in range(len(generator)):
            image_batch, label_batch = next(generator)
            predicted_classes = model.predict(image_batch).argmax(axis=-1)

            all_images.extend(image_batch)
            all_true_classes.extend(label_batch)
            all_predicted_classes.extend(predicted_classes)

        # Organize examples by class and prediction correctness
        for image, true_class, predicted_class in zip(
            all_images, all_true_classes, all_predicted_classes
        ):
            class_name = class_names[int(true_class)]
            if true_class == predicted_class:  # True prediction
                if len(true_examples_per_class[class_name]) < 5:
                    true_examples_per_class[class_name].append(
                        (image, int(true_class), predicted_class)
                    )
            else:  # False prediction
                if len(false_examples_per_class[class_name]) < 5:
                    false_examples_per_class[class_name].append(
                        (image, int(true_class), predicted_class)
                    )

        # Plot true prediction examples for each class
        plt.figure(figsize=(15, 8))
        for i, class_name in enumerate(class_names):
            true_examples = true_examples_per_class[class_name]
            plt.subplot(2, num_classes, i + 1)
            if len(true_examples) > 0:
                image, true_class, predicted_class = true_examples[
                    0
                ]  # Choose the first true example for plotting
                plt.imshow(image.reshape(img_height, img_width), cmap="gray")
                plt.title(f"True: {true_class}, Predicted: {predicted_class}")
            plt.axis("off")

        # Plot false prediction examples for each class
        for i, class_name in enumerate(class_names):
            false_examples = false_examples_per_class[class_name]
            plt.subplot(2, num_classes, num_classes + i + 1)
            if len(false_examples) > 0:
                image, true_class, predicted_class = false_examples[
                    0
                ]  # Choose the first false example for plotting
                plt.imshow(image.reshape(img_height, img_width), cmap="gray")
                plt.title(f"True: {true_class}, Predicted: {predicted_class}")
            plt.axis("off")


        plt.tight_layout()
        NeuralNetwork.save_plot_to_folder(
            plt,
            "Prediction_Examples",
            directory,
        )

    def generate_classification_report(
        true_labels,
        predicted_labels,
        class_names,
        excel_filename,
        directory,
    ):
        # Generate the classification report
        report = classification_report(
            true_labels, predicted_labels, target_names=class_names, output_dict=True
        )
        report.update(
            {
                "accuracy": {
                    "precision": None,
                    "recall": None,
                    "f1-score": report["accuracy"],
                    "support": report["macro avg"]["support"],
                }
            }
        )
        # Convert the classification report dictionary to a DataFrame
        report_df = pd.DataFrame(report).transpose()
        # Check if the folder exists, and if not, create it
        if not os.path.exists(directory):
            os.makedirs(directory)
        full_file_path = os.path.join(directory, excel_filename)
        
        # Save Report to the specified folder path as an Excel file
        report_df.to_excel(
            full_file_path, index=True
        )  # index=True to include row names

        print(f"Excel file saved to: {full_file_path}")

    def generate_confusion_matrix(
        true_labels,
        predicted_labels,
        directory,
        file_name="Confusion_Matrix",
        translate=False,
    ):
        cm = confusion_matrix(true_labels, predicted_labels)

        # Calculate percentages
        cm_percent = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis] * 100

        plt.figure(figsize=(8, 6))
        plt.imshow(cm_percent, interpolation="nearest", cmap=plt.cm.Blues)
        if translate:
            # plt.title("Konfusionsmatrix")
            plt.xlabel("Vorhergesagte Klasse")
            plt.ylabel("Wahre Klasse")
        else:
            # plt.title("Confusion Matrix")
            plt.xlabel("Predicted Label")
            plt.ylabel("True Label")

        cbar = plt.colorbar(format="%d%%")  # Add the '%' sign to the colorbar labels
        # cbar.set_label("Percentage", rotation=270, labelpad=15)

        num_classes = len(np.unique(true_labels))
        tick_marks = np.arange(num_classes)
        plt.xticks(tick_marks, np.arange(num_classes))
        plt.yticks(tick_marks, np.arange(num_classes))

        # Display percentages in the cells
        for i in range(num_classes):
            for j in range(num_classes):
                if cm_percent[i, j] > 50:
                    color_percent = "white"
                else:
                    color_percent = "black"
                plt.text(
                    j,
                    i,
                    "{:.1f}%".format(cm_percent[i, j]),
                    ha="center",
                    va="center",
                    color=color_percent,
                )

        NeuralNetwork.save_plot_to_folder(
            plt,
            file_name,
            directory,
        )

    def save_model_to_folder(model, directory):
        """Save the model to a folder, create the folder if it does not exist"""
        # Check if the folder exists, and if not, create it
        if not os.path.exists(directory):
            os.makedirs(directory)
        full_file_path = os.path.join(directory, "image_classifier_model.h5")
        # Save the trained model
        model.save(full_file_path)

    def save_plot_to_folder(plt, file_name, directory):
        """Save the plot to a folder, create the folder if it does not exist"""
        format = "png"
        dpi = 600
        if not os.path.exists(directory):
            os.makedirs(directory)
        plt.savefig(
            os.path.join(directory, file_name + "." + format),
            dpi=dpi,
            bbox_inches="tight",
            format=format,
        )
        # print(f"Plot saved to: {directory}")
