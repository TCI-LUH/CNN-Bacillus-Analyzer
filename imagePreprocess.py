import os
import matplotlib.pyplot as plt
import numpy as np
from skimage import color, draw, exposure, filters, io, measure, morphology
import skimage
from regionOfInterest import RegionOfInterest
from sqliteDatabase import LabeledRegionsOfInterestDB as LabeledROIDB


class ImagePreprocess:
    def image_enhance(
        image, show_diagrams=False, intensity_range=(0, 120), filter_size=3
    ):
        # Median Filtering (3 x 3 Neighborhood)
        image_enhanced = filters.median(image, morphology.square(filter_size)) #

        if show_diagrams:
            # Custom figure size (width, height)
            fig, ax = plt.subplots(1, 2, figsize=(18, 6))

            # First column:
            ax[0].imshow(image, cmap="gray")
            ax[0].set_title("Original Image")
            ax[0].axis("off")

            # Second column:
            ax[1].imshow(image_enhanced, cmap="gray")
            ax[1].set_title("Contrast enhanced and noise reduced Image")
            ax[1].axis("off")

            plt.tight_layout()
            plt.show()
        return image_enhanced

    def image_thresholding(image, show_diagrams=False, window_size=7, k=0.06):
        # Thresholding with Sauvola local thresholding
        threshold = filters.threshold_sauvola(image, window_size=window_size, k=k)
        # apply threshold
        image_binary = image > threshold

        if show_diagrams:
            # Custom figure size (width, height)
            fig, ax = plt.subplots(1, 2, figsize=(18, 6))

            # First column:
            ax[0].imshow(image, cmap="gray")
            ax[0].set_title("Enhanced Image")
            ax[0].axis("off")

            # Second column:
            ax[1].imshow(image_binary, cmap="gray")
            ax[1].set_title("Thresholded Image")
            ax[1].axis("off")

            plt.tight_layout()
            plt.show()
        return image_binary

    def image_opening(image_binary, show_diagrams=False, filter_size=3):
        return morphology.opening(image_binary, morphology.square(filter_size))

    def get_regions_of_interest(
        file_name,
        image_labels,
        min_pixel_size=30,
        max_pixel_size=1000,
        check_border=True,
    ):
        """
        Returns a list of bounding boxes for all regions in the image that are larger than min_pixel_size, smaller than max_pixel_size, and optionally not touching the border.
        """
        ls_regions_of_interest = []
        image_height, image_width = image_labels.shape

        for region in measure.regionprops(image_labels):
            if region.area >= min_pixel_size and region.area <= max_pixel_size:
                if not check_border:
                    ls_regions_of_interest.append(
                        RegionOfInterest(
                            file_name,
                            region.bbox[0],
                            region.bbox[1],
                            region.bbox[2],
                            region.bbox[3],
                            None,
                        )
                    )
                elif (
                    region.bbox[0] != 0
                    and region.bbox[1] != 0
                    and region.bbox[2] != image_height
                    and region.bbox[3] != image_width
                ):
                    ls_regions_of_interest.append(
                        RegionOfInterest(
                            file_name,
                            region.bbox[0],
                            region.bbox[1],
                            region.bbox[2],
                            region.bbox[3],
                            None,
                        )
                    )
        return ls_regions_of_interest

    # technically a bit redundant
    def region_of_interest_resize(image, region_of_interest, target_size):
        """
        Resizes the bounding box to the target_size.
        """

        image_height = image.shape[0]
        image_width = image.shape[1]

        (
            min_row,
            min_col,
            max_row,
            max_col,
        ) = region_of_interest.get_coordinates()

        # Calculate height and width (adjust)
        current_height = max_row - min_row
        if current_height % 2 == 1:
            max_row -= 1
            current_height = max_row - min_row
        current_width = max_col - min_col

        if current_width % 2 == 1:
            max_col -= 1
            current_width = max_col - min_col

        pad_height = target_size - current_height
        pad_width = target_size - current_width

        new_min_row = max(min_row - pad_height / 2, 0)
        new_min_col = max(min_col - pad_width / 2, 0)

        new_max_row = min(max_row + pad_height / 2, image_height)
        new_max_col = min(max_col + pad_width / 2, image_width)

        if new_min_row == 0:
            new_max_row = target_size
        elif new_max_row == image_height:
            new_min_row = image_height - target_size

        if new_min_col == 0:
            new_max_col = target_size
        elif new_max_col == image_width:
            new_min_col = image_width - target_size

        return RegionOfInterest(
            region_of_interest.file_name,
            new_min_row,
            new_min_col,
            new_max_row,
            new_max_col,
            region_of_interest.label,
        )

    def regions_of_interest_resize(image, ls_regions_of_interest, target_size):
        """
        Resizes each bounding box to the target_size.
        """

        image_height = image.shape[0]
        image_width = image.shape[1]

        ls_resized_regions_of_interest = []
        for region_of_interest in ls_regions_of_interest:
            (
                min_row,
                min_col,
                max_row,
                max_col,
            ) = region_of_interest.get_coordinates()

            # Calculate height and width (adjust)
            current_height = max_row - min_row
            if current_height % 2 == 1:
                max_row -= 1
                current_height = max_row - min_row
            current_width = max_col - min_col

            if current_width % 2 == 1:
                max_col -= 1
                current_width = max_col - min_col

            pad_height = target_size - current_height
            pad_width = target_size - current_width

            new_min_row = max(min_row - pad_height / 2, 0)
            new_min_col = max(min_col - pad_width / 2, 0)

            new_max_row = min(max_row + pad_height / 2, image_height)
            new_max_col = min(max_col + pad_width / 2, image_width)

            if new_min_row == 0:
                new_max_row = target_size
            elif new_max_row == image_height:
                new_min_row = image_height - target_size

            if new_min_col == 0:
                new_max_col = target_size
            elif new_max_col == image_width:
                new_min_col = image_width - target_size

            ls_resized_regions_of_interest.append(
                RegionOfInterest(
                    region_of_interest.file_name,
                    new_min_row,
                    new_min_col,
                    new_max_row,
                    new_max_col,
                    region_of_interest.label,
                )
            )
        return ls_resized_regions_of_interest

    def region_of_interest_draw(
        image, region_of_interest, box_color=(0, 255, 0), show_diagrams=False
    ):
        """
        Draws a bounding box on the image.
        """
        if len(image.shape) == 2:
            # print("Image is grayscale, converting to RGB")
            image = color.gray2rgb(image)
        # Check if image data is of type uint8 for compatibility with scikit-image

        row, column = draw.rectangle_perimeter(
            (region_of_interest.min_row, region_of_interest.min_col),
            extent=(
                region_of_interest.max_row - region_of_interest.min_row,
                region_of_interest.max_col - region_of_interest.min_col,
            ),
            shape=image.shape,
        )
        image[row, column] = box_color
        if show_diagrams:
            # Custom figure size (width, height)
            fig, ax = plt.subplots(1, 1, figsize=(18, 6))

            # First column:
            ax.imshow(image, cmap="gray")
            ax.set_title("Image with Bounding Box")
            ax.axis("off")

            plt.tight_layout()
            plt.show()
        return image

    def regions_of_interest_draw(
        image, ls_regions_of_interest, box_color=(0, 255, 0), show_diagrams=False
    ):
        """
        Draws bounding boxes on the image for each bounding box in the list.
        """
        # print(image.dtype)
        if len(image.shape) == 2:
            # print("Image is grayscale, converting to RGB")
            image = color.gray2rgb(image)

        for region_of_interest in ls_regions_of_interest:
            row, column = draw.rectangle_perimeter(
                (region_of_interest.min_row, region_of_interest.min_col),
                extent=(
                    region_of_interest.max_row - region_of_interest.min_row,
                    region_of_interest.max_col - region_of_interest.min_col,
                ),
                shape=image.shape,
            )
            image[row, column] = box_color
        if show_diagrams:
            # Custom figure size (width, height)
            fig, ax = plt.subplots(1, 1, figsize=(18, 6))

            # First column:
            ax.imshow(image, cmap="gray")
            ax.set_title("Image with Bounding Box")
            ax.axis("off")

            plt.tight_layout()
            plt.show()
        return image

    def regions_of_interest_save(image, ls_regions_of_interest, save_directory, color_images=False):
        """
        Saves each bounding box as a separate image in the save_directory.
        """
        # Create the main save directory if it does not exist
        if not os.path.exists(save_directory):
            os.makedirs(save_directory)

        i = 0  # initialize counter
        for region_of_interest in ls_regions_of_interest:
            region_image = image[
                int(region_of_interest.min_row) : int(region_of_interest.max_row),
                int(region_of_interest.min_col) : int(region_of_interest.max_col),
            ]
            sub_save_directory = os.path.join(
                save_directory, str(region_of_interest.label)
            )
            if not os.path.exists(sub_save_directory):
                os.makedirs(sub_save_directory)

            file_path = os.path.join(
                sub_save_directory,
                f"r{i}_" + region_of_interest.file_name[:-4] + ".png",
            )
            # test - save as color image
            if(color_images):
                region_image = skimage.color.gray2rgb(region_image)

            io.imsave(file_path, region_image)
            
            i += 1  # increment counter only if the condition is met, consistent naming

    def preprocess_image(folder_path, file_name, show_diagrams=False):
        """
        Preprocesses the image and returns the regions of interest.
        """
        # Construct the full path to the image
        file_path = os.path.join(folder_path, file_name)

        # Load the image
        image = io.imread(file_path)

        # Enhance the image
        image_enhanced = ImagePreprocess.image_enhance(
            image, show_diagrams=show_diagrams
        )

        # Threshold the image
        image_binary = ImagePreprocess.image_thresholding(
            image_enhanced, show_diagrams=show_diagrams
        )

        # Perform Image Opening (Erosion followed by Dilation)
        # (In this case the white pixels are eroded, so its actually a closing operation on the black pixels)
        image_closing = ImagePreprocess.image_opening(image_binary)

        # Labeling of Regions
        image_labels = measure.label(image_closing, 1)

        # Get the regions of interest
        return ImagePreprocess.get_regions_of_interest(file_name, image_labels)

    def preprocess_images_folder(folder_path, image_type=".png"):
        """
        Preprocess all the images in the folder and returns all the regions of interest.
        """

        # Get all files in the folder
        files = os.listdir(folder_path)

        # Get all files ending with .bmp
        files = [file for file in files if file.endswith(image_type)]

        # Initialize list to store all regions of interest
        ls_regions_of_interest = []
        for file in files:
            print(file)
            ls_regions_of_interest.extend(
                ImagePreprocess.preprocess_image(folder_path, file)
            )
        print(ls_regions_of_interest)
        return ls_regions_of_interest

    def process_labeled_image(folder_path, file_name, save_directory, image_size, color_images=False):
        """
        Processes the image and saves the regions of interest as individual images.
        """

        # Construct the full path to the image
        file_path = os.path.join(folder_path, file_name)

        # Load the image
        image = io.imread(file_path)

        #median
        image = filters.median(image=image) #footprint=morphology.square(3)


        current_ls_roi = LabeledROIDB("LabelDatabase.db").get_all_roi_w_file_name(
            file_name
        )
        current_ls_roi = ImagePreprocess.regions_of_interest_resize(
            image, current_ls_roi, image_size
        )
        ImagePreprocess.regions_of_interest_save(image, current_ls_roi, save_directory, color_images=False)

    def process_labeled_images_folder(folder_path, save_directory, image_size=32, color_images = False):
        """
        Processes the images in the folder and saves the regions of interest as individual images.
        """
        # Get all files in the folder
        files = os.listdir(folder_path)

        # Get all files ending with .bmp
        files = [file for file in files if file.endswith(".bmp")]

        for file in files:
            print(file)
            ImagePreprocess.process_labeled_image(
                folder_path, file, save_directory, image_size
            )


# ###Testing
# show_diagrams = False
# # Get the directory
# file_directory = os.path.dirname(os.path.abspath(__file__))

# # Name of the file
# # DOE1 Test

# # Construct relative path to the folder containing the image
# folder_path = os.path.join(file_directory, "data", "Bilder B.coagulans", "DOE1Test")

# file_name = "20220802_155316_DOE1 t5 Test_2_20_BCoa_24_10_3.bmp"
# # file_name = "20220802_163142_DOE1 t5 Test_4_23_BCoa_24_20_1.bmp"
# # file_name = "20220802_155316_DOE1 t5 Test_2_20_BCoa_24_10_3.bmp"

# # Sample1
# # folder_path = os.path.join(file_directory, "data", "Bilder B.coagulans", "Sample1")
# # file_name = "20230627_160328_t2_1_7_BCoa_4_10_1.bmp"
# # file_name = "20230627_161020_t2_1_15_BCoa_4_10_2.bmp"

# # Construct the full path to the image
# file_path = os.path.join(folder_path, file_name)


# ls_regions_of_interest = ImagePreprocess.preprocess_image(
#     folder_path, file_name, show_diagrams=False
# )


# # Load the image
# image = io.imread(file_path)

# image = ImagePreprocess.regions_of_interest_draw(
#     image, ls_regions_of_interest, show_diagrams=True
# )
# ls_regions_of_interest_resized = ImagePreprocess.regions_of_interest_resize(
#     image, ls_regions_of_interest, 30
# )

# ImagePreprocess.region_of_interest_draw(
#     image, ls_regions_of_interest_resized[50], show_diagrams=True, box_color=(255, 0, 0)
# )


# # save_directory = "testregions"
# # regions_of_interest_save(image, boundingboxes_30, save_directory)
# # print("Regions saved to directory: ", save_directory)
