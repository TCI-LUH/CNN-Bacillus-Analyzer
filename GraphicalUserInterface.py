import os
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QFileDialog,
    QGraphicsScene,
)
from PyQt5.QtCore import QRectF, Qt
from PyQt5.QtGui import QPixmap, QImage
from PyQt5 import uic
from skimage import io, color
import numpy as np
from imagePreprocess import ImagePreprocess
from sqliteDatabase import LabeledRegionsOfInterestDB as LabeledROIDB


class MyGUI(QMainWindow):
    def __init__(
        self,
    ):
        super().__init__()
        self.database = LabeledROIDB("LabelDatabase.db")
        # Load the GUI
        uic.loadUi("MyGUI.ui", self)
        # Set the text to be wrapped for the label file path and the labeled file name
        self.label_file_path.setWordWrap(True)
        self.label_file_name.setWordWrap(True)

        # Set table widget Header Labels
        self.tableWidget.setHorizontalHeaderLabels(["Region", "Labels"])

        # Create the QGraphicsScene and set it to the QGraphicsView
        self.scene = QGraphicsScene()
        self.QgraphicsView.setScene(self.scene)

        # Connect the open file button/ open folder button to the open_file_pressed and open_folder_pressed functions
        self.actionOpen_File.triggered.connect(self.open_file_pressed)
        self.actionOpen_File.setShortcut("Ctrl+O")

        self.actionOpen_Folder.triggered.connect(self.open_folder_pressed)
        self.actionOpen_Folder.setShortcut("Ctrl+F")

        # Connect the push buttons to their respective functions
        self.pushButton_OriginalImg.clicked.connect(self.pushButton_OriginalImg_clicked)
        self.pushButton_Enhance.clicked.connect(self.pushButton_Enhance_clicked)
        self.pushButton_Binarized.clicked.connect(self.pushButton_Binarized_clicked)
        self.pushButton_DrawBoxes.clicked.connect(self.pushButton_DrawBoxes_clicked)

        self.pushButton_StartLabeling.clicked.connect(
            self.pushButton_StartLabeling_clicked
        )
        self.pushButton_StartLabeling.setShortcut("Space")

        self.pushButton_NxtImage.clicked.connect(self.pushButton_NxtImage_clicked)
        self.pushButton_NxtImage.setShortcut("Ctrl+Right")

        self.pushButton_PrevImage.clicked.connect(self.pushButton_PrevImage_clicked)
        self.pushButton_PrevImage.setShortcut("Ctrl+Left")

        self.pushButton_NextROI.clicked.connect(self.pushButton_NextROI_clicked)
        self.pushButton_NextROI.setShortcut("Right")
        self.pushButton_PreviousROI.clicked.connect(self.pushButton_PreviousROI_clicked)
        self.pushButton_PreviousROI.setShortcut("Left")

        self.pushButton_LabelOther.clicked.connect(self.pushButton_LabelOther_clicked)
        self.pushButton_LabelOther.setShortcut("0")

        self.pushButton_LabelCell.clicked.connect(self.pushButton_LabelCell_clicked)
        self.pushButton_LabelCell.setShortcut("1")

        self.pushButton_LabelPrespore.clicked.connect(
            self.pushButton_LabelPrespore_clicked
        )
        self.pushButton_LabelPrespore.setShortcut("2")

        self.pushButton_LabelSpore.clicked.connect(self.pushButton_LabelSpore_clicked)
        self.pushButton_LabelSpore.setShortcut("3")

        self.pushButton_Label2cells.clicked.connect(self.pushButton_Label2cells_clicked)
        self.pushButton_Label2cells.setShortcut("4")

        self.pushButton_Label3cells.clicked.connect(self.pushButton_Label3cells_clicked)
        self.pushButton_Label3cells.setShortcut("5")

        self.pushButton_Labelmulticells.clicked.connect(
            self.pushButton_Labelmulticells_clicked
        )
        self.pushButton_Labelmulticells.setShortcut("6")

        self.pushButton_LabelNotSure.clicked.connect(
            self.pushButton_LabelNotSure_clicked
        )
        self.pushButton_LabelNotSure.setShortcut("9")

        self.checkBox_AutoZoom.stateChanged.connect(self.update_label_and_region)
        self.checkBox_Database.stateChanged.connect(self.checkBox_Database_clicked)

        # show the GUI
        self.show()

    def checkBox_Database_clicked(self):
        if self.checkBox_Database.isChecked():
            self.database = LabeledROIDB("TempLabelDatabase.db")
        else:
            self.database = LabeledROIDB("LabelDatabase.db")

        self.current_ls_roi = LabeledROIDB.get_all_roi_w_file_name(
            self.database, os.path.basename(self.current_file_path)
        )
        if len(self.current_ls_roi) != 0:
            self.pushButton_StartLabeling.setEnabled(True)
            self.tableWidget.setRowCount(len(self.current_ls_roi))
            tablerow = 0
            for roi in self.current_ls_roi:
                self.tableWidget.setItem(
                    tablerow,
                    0,
                    QtWidgets.QTableWidgetItem(roi.get_coordinates_string()),
                )
                self.tableWidget.setItem(
                    tablerow, 1, QtWidgets.QTableWidgetItem(str(roi.label))
                )
                tablerow += 1
        else:
            self.pushButton_StartLabeling.setEnabled(False)
            self.tableWidget.setRowCount(0)

    def set_label(self, label):
        self.current_ls_roi[self.roi_index].label = label
        self.label_CurrentLabel.setText(
            "Current Label: " + str(self.current_ls_roi[self.roi_index].label)
        )
        self.tableWidget.setItem(
            self.roi_index, 1, QtWidgets.QTableWidgetItem(str(label))
        )
        self.database.update_label(self.current_ls_roi[self.roi_index], label)

    def update_label_and_region(self):
        self.current_image = io.imread(self.current_file_path)
        if self.roi_index != None:
            self.tableWidget.selectRow(self.roi_index)
            if self.checkBox_AutoZoom.isChecked():
                self.current_image = ImagePreprocess.region_of_interest_draw(
                    self.current_image,
                    ImagePreprocess.region_of_interest_resize(
                        self.current_image, self.current_ls_roi[self.roi_index], 32
                    ),
                )
                tempROI = ImagePreprocess.region_of_interest_resize(
                    self.current_image, self.current_ls_roi[self.roi_index], 300
                )
                self.current_image = self.current_image[
                    int(tempROI.min_row) : int(tempROI.max_row),
                    int(tempROI.min_col) : int(tempROI.max_col),
                ]
            else:
                self.current_image = ImagePreprocess.region_of_interest_draw(
                    self.current_image,
                    ImagePreprocess.region_of_interest_resize(
                        self.current_image, self.current_ls_roi[self.roi_index], 32
                    ),
                )
            self.label_CurrentLabel.setText(
                "Current Label: " + str(self.current_ls_roi[self.roi_index].label)
            )

        self.display_image()

    def display_image(self):
        if len(self.current_image.shape) == 2:
            self.current_image = color.gray2rgb(self.current_image)
        if self.current_image.dtype != np.uint8:
            if np.max(self.current_image) <= 1.0:
                self.current_image = (self.current_image * 255).astype(
                    np.uint8
                )  # image = float [0,1]? convert to uint8
        else:
            self.current_image = self.current_image.astype(
                np.uint8
            )  # image = float [0, 255]? convert to uint8
        try:
            # Convert the image to QImage
            height, width, channel = self.current_image.shape
            qimage = QImage(
                self.current_image.data,
                width,
                height,
                channel * width,
                QImage.Format_RGB888,
            )

            # Create a QPixmap from the QImage
            pixmap = QPixmap.fromImage(qimage)

            # Clear the scene and add the pixmap item to the scene
            self.scene.clear()

            # # Fit the image to the view
            self.QgraphicsView.fitInView(QRectF(pixmap.rect()), Qt.KeepAspectRatio)
            self.scene.addPixmap(pixmap)
        except Exception as e:
            print("An error occurred:", str(e))

    def load_image_and_data(self):
        self.roi_index = None
        self.pushButton_LabelOther.setEnabled(False)
        self.pushButton_LabelCell.setEnabled(False)
        self.pushButton_LabelPrespore.setEnabled(False)
        self.pushButton_LabelSpore.setEnabled(False)
        self.pushButton_Label2cells.setEnabled(False)
        self.pushButton_Label3cells.setEnabled(False)
        self.pushButton_LabelNotSure.setEnabled(False)
        self.pushButton_Labelmulticells.setEnabled(False)
        self.pushButton_NextROI.setEnabled(False)
        self.pushButton_PreviousROI.setEnabled(False)

        self.label_file_path.setText("PATH: " + self.current_file_path)
        self.label_file_name.setText(
            "file: " + os.path.basename(self.current_file_path)
        )
        # print("File exists")
        image = io.imread(self.current_file_path)
        self.current_image = image
        self.display_image()
        self.pushButton_OriginalImg.setEnabled(False)
        self.pushButton_Enhance.setEnabled(True)
        self.pushButton_Binarized.setEnabled(True)
        self.pushButton_DrawBoxes.setEnabled(True)

        self.current_ls_roi = LabeledROIDB.get_all_roi_w_file_name(
            self.database, os.path.basename(self.current_file_path)
        )
        if len(self.current_ls_roi) != 0:
            self.pushButton_StartLabeling.setEnabled(True)
            self.tableWidget.setRowCount(len(self.current_ls_roi))
            tablerow = 0
            for roi in self.current_ls_roi:
                self.tableWidget.setItem(
                    tablerow,
                    0,
                    QtWidgets.QTableWidgetItem(roi.get_coordinates_string()),
                )
                self.tableWidget.setItem(
                    tablerow, 1, QtWidgets.QTableWidgetItem(str(roi.label))
                )
                tablerow += 1
        else:
            self.pushButton_StartLabeling.setEnabled(False)
            self.tableWidget.setRowCount(0)
        if self.current_file_path:
            self.update_label_and_region()

    def open_file_pressed(self):
        # print("Open file pressed")
        # Get the directory
        file_directory = os.path.dirname(os.path.abspath(__file__))

        # Construct relative path to the folder containing the image
        folder_path = os.path.join(file_directory, "data", "Bilder B.coagulans")

        file_path = QFileDialog.getOpenFileName(
            self, "Open file", folder_path, "BMP (*.bmp)"
        )
        current_file_path = file_path[0]
        if os.path.isfile(current_file_path):
            self.current_file_path = current_file_path
            self.load_image_and_data()
            self.pushButton_PrevImage.setEnabled(False)
            self.pushButton_NxtImage.setEnabled(False)
        else:
            print("No file chosen")

    def open_folder_pressed(self):
        # Get the directory
        file_directory = os.path.dirname(os.path.abspath(__file__))
        folder_path = os.path.join(file_directory, "data", "Bilder B.coagulans")
        current_directory = QFileDialog.getExistingDirectory(
            self, "Open folder", folder_path
        )
        if current_directory:
            self.current_directory = current_directory
            # Get all files in the chosen directory
            self.current_files_list = os.listdir(self.current_directory)
            # print(files)
            # Get all files ending with .bmp
            self.current_files_list = [
                file for file in self.current_files_list if file.endswith(".bmp")
            ]
            self.current_image_index = 0
            self.current_file_path = os.path.join(
                self.current_directory,
                self.current_files_list[self.current_image_index],
            )
            self.pushButton_PrevImage.setEnabled(False)
            self.pushButton_NxtImage.setEnabled(True)
            self.label_imageNumber.setText(
                "{} from {}".format(
                    self.current_image_index + 1, len(self.current_files_list)
                )
            )
            self.load_image_and_data()
        else:
            print("No directory chosen")

    def pushButton_NxtImage_clicked(self):
        if self.current_image_index < len(self.current_files_list) - 1:
            self.current_image_index += 1
            self.current_file_path = os.path.join(
                self.current_directory,
                self.current_files_list[self.current_image_index],
            )
            self.label_imageNumber.setText(
                "{} from {}".format(
                    self.current_image_index + 1, len(self.current_files_list)
                )
            )
        self.load_image_and_data()
        if self.current_image_index == len(self.current_files_list) - 1:
            self.pushButton_NxtImage.setEnabled(False)
        if self.current_image_index > 0:
            self.pushButton_PrevImage.setEnabled(True)

    def pushButton_PrevImage_clicked(self):
        if self.current_image_index > 0:
            self.current_image_index -= 1
            self.current_file_path = os.path.join(
                self.current_directory,
                self.current_files_list[self.current_image_index],
            )
            self.label_imageNumber.setText(
                "{} from {}".format(
                    self.current_image_index + 1, len(self.current_files_list)
                )
            )
            self.load_image_and_data()
        if self.current_image_index == 0:
            self.pushButton_PrevImage.setEnabled(False)
        if self.current_image_index < len(self.current_files_list) - 1:
            self.pushButton_NxtImage.setEnabled(True)

    def pushButton_OriginalImg_clicked(self):
        self.current_image = io.imread(self.current_file_path)
        self.display_image()
        self.pushButton_OriginalImg.setEnabled(False)
        self.pushButton_Enhance.setEnabled(True)
        self.pushButton_Binarized.setEnabled(True)
        self.pushButton_DrawBoxes.setEnabled(True)

    def pushButton_Enhance_clicked(self):
        image = io.imread(self.current_file_path)
        image = ImagePreprocess.image_enhance(image)
        self.current_image = image
        self.display_image()
        self.pushButton_OriginalImg.setEnabled(True)
        self.pushButton_Enhance.setEnabled(False)
        self.pushButton_Binarized.setEnabled(True)
        self.pushButton_DrawBoxes.setEnabled(True)

    def pushButton_Binarized_clicked(self):
        image = io.imread(self.current_file_path)
        image = ImagePreprocess.image_thresholding(image)
        image = ImagePreprocess.image_opening(image)
        self.current_image = image
        self.display_image()
        self.pushButton_OriginalImg.setEnabled(True)
        self.pushButton_Enhance.setEnabled(True)
        self.pushButton_Binarized.setEnabled(False)
        self.pushButton_DrawBoxes.setEnabled(True)

    def pushButton_DrawBoxes_clicked(self):
        image = io.imread(self.current_file_path)
        # regions_of_interest = LabeledROIDB.get_all_roi_w_file_name(
        #     self.database, os.path.basename(self.current_file_path)
        # )

        regions_of_interest = ImagePreprocess.preprocess_image(
            os.path.dirname(self.current_file_path),
            os.path.basename(self.current_file_path),
        )
        image = ImagePreprocess.regions_of_interest_draw(image, regions_of_interest)
        self.current_image = image
        self.display_image()
        self.pushButton_OriginalImg.setEnabled(True)
        self.pushButton_Enhance.setEnabled(True)
        self.pushButton_Binarized.setEnabled(True)
        self.pushButton_DrawBoxes.setEnabled(False)

    def pushButton_StartLabeling_clicked(self):
        self.roi_index = 0
        self.update_label_and_region()
        self.pushButton_StartLabeling.setEnabled(False)
        self.pushButton_NextROI.setEnabled(True)
        self.pushButton_LabelOther.setEnabled(True)
        self.pushButton_LabelCell.setEnabled(True)
        self.pushButton_LabelPrespore.setEnabled(True)
        self.pushButton_LabelSpore.setEnabled(True)
        self.pushButton_Label2cells.setEnabled(True)
        self.pushButton_Label3cells.setEnabled(True)
        self.pushButton_LabelNotSure.setEnabled(True)
        self.pushButton_Labelmulticells.setEnabled(True)

        self.checkBox_Database.setEnabled(False)

    def pushButton_NextROI_clicked(self):
        if self.roi_index < len(self.current_ls_roi) - 1:
            self.roi_index += 1
            self.update_label_and_region()
        if self.roi_index == len(self.current_ls_roi) - 1:
            self.pushButton_NextROI.setEnabled(False)
        if self.roi_index > 0:
            self.pushButton_PreviousROI.setEnabled(True)

    def pushButton_PreviousROI_clicked(self):
        if self.roi_index > 0:
            self.roi_index -= 1
            self.update_label_and_region()
        if self.roi_index == 0:
            self.pushButton_PreviousROI.setEnabled(False)
        if self.roi_index < len(self.current_ls_roi) - 1:
            self.pushButton_NextROI.setEnabled(True)

    def pushButton_LabelOther_clicked(self):
        self.set_label(0)
        self.pushButton_NextROI_clicked()

    def pushButton_LabelCell_clicked(self):
        self.set_label(1)
        self.pushButton_NextROI_clicked()

    def pushButton_LabelPrespore_clicked(self):
        self.set_label(2)
        self.pushButton_NextROI_clicked()

    def pushButton_LabelSpore_clicked(self):
        self.set_label(3)
        self.pushButton_NextROI_clicked()

    def pushButton_Label2cells_clicked(self):
        self.set_label(4)
        self.pushButton_NextROI_clicked()

    def pushButton_Label3cells_clicked(self):
        self.set_label(5)
        self.pushButton_NextROI_clicked()

    def pushButton_Labelmulticells_clicked(self):
        self.set_label(6)
        self.pushButton_NextROI_clicked()

    def pushButton_LabelNotSure_clicked(self):
        self.set_label(9)
        self.pushButton_NextROI_clicked()

    def closeEvent(self, event):
        self.database.close()
        # print("Connection closed")
        event.accept()


def main():
    app = QApplication([])
    window = MyGUI()
    app.exec_()


if __name__ == "__main__":
    main()
