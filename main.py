import sys
from PyQt5.QtGui import QPixmap, QImage
import cv2
import numpy as np
import os
from PyQt5.QtCore import QCoreApplication
from PyQt5.QtCore import Qt
import sys
import os
import cv2
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QLabel, QPushButton, QFileDialog, QComboBox
from PyQt5.QtGui import QPixmap, QImage
from skimage.morphology import skeletonize


def skeleton_image_binarization(image):
    green_thres = 120
    red_thres = 110

    red = image[:,:,2]
    green = image[:,:,1]

    red_mask = red < red_thres
    green_mask = green > green_thres

    mask = red_mask & green_mask
    return mask

def binarization_postprocessing(image):
    kernel = np.ones((20, 20), np.uint8)
    image = cv2.dilate(image.astype(np.uint8), kernel, iterations=1)
    image = cv2.morphologyEx(image.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
    return image

def retrieve_skeleton(image):
    image = skeletonize(image)
    return image

def skeleton_postprocessing(image):
    kernel = np.ones((12,12), np.uint8)
    image = cv2.dilate(image.astype(np.uint8), kernel, iterations=1)
    return image

def get_skeleton_pro(image):
    history = [image]
    image = skeleton_image_binarization(image)
    history.append(image.astype(np.uint8) * 255)
    image = binarization_postprocessing(image)
    history.append(image.astype(np.uint8) * 255)
    image = retrieve_skeleton(image)
    history.append(image.astype(np.uint8) * 255)
    image = skeleton_postprocessing(image)
    history.append(image.astype(np.uint8) * 255)
    return image, history


def scale_array(array):
    min_value = np.min(array)
    max_value = np.max(array)
    scaled_array = (array - min_value) / (max_value - min_value)
    return scaled_array

def vertices_binary(image):
    blue_mask = image[:,:,0] < 40
    red_mask = image[:,:,2] < 40
    green_mask = image[:,:,1] < 40

    mask = blue_mask & red_mask & green_mask
    return mask

def clean_vertices_image(image):
    kernel = np.ones((3,3), np.uint8)
    image = cv2.erode(image.astype(np.uint8), kernel, iterations=1)
    kernel = np.ones((7,7), np.uint8)
    image = cv2.dilate(image.astype(np.uint8), kernel, iterations=1)
    kernel = np.ones((15,15), np.uint8)
    image = cv2.morphologyEx(image.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
    return image

def make_vertices_thicker(image):
    kernel = np.ones((25, 25), np.uint8)
    image = cv2.dilate(image.astype(np.uint8), kernel, iterations=1)
    return image

def impose_vertices_on_skeleton(skeleton, vertices):
    skeleton = scale_array(skeleton)
    vertices = scale_array(vertices)
    kernel = np.ones((5, 5), np.uint8)
    vertices = cv2.dilate(vertices.astype(np.uint8), kernel, iterations=1)
    kernel = np.ones((10, 10), np.uint8)
    skeleton = cv2.dilate(skeleton.astype(np.uint8), kernel, iterations=1)

    skeleton = skeleton > 0.5
    vertices = vertices > 0.5
    mask = skeleton & vertices
    return mask

def get_skeleton_begginer(image):
    # make open close and dilation operations to clean image
    kernel = np.ones((5,5), np.uint8)
    image = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
    image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
    image = cv2.dilate(image, kernel, iterations=1)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    _, binary_image = cv2.threshold(edges, 127, 255, cv2.THRESH_BINARY)
    binary_image = binary_image // 255
    skeleton = skeletonize(binary_image)
    skeleton = (skeleton * 255).astype(np.uint8)
    #  dilate the image to make the lines thicker
    kernel = np.ones((16,16), np.uint8)
    skeleton = cv2.dilate(skeleton, kernel, iterations=1)
    return skeleton

def get_vertices(image, figure_mask):
    history = [image]
    image = vertices_binary(image)
    history.append(image.astype(np.uint8) * 255)
    image = clean_vertices_image(image)
    history.append(image.astype(np.uint8) * 255)
    image = make_vertices_thicker(image)
    history.append(image.astype(np.uint8) * 255)
    image = impose_vertices_on_skeleton(figure_mask, image)
    history.append(image.astype(np.uint8) * 255)
    return image, history


def get_vertices_centers(image):
    image = image.astype(np.uint8) * 255
    edges = cv2.Canny(image, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # filter couhtours by area
    contours = [contour for contour in contours if cv2.contourArea(contour) > 100]

    print(len(contours))
    centers = []
    for contour in contours:
        M = cv2.moments(contour)
        if M['m00'] != 0:
            center_x = int(M['m10'] / M['m00'])
            center_y = int(M['m01'] / M['m00'])
            centers.append((center_x, center_y))
    
    # clear duplicates
    centers = list(set(centers))
    return centers

def draw_vertices_centers(image, centers):
    for center in centers:
        cv2.circle(image, center, 10, (0, 0, 255), -1)
    return image
    

def crop_images_around_centers(image, centers, crop_size):
    cropped_images = []
    for center in centers:
        x, y = center
        x1 = max(0, x - crop_size // 2)
        x2 = min(image.shape[1], x + crop_size // 2)
        y1 = max(0, y - crop_size // 2)
        y2 = min(image.shape[0], y + crop_size // 2)

        # Проверяем, чтобы область обрезки была допустимого размера
        if x2 > x1 and y2 > y1:
            cropped_image = image[y1:y2, x1:x2]
            cropped_images.append(cropped_image)
        else:
            print(f"Invalid crop area for center {center} with crop size {crop_size}")

    return cropped_images

def get_border_array(array):
    # Выбор крайних элементов по горизонтали и вертикали
    top_row = array[0, :]      # верхняя строка
    bottom_row = array[-1, :]  # нижняя строка
    left_column = array[:, 0]  # левый столбец
    right_column = array[:, -1]  # правый столбец
    edge_values = np.concatenate([top_row, right_column, bottom_row, left_column])
    return edge_values

def count_vp(vertice_image):
    if vertice_image.shape[0] == 0 or vertice_image.shape[1] == 0:
        return 1

    min_ = np.min(vertice_image)
    max_ = np.max(vertice_image)
    if min_ == max_:
        return 1
    vertice_image = scale_array(vertice_image)
    border = get_border_array(vertice_image)
    cur = border[0]
    power = 0
    for i in range(1, len(border)):
        if border[i] != cur:
            if cur == 1:
                power += 1
            cur = border[i]
    return min(power, 4)


def count_vertices_powers(image, vertices):
    cropped_images = crop_images_around_centers(image, vertices, 160)
    powers = []
    for cropped_image in cropped_images:
        power = count_vp(cropped_image)
        powers.append(power)
    label = []
    for i in range(1, max(powers) + 1):
        label.append(powers.count(i))
    return label

def get_image_label_expert(image):
    skeleton, history = get_skeleton_pro(image)
    mask = history[2]
    vertices, history = get_vertices(image, mask)
    centers = get_vertices_centers(vertices)
    label = count_vertices_powers(skeleton, centers)
    return label

def get_image_label_beginner(image):
    skeleton = get_skeleton_begginer(image)
    vertices, _ = get_vertices(image, skeleton)
    centers = get_vertices_centers(vertices)
    label = count_vertices_powers(skeleton, centers)
    return label


class GraphSegmentationApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle('Graph Segmentation')
        self.setGeometry(100, 100, 800, 600)

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)

        self.layout = QVBoxLayout()
        self.central_widget.setLayout(self.layout)

        self.image_label = QLabel(self)
        self.layout.addWidget(self.image_label)

        self.open_button = QPushButton('Open Image', self)
        self.open_button.clicked.connect(self.open_image)
        self.layout.addWidget(self.open_button)

        self.segment_button = QPushButton('Segment', self)
        self.segment_button.clicked.connect(self.segment_image)
        self.layout.addWidget(self.segment_button)

        self.label = QLabel(self)
        self.layout.addWidget(self.label)

        # Добавляем поле выбора режима
        self.mode_selector = QComboBox(self)
        self.mode_selector.addItem('Beginner')
        self.mode_selector.addItem('Intermediate/Expert')
        self.mode_selector.currentIndexChanged.connect(self.change_mode)
        self.layout.addWidget(self.mode_selector)

        # Задаем начальный режим
        self.current_mode = 'Beginner'

    def change_mode(self, index):
        self.current_mode = self.mode_selector.itemText(index)
        print(f"Current mode: {self.current_mode}")
        # Здесь можно добавить дополнительную логику в зависимости от выбранного режима

    def open_image(self):
        file_dialog = QFileDialog(self)
        file_dialog.setNameFilter('Images (*.png *.jpg *.jpeg)')
        file_dialog.setViewMode(QFileDialog.Detail)
        file_dialog.setFileMode(QFileDialog.ExistingFile)
        file_dialog.setDirectory(os.path.dirname(os.path.realpath(__file__)))

        if file_dialog.exec_():
            file_path = file_dialog.selectedFiles()[0]
            self.image = cv2.imread(file_path)
            self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
            self.image = cv2.resize(self.image, (800, 600))
            self.pixmap = QPixmap(QImage(self.image, self.image.shape[1], self.image.shape[0], self.image.shape[1] * 3, QImage.Format_RGB888))
            self.image_label.setPixmap(self.pixmap)

    def draw_centers(self):
        if self.current_mode == 'Beginner':
            skeleton = get_skeleton_begginer(self.image)
            vertices, _ = get_vertices(self.image, skeleton)
            centers = get_vertices_centers(vertices)
            image = draw_vertices_centers(self.image.copy(), centers)
        else:
            skeleton, history = get_skeleton_pro(self.image)
            mask = history[2]
            vertices, history = get_vertices(self.image, mask)
            centers = get_vertices_centers(vertices)
            image = draw_vertices_centers(self.image.copy(), centers)
        self.pixmap = QPixmap(QImage(image, image.shape[1], image.shape[0], image.shape[1] * 3, QImage.Format_RGB888))
        self.image_label.setPixmap(self.pixmap)

    def classification(self):
        if self.current_mode == 'Beginner':
            label = get_image_label_beginner(self.image)
        else:
            label = get_image_label_expert(self.image)
        self.label.setText(f"Label: {label}")



plugin_path = QCoreApplication.libraryPaths()[0]

if __name__ == '__main__':
    print(plugin_path)
    os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = plugin_path
    os.environ['QT_PLUGIN_PATH'] = plugin_path
    print("Initilizing app...")
    app = QApplication(sys.argv)
    print("App initialized")
    window = GraphSegmentationApp()
    window.show()
    sys.exit(app.exec_())