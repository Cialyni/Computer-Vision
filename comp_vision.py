import cv2
import numpy as np
import os
import warnings
warnings.filterwarnings('ignore')

class ComputerVisionProcessor:
    def __init__(self, image_path):
        self.image_path = image_path
        self.image = cv2.imread(image_path)
        if self.image is None:
            raise ValueError(f"Не удалось загрузить изображение: {image_path}")

        os.makedirs('results', exist_ok=True)
        print(f"Папка 'results' создана/открыта\n")
        
        self.markdown_lines = []
    
    def save_image(self, image, filename, title=None):
        if len(image.shape) == 3 and image.shape[2] == 3:
            save_img = image.copy()
        else:
            save_img = image
        
        filepath = f"results/{filename}"
        cv2.imwrite(filepath, save_img)
        print(f"Сохранено: {filename}")
        self.markdown_lines.append(f"![{title}]({filename})")
        
        return filepath
    
    def find_orb_features(self):
        print("1. Поиск ORB features")
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        
        orb = cv2.ORB_create(nfeatures=1000)
        keypoints, descriptors = orb.detectAndCompute(gray, None)
        
        image_with_keypoints = cv2.drawKeypoints(
            self.image, keypoints, None, 
            flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
        )
        
        self.markdown_lines.append("## 1. ORB Features")
        self.save_image(image_with_keypoints, "01_orb_features.jpg", "ORB Features")
        print(f"   Найдено ORB точек: {len(keypoints)}")
        
        return keypoints, descriptors
    
    def find_sift_features(self):
        print("2. Поиск SIFT features")
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        sift = cv2.SIFT_create()
        keypoints, descriptors = sift.detectAndCompute(gray, None)
        
        image_with_keypoints = cv2.drawKeypoints(
            self.image, keypoints, None, 
            flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
        )
        
        self.markdown_lines.append("## 2. SIFT Features")
        self.save_image(image_with_keypoints, "02_sift_features.jpg", "SIFT Features")
        print(f"   Найдено точек: {len(keypoints)}")
        
        return keypoints, descriptors
    
    def find_canny_edges(self):
        print("3. Детекция границ Canny")
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 100, 200)
        
        self.markdown_lines.append("## 3. Canny Edges")
        self.save_image(edges, "03_canny_edges.jpg", "Canny Edges")
        return edges
    
    def convert_to_grayscale(self):
        print("4. Преобразование в grayscale")
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        
        self.markdown_lines.append("## 4. Grayscale")
        self.save_image(gray, "04_grayscale.jpg", "Grayscale")
        return gray
    
    def convert_to_hsv(self):
        print("5. Преобразование в HSV")
        hsv = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)
        
        self.markdown_lines.append("## 5. HSV")
        self.save_image(hsv, "05_hsv.jpg", "HSV Image")
        
        return hsv
    
    def flip_right(self):
        print("6. Отражение по правой границе")
        flipped = cv2.flip(self.image, 1)
        
        self.markdown_lines.append("## 6. Отражение по правой границе")
        self.save_image(flipped, "06_flipped_right.jpg", "Flipped Right")
        return flipped
    
    def flip_bottom(self):
        print("7. Отражение по нижней границе")
        flipped = cv2.flip(self.image, 0)
        
        self.markdown_lines.append("## 7. Отражение по нижней границе")
        self.save_image(flipped, "07_flipped_bottom.jpg", "Flipped Bottom")
        return flipped
    
    def rotate_45_degrees(self):
        print("8. Поворот на 45 градусов")
        (h, w) = self.image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, 45, 1.0)
        rotated = cv2.warpAffine(self.image, M, (w, h))
        
        self.markdown_lines.append("## 8. Поворот на 45°")
        self.save_image(rotated, "08_rotated_45.jpg", "Rotated 45°")
        return rotated
    
    def rotate_30_degrees_around_point(self, point):
        print(f"9. Поворот на 30° вокруг точки {point}")
        (h, w) = self.image.shape[:2]
        M = cv2.getRotationMatrix2D(point, 30, 1.0)
        rotated = cv2.warpAffine(self.image, M, (w, h))
        
        self.markdown_lines.append(f"## 9. Поворот на 30° вокруг точки {point}")
        self.save_image(rotated, "09_rotated_30_point.jpg", f"Rotated 30° around {point}")
        return rotated
    
    def shift_right_10px(self):
        print("10. Сдвиг на 10px вправо")
        (h, w) = self.image.shape[:2]
        M = np.float32([[1, 0, 10], [0, 1, 0]])
        shifted = cv2.warpAffine(self.image, M, (w, h))
        
        self.markdown_lines.append("## 10. Сдвиг на 10px вправо")
        self.save_image(shifted, "10_shifted_right.jpg", "Shifted 10px Right")
        
        return shifted
    
    def adjust_brightness(self):
        print("11. Изменение яркости")
        value = 50
        
        brightened = cv2.convertScaleAbs(self.image, alpha=1, beta=value)
        
        self.markdown_lines.append("## 11. Изменение яркости")
        self.save_image(brightened, "11_brightened.jpg", f"Brightened +{value}")
        
        return brightened
    
    def adjust_contrast(self):
        print("12. Изменение контраста")
        alpha = 1.5
        
        high_contrast = cv2.convertScaleAbs(self.image, alpha=alpha, beta=0)
        
        self.markdown_lines.append("## 12. Изменение контраста")
        self.save_image(high_contrast, "12_high_contrast.jpg", f"High Contrast (α={alpha})")
        
        return high_contrast
    
    def gamma_correction(self):
        print("13. Гамма-преобразование")
        gamma = 2.2
        
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255 
                         for i in np.arange(0, 256)]).astype("uint8")
        
        gamma_corrected = cv2.LUT(self.image, table)
        
        self.markdown_lines.append("## 13. Гамма-преобразование")
        self.save_image(gamma_corrected, "13_gamma_corrected.jpg", f"Gamma Correction (γ={gamma})")
        
        return gamma_corrected
    
    def histogram_equalization(self):
        print("14. Гистограмная эквализация")
        
        lab = cv2.cvtColor(self.image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l_eq = cv2.equalizeHist(l)
        lab_eq = cv2.merge((l_eq, a, b))
        equalized = cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)
        
        self.markdown_lines.append("## 14. Гистограмная эквализация")
        self.save_image(equalized, "14_color_equalized.jpg", "Color Histogram Equalization")
        
        return equalized
    
    def warm_white_balance(self):
        print("15. Теплый баланс белого")
        
        warm = self.image.copy().astype(np.float32)
        warm[:, :, 2] = np.clip(warm[:, :, 2] * 1.2, 0, 255)
        warm[:, :, 1] = np.clip(warm[:, :, 1] * 1.1, 0, 255)
        warm[:, :, 0] = np.clip(warm[:, :, 0] * 0.9, 0, 255)
        
        warm = warm.astype(np.uint8)
        
        self.markdown_lines.append("## 15. Теплый баланс белого")
        self.save_image(warm, "15_warm_white_balance.jpg", "Warm White Balance")
        
        return warm
    
    def cool_white_balance(self):
        print("16. Холодный баланс белого")
        
        cool = self.image.copy().astype(np.float32)
        cool[:, :, 0] = np.clip(cool[:, :, 0] * 1.2, 0, 255)
        cool[:, :, 1] = np.clip(cool[:, :, 1] * 1.1, 0, 255)
        cool[:, :, 2] = np.clip(cool[:, :, 2] * 0.9, 0, 255)
        
        cool = cool.astype(np.uint8)
        
        self.markdown_lines.append("## 16. Холодный баланс белого")
        self.save_image(cool, "16_cool_white_balance.jpg", "Cool White Balance")
        
        return cool
    
    def apply_color_map(self):
        print("17. Изменение цветовой палитры")
        
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        
        color_mapped = cv2.applyColorMap(gray, cv2.COLORMAP_JET)
        
        self.markdown_lines.append("## 17. Изменение цветовой палитры: JET")
        self.save_image(color_mapped, "17_colormap_jet.jpg", "Color Map: Jet")
        
        return color_mapped
    
    def binarize_image(self):
        print("18. Бинаризация")
        
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        
        _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        
        self.markdown_lines.append("## 18. Бинаризация")
        self.save_image(binary, "18_binary_threshold.jpg", "Binary Threshold")
        
        return binary
    
    def find_contours_binary(self):
        print("19. Поиск контуров на бинаризированном изображении")
        
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        
        contours, hierarchy = cv2.findContours(
            binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
        )
        
        contours_image = self.image.copy()
        cv2.drawContours(contours_image, contours, -1, (0, 255, 0), 2)
        
        self.markdown_lines.append("## 19. Контуры на бинаризированном изображении")
        self.save_image(contours_image, "19_contours_on_image.jpg", f"Contours ({len(contours)} found)")
        
        print(f"   Найдено контуров: {len(contours)}")
        return contours, hierarchy
    
    def find_contours_with_filters(self):
        print("20. Поиск контуров с фильтрами")
        
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        sobel_combined = cv2.magnitude(sobelx, sobely)
        sobel_combined = np.uint8(np.clip(sobel_combined, 0, 255))
        
        _, sobel_binary = cv2.threshold(sobel_combined, 30, 255, cv2.THRESH_BINARY)
        contours_sobel, _ = cv2.findContours(
            sobel_binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
        )
        
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        laplacian = np.uint8(np.clip(np.abs(laplacian), 0, 255))
        
        _, laplacian_binary = cv2.threshold(laplacian, 30, 255, cv2.THRESH_BINARY)
        contours_laplacian, _ = cv2.findContours(
            laplacian_binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
        )
        
        sobel_contours = self.image.copy()
        cv2.drawContours(sobel_contours, contours_sobel, -1, (0, 255, 0), 2)
        
        laplacian_contours = self.image.copy()
        cv2.drawContours(laplacian_contours, contours_laplacian, -1, (0, 255, 0), 2)
        
        self.markdown_lines.append("## 20. Контуры с фильтрами")
        self.save_image(sobel_contours, "20_sobel_contours.jpg", f"Sobel Contours ({len(contours_sobel)})")
        self.save_image(laplacian_contours, "20_laplacian_contours.jpg", f"Laplacian Contours ({len(contours_laplacian)})")
        
        print(f"   Контуров с Собелем: {len(contours_sobel)}")
        print(f"   Контуров с Лапласианом: {len(contours_laplacian)}")
        
        return sobel_contours, laplacian_contours
    
    def blur_image(self):
        print("21. Размытие изображения")
        
        gaussian_blur = cv2.GaussianBlur(self.image, (15, 15), 0)
        
        self.markdown_lines.append("## 21. Размытие изображения: Гауссово")
        self.save_image(gaussian_blur, "21_gaussian_blur.jpg", "Gaussian Blur (15x15)")
        
        return gaussian_blur
    
    def fourier_filter_high_freq(self):
        print("22. Фурье фильтрация: высокие частоты")
        
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        
        f = np.fft.fft2(gray)
        fshift = np.fft.fftshift(f)
        
        rows, cols = gray.shape
        crow, ccol = rows // 2, cols // 2
        
        mask = np.ones((rows, cols), np.uint8)
        r = 30
        mask[crow-r:crow+r, ccol-r:ccol+r] = 0
        
        fshift_filtered = fshift * mask
        f_ishift = np.fft.ifftshift(fshift_filtered)
        img_back = np.fft.ifft2(f_ishift)
        img_back = np.abs(img_back)
        
        high_freq = np.uint8(np.clip(img_back, 0, 255))
        
        self.markdown_lines.append("## 22. Фурье фильтрация: высокие частоты")
        self.save_image(high_freq, "22_high_frequency.jpg", "High Frequency (Fourier)")
        
        return high_freq
    
    def fourier_filter_low_freq(self):
        print("23. Фурье фильтрация: низкие частоты")
        
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        
        f = np.fft.fft2(gray)
        fshift = np.fft.fftshift(f)
        
        rows, cols = gray.shape
        crow, ccol = rows // 2, cols // 2
        
        mask = np.zeros((rows, cols), np.uint8)
        r = 30
        mask[crow-r:crow+r, ccol-r:ccol+r] = 1
        
        fshift_filtered = fshift * mask
        f_ishift = np.fft.ifftshift(fshift_filtered)
        img_back = np.fft.ifft2(f_ishift)
        img_back = np.abs(img_back)
        
        low_freq = np.uint8(np.clip(img_back, 0, 255))
        
        self.markdown_lines.append("## 23. Фурье фильтрация (низкие частоты)")
        self.save_image(low_freq, "23_low_frequency.jpg", "Low Frequency (Fourier)")
        
        return low_freq
    
    def apply_erosion(self):
        print("24. Операция эрозии")
        
        kernel = np.ones((5, 5), np.uint8)
        eroded = cv2.erode(self.image, kernel, iterations=1)
        
        self.markdown_lines.append("## 24. Операция эрозии")
        self.save_image(eroded, "24_erosion.jpg", "Erosion")
        
        return eroded
    
    def apply_dilation(self):
        print("25. Операция дилатации")
        
        kernel = np.ones((5, 5), np.uint8)
        dilated = cv2.dilate(self.image, kernel, iterations=1)
        
        self.markdown_lines.append("## 25. Операция дилатации")
        self.save_image(dilated, "25_dilation.jpg", "Dilation")
        
        return dilated
    
    def run_all_operations(self, rotation_point=(100, 100)):
        self.markdown_lines = [
            "# Отчет по обработке изображений",
            "",
            f"**Исходное изображение:** {self.image_path}",
            f"**Размер:** {self.image.shape[1]}×{self.image.shape[0]} пикселей",
            "",
            "---",
            ""
        ]
        
        self.save_image(self.image, "00_original.jpg", "Original Image")
        
        operations = [
            ("ORB Features", self.find_orb_features),
            ("SIFT Features", self.find_sift_features),
            ("Canny Edges", self.find_canny_edges),
            ("Grayscale", self.convert_to_grayscale),
            ("HSV", self.convert_to_hsv),
            ("Flip Right", self.flip_right),
            ("Flip Bottom", self.flip_bottom),
            ("Rotate 45°", self.rotate_45_degrees),
            ("Rotate 30° around point", lambda: self.rotate_30_degrees_around_point(rotation_point)),
            ("Shift Right 10px", self.shift_right_10px),
            ("Adjust Brightness", self.adjust_brightness),
            ("Adjust Contrast", self.adjust_contrast),
            ("Gamma Correction", self.gamma_correction),
            ("Histogram Equalization", self.histogram_equalization),
            ("Warm White Balance", self.warm_white_balance),
            ("Cool White Balance", self.cool_white_balance),
            ("Color Map", self.apply_color_map),
            ("Binarization", self.binarize_image),
            ("Contours on Binary", self.find_contours_binary),
            ("Contours with Filters", self.find_contours_with_filters),
            ("Blur Image", self.blur_image),
            ("Fourier High Freq", self.fourier_filter_high_freq),
            ("Fourier Low Freq", self.fourier_filter_low_freq),
            ("Erosion", self.apply_erosion),
            ("Dilation", self.apply_dilation),
        ]
        
        results = {}
        for name, func in operations:
            try:
                results[name] = func()
            except Exception as e:
                print(f"Ошибка в {name}: {e}")
                results[name] = None
        
        self.create_markdown_report()
        
        return results
    
    def create_markdown_report(self):
        with open("results/report.md", "w", encoding="utf-8") as f:
            f.write("\n".join(self.markdown_lines))
        

def main():
    image_path = input("Введите путь к изображению (или нажмите Enter для тестового): ").strip()
    
    if not image_path:
        print("Создание тестового изображения")
        test_image = np.zeros((400, 600, 3), dtype=np.uint8)
        
        cv2.rectangle(test_image, (50, 50), (200, 200), (0, 255, 0), -1)
        cv2.circle(test_image, (400, 200), 80, (255, 0, 0), -1)
        cv2.line(test_image, (100, 300), (500, 300), (0, 0, 255), 5)
        cv2.putText(test_image, 'Computer Vision Test', (150, 100), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        image_path = "test_image.jpg"
        cv2.imwrite(image_path, test_image)
        print(f"Создано тестовое изображение: {image_path}\n")
    
    try:
        processor = ComputerVisionProcessor(image_path)
        results = processor.run_all_operations(rotation_point=(150, 150))
    
        result_files = os.listdir('results')
        image_files = [f for f in result_files if f.endswith(('.jpg', '.png', '.jpeg'))]
        
        print(f"Всего файлов в папке 'results/': {len(result_files)}")
        print(f"Изображений: {len(image_files)}")
        print(f"Markdown отчет: report.md")
        
        print("\nСозданные файлы:")
        for i, file in enumerate(sorted(image_files), 1):
            print(f"  {i:2d}. {file}")
        
        print(f"\nОтчет: results/report.md")
        
    except Exception as e:
        print(f"\nОшибка: {e}")
        print("\nУбедитесь, что:")
        print("1. Указан правильный путь к изображению")
        print("2. Установлены библиотеки: pip install opencv-python numpy")
        if "SIFT" in str(e):
            print("3. Для SIFT требуется: pip install opencv-contrib-python")

if __name__ == "__main__":
    main()