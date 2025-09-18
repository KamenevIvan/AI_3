import os
import cv2
import numpy as np

def load_img(file_path):
    """
    Загружает изображение в формате BGR.

    Args:
        file_path (str): Путь к файлу изображения.

    Returns:
        numpy.ndarray: Изображение в формате BGR (3 канала), если загрузка успешна.
        None: Если файл не найден или не удалось загрузить изображение.

    Raises:
        ValueError: Если file_path не является строкой или пустой.
    """
    if not isinstance(file_path, str) or not file_path:
        raise ValueError("Ошибка: путь к файлу должен быть непустой строкой")
    
    if not os.path.exists(file_path):
        print(f"Ошибка: файл {file_path} не найден")
        return None
    
    image = cv2.imread(file_path, cv2.IMREAD_COLOR)
    if image is None:
        print(f"Ошибка: не удалось загрузить изображение {file_path}")
        return None
    
    return image

def brg_to_hsv(image):
    """
    Преобразует изображение из цветового пространства BGR в HSV.

    Args:
        image (numpy.ndarray): Входное изображение в формате BGR.

    Returns:
        numpy.ndarray: Изображение в формате HSV.

    Raises:
        ValueError: Если входное изображение None или имеет некорректный формат.
    """
    if image is None:
        raise ValueError("Ошибка: входное изображение не может быть None")
    
    if not isinstance(image, np.ndarray) or len(image.shape) != 3 or image.shape[2] != 3:
        raise ValueError("Ошибка: входное изображение должно быть в формате BGR (3 канала)")
    
    return cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

def create_mask(image_hsv, lower_color_green=np.array([35, 40, 40]),
                upper_color_green=np.array([85, 255, 255]), kernel_size=(5, 5)):
    """
    Создаёт бинарную маску для выделения объекта (объект = 255, фон = 0).

    Args:
        image_hsv (numpy.ndarray): Изображение в формате HSV.
        lower_color_green (numpy.ndarray): Нижняя граница диапазона цвета (HSV) для фона.
        upper_color_green (numpy.ndarray): Верхняя граница диапазона цвета (HSV) для фона.
        kernel_size (tuple): Размер ядра для морфологических операций (ширина, высота).

    Returns:
        numpy.ndarray: Бинарная маска (одноканальная, 255 для объекта, 0 для фона).

    Raises:
        ValueError: Если входное изображение None, имеет некорректный формат,
                    или параметры lower_color_green/upper_color_green некорректны.
        ValueError: Если kernel_size не является кортежем из двух положительных чисел.
    """
    if image_hsv is None:
        raise ValueError("Ошибка: входное изображение HSV не может быть None")
    
    if not isinstance(image_hsv, np.ndarray) or len(image_hsv.shape) != 3 or image_hsv.shape[2] != 3:
        raise ValueError("Ошибка: входное изображение должно быть в формате HSV (3 канала)")
    
    if not isinstance(lower_color_green, np.ndarray) or not isinstance(upper_color_green, np.ndarray):
        raise ValueError("Ошибка: lower_color_green и upper_color_green должны быть массивами numpy")
    
    if not isinstance(kernel_size, tuple) or len(kernel_size) != 2 or not all(isinstance(x, int) and x > 0 for x in kernel_size):
        raise ValueError("Ошибка: kernel_size должен быть кортежем из двух положительных чисел")

    mask = cv2.inRange(image_hsv, lower_color_green, upper_color_green)
    mask = cv2.bitwise_not(mask)  # Инвертируем: объект = 255, фон = 0

    kernel = np.ones(kernel_size, np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)  # Удаление шумов
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)  # Закрытие дыр

    return mask

def remove(file_input):
    """
    Удаляет фон изображения и возвращает результат с альфа-каналом (BGRA).

    Args:
        file_input (str): Путь к входному изображению.

    Returns:
        numpy.ndarray: Изображение в формате BGRA (4 канала), где альфа-канал определяет прозрачность.
        None: Если обработка не удалась.

    Raises:
        ValueError: Если file_input не является строкой или пустой.
    """
    if not isinstance(file_input, str) or not file_input:
        raise ValueError("Ошибка: путь к файлу должен быть непустой строкой")

    img = load_img(file_input)
    if img is None:
        return None

    try:
        img_hsv = brg_to_hsv(img)
        mask = create_mask(img_hsv)
    except ValueError as e:
        print(f"Ошибка при создании маски: {e}")
        return None

    # Создаём изображение с альфа-каналом
    try:
        b, g, r = cv2.split(img)
        alpha = mask  # Маска становится альфа-каналом
        result = cv2.merge([b, g, r, alpha])  # Изображение в формате BGRA
    except Exception as e:
        print(f"Ошибка при создании изображения с альфа-каналом: {e}")
        return None

    # Диагностика: сохраняем маску и результат
    try:
        cv2.imwrite("mask.jpg", mask)
        cv2.imwrite("result_remove.jpg", result[:, :, :3])  # Только BGR для просмотра
    except Exception as e:
        print(f"Ошибка при сохранении диагностических файлов: {e}")

    return result

def main(img_input, img_background, img_output, showRes=False):
    """
    Удаляет фон изображения, заменяет его новым и сохраняет результат.

    Args:
        img_input (str): Путь к входному изображению.
        img_background (str): Путь к фоновому изображению.
        img_output (str): Путь для сохранения результата.
        showRes (bool): Если True, отображает результат в окне.

    Returns:
        numpy.ndarray: Итоговое изображение в формате BGR, если обработка успешна.
        None: Если обработка не удалась.

    Raises:
        ValueError: Если входные пути не являются строками или пусты,
                    или showRes не является булевым значением.
    """
    # Проверка входных параметров
    if not isinstance(img_input, str) or not img_input:
        raise ValueError("Ошибка: путь к входному изображению должен быть непустой строкой")
    if not isinstance(img_background, str) or not img_background:
        raise ValueError("Ошибка: путь к фоновому изображению должен быть непустой строкой")
    if not isinstance(img_output, str) or not img_output:
        raise ValueError("Ошибка: путь для сохранения результата должен быть непустой строкой")
    if not isinstance(showRes, bool):
        raise ValueError("Ошибка: showRes должен быть булевым значением")

    # Удаляем фон
    result = remove(img_input)
    if result is None:
        print("Не удалось обработать изображение")
        return None

    # Загружаем фоновое изображение
    background = load_img(img_background)
    if background is None:
        return None

    # Проверка совместимости размеров
    try:
        background = cv2.resize(background, (result.shape[1], result.shape[0]))
    except Exception as e:
        print(f"Ошибка при изменении размера фонового изображения: {e}")
        return None

    # Извлекаем маску (альфа-канал) и передний план
    try:
        mask = result[:, :, 3]  # Альфа-канал
        foreground = result[:, :, :3]  # BGR-часть
    except Exception as e:
        print(f"Ошибка при извлечении маски или переднего плана: {e}")
        return None

    # Диагностика: проверяем, не пустая ли маска
    if np.sum(mask) == 0:
        print("Ошибка: маска полностью чёрная. Проверьте mask.jpg и result_remove.jpg.")
        return None

    # Нормализуем маску (0-1) как float
    try:
        mask_normalized = mask.astype(float) / 255.0
        # Создаём трёхканальную маску
        mask_3c = np.repeat(mask_normalized[:, :, np.newaxis], 3, axis=2)
    except Exception as e:
        print(f"Ошибка при нормализации маски: {e}")
        return None

    # Накладываем передний план на фон
    try:
        output = (background * (1 - mask_3c) + foreground * mask_3c).astype(np.uint8)
    except Exception as e:
        print(f"Ошибка при наложении изображений: {e}")
        return None

    # Сохраняем результат
    try:
        cv2.imwrite(img_output, output)
    except Exception as e:
        print(f"Ошибка при сохранении результата в {img_output}: {e}")
        return None

    if showRes:
        try:
            cv2.imshow("Result", output)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        except Exception as e:
            print(f"Ошибка при отображении результата: {e}")
            return None

    return output

if __name__ == "__main__":
    try:
        main("image1.jpg", "background1.jpg", "out.jpg", showRes=True)
    except ValueError as e:
        print(f"Ошибка: {e}")
    except Exception as e:
        print(f"Неизвестная ошибка: {e}")