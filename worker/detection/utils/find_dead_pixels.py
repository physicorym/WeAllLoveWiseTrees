import numpy as np
import pandas as pd
import tifffile as tiff

from image import load_multichannel_tiff_image

def detect_and_fix_dead_pixels(channel, channel_number, ratio_threshold=5, percentage_threshold=0.2):
    """
    Обнаруживает и исправляет мертвые пиксели в канале изображения.
    Args:
        channel (numpy array): Канал изображения для обработки.
        channel_number (int): Номер канала для отчета.
        ratio_threshold (float): Порог отношения для обнаружения мертвых пикселей.
        percentage_threshold (float): Порог отношения для исправления мертвых пикселей.
    Returns:
        tuple: Канал с исправленными мертвыми пикселями и отчет.
    """
    fixed_channel = channel.copy()
    rows, cols = channel.shape
    report = []

    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            region = channel[i-1:i+2, j-1:j+2]
            mean_value = (np.sum(region) - channel[i, j]) / 8.0
            central_pixel = channel[i, j]

            if central_pixel > mean_value * ratio_threshold:
                fixed_channel[i, j] = mean_value
                report.append([i, j, channel_number, central_pixel, mean_value])

            elif central_pixel < mean_value * percentage_threshold:
                fixed_channel[i, j] = mean_value
                report.append([i, j, channel_number, central_pixel, mean_value])

    return fixed_channel, report

def detect_and_fix_border_dead_pixels(channel, channel_number, ratio_threshold=5, percentage_threshold=0.2):
    """
    Обнаруживает и исправляет мертвые пиксели на границах канала изображения.
    Args:
        channel (numpy array): Канал изображения для обработки.
        channel_number (int): Номер канала для отчета.
        ratio_threshold (float): Порог отношения для обнаружения мертвых пикселей.
        percentage_threshold (float): Порог отношения для исправления мертвых пикселей.
    Returns:
        tuple: Канал с исправленными мертвыми пикселями и отчет.
    """
    rows, cols = channel.shape
    report = []

    for j in range(cols):
        for i in [0, rows - 1]:
            region = channel[max(0, i-1):min(rows, i+3), max(0, j-1):min(cols, j+3)]
            mean_value = np.mean(region)
            central_pixel = channel[i, j]

            if central_pixel > mean_value * ratio_threshold:
                channel[i, j] = mean_value
                report.append([i, j, channel_number, central_pixel, mean_value])

            elif central_pixel < mean_value * percentage_threshold:
                channel[i, j] = mean_value
                report.append([i, j, channel_number, central_pixel, mean_value])

    for i in range(rows):
        for j in [0, cols - 1]:
            region = channel[max(0, i-1):min(rows, i+3), max(0, j-1):min(cols, j+3)]
            mean_value = np.mean(region)
            central_pixel = channel[i, j]

            if central_pixel > mean_value * ratio_threshold:
                channel[i, j] = mean_value
                report.append([i, j, channel_number, central_pixel, mean_value])

            elif central_pixel < mean_value * percentage_threshold:
                channel[i, j] = mean_value
                report.append([i, j, channel_number, central_pixel, mean_value])

    return channel, report

def apply_custom_padding(channel):
    """
    Применяет паддинг к изображению, используя среднее значение 5 ближайших пикселей.
    """
    rows, cols = channel.shape
    padded_channel = np.zeros((rows + 2, cols + 2), dtype=channel.dtype)
    padded_channel[1:-1, 1:-1] = channel

    for j in range(cols):
        padded_channel[0, j + 1] = np.mean(channel[0:4, j])
        padded_channel[-1, j + 1] = np.mean(channel[-4:, j])

    for i in range(rows):
        padded_channel[i + 1, 0] = np.mean(channel[i, 0:4])
        padded_channel[i + 1, -1] = np.mean(channel[i, -4:])

    padded_channel[0, 0] = np.mean(channel[0:4, 0:4])
    padded_channel[0, -1] = np.mean(channel[0:4, -4:])
    padded_channel[-1, 0] = np.mean(channel[-4:, 0:4])
    padded_channel[-1, -1] = np.mean(channel[-4:, -4:])

    return padded_channel

def save_report_to_csv(report_data, file_path):
    """Сохраняет отчет о исправленных пикселях в файл CSV."""
    df = pd.DataFrame(report_data, columns=[
        'Номер строки',
        'Номер столбца',
        'Номер канала',
        'Битое значение',
        'Исправленное значение'
    ])
    df.to_csv(file_path, index=False)

def process_image(file_path):
    """Обрабатывает изображение и создает отчет об исправленных мертвых пикселях."""
    image_array = load_multichannel_tiff_image(file_path)
    report_data = []

    if image_array.shape[-1] == 4:
        for i in range(4):
            channel = image_array[:, :, i]
            padded_channel = apply_custom_padding(channel)
            fixed_padded_channel, channel_report = detect_and_fix_dead_pixels(padded_channel, i + 1)
            fixed_channel = fixed_padded_channel[1:-1, 1:-1]
            fixed_channel, border_report = detect_and_fix_border_dead_pixels(fixed_channel, i + 1)
            report_data.extend(channel_report)
            report_data.extend(border_report)

        fixed_image_array = np.stack(
            [fixed_channel for _, fixed_channel, _ in [(i, image_array[:, :, i], i) for i in range(4)]], axis=-1
        )
        tiff.imwrite(file_path.replace('.tif', '_fixed.tif'), fixed_image_array.astype(np.uint16))

        report_path = file_path.replace('.tif', '_report2.csv')
        save_report_to_csv(report_data, report_path)

        df = pd.DataFrame(report_data, columns=[
            'Номер строки',
            'Номер столбца',
            'Номер канала',
            'Битое значение',
            'Исправленное значение'
        ])
        return df

multichannel_tiff_image_path = '/Users/user/virtualenv/LCT_2024_18/data/Sitronics/1_20/crop_0_0_0000.tif'
df_report = process_image(multichannel_tiff_image_path)
print(df_report)