import numpy as np
import pandas as pd

def detect_and_fix_dead_pixels(channel, channel_number, ratio_threshold=5, percentage_threshold=0.15):
    """Детектирование "битых" пикселей и замени их значений по периметру (rows-1), (columns-1) кропа."""
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

def detect_and_fix_border_dead_pixels(channel, channel_number, ratio_threshold=5, percentage_threshold=0.15):
    """Детектирование "битых" пикселей и замени их значений по внешней границе кропа."""
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
    """Введение кастомного паддинга для последующей обработки границы пикселей кропа."""
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
    """Сохранение отчета о "битых" пикселях в виде таблицы."""
    df = pd.DataFrame(report_data, columns=[
        'номер строки',
        'номер столбца',
        'номер канала',
        '«битое» значение',
        'исправленное значение'
    ])
    df.to_csv(file_path, index=False)

def process_and_display_image(image_array, ratio_threshold=5, percentage_threshold=0.15):
    """Обработка кропа с "битыми" пикселями и возвращение "исправленного" кропа с информацией о исправлениях."""

    report_data = []
    image_array = image_array.transpose((0, 2, 1))

    if image_array.shape[-1] == 4:
        channels = ['Red', 'Green', 'Blue', 'NIR']
        cmap_list = ['Reds', 'Greens', 'Blues', 'gray']
        processed_channels = []

        for i, channel_name in enumerate(channels):
            channel = image_array[:, :, i]
            padded_channel = apply_custom_padding(channel)
            (fixed_padded_channel,
             channel_report) = detect_and_fix_dead_pixels(padded_channel, i + 1, ratio_threshold,
                                                          percentage_threshold)
            fixed_channel = fixed_padded_channel[1:-1, 1:-1]
            (fixed_channel,
             border_report) = detect_and_fix_border_dead_pixels(fixed_channel, i + 1, ratio_threshold,
                                                                percentage_threshold)
            report_data.extend(channel_report)
            report_data.extend(border_report)
            processed_channels.append((channel, fixed_channel, channel_name, cmap_list[i]))

        fixed_image_array = np.stack([fixed_channel for _, fixed_channel, _, _ in processed_channels], axis=-1)
        df = pd.DataFrame(report_data, columns=[
            'номер строки',
            'номер столбца',
            'номер канала',
            '«битое» значение',
            'исправленное значение'
        ])
        return df, fixed_image_array
