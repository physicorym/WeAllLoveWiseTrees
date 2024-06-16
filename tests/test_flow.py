import pytest
import requests
import time

import tifffile as tiff


def load_multichannel_tiff_image(file_path):
    image = tiff.imread(file_path)
    return image


@pytest.mark.asyncio
async def test_flow_single_request(layout_name, crop_path):
    img = load_multichannel_tiff_image(crop_path)
    height, width, _ = img.shape
    payload = {'layout_name': layout_name, "crop_height": height, "crop_width": width}
    files = {'file': ('filename', img.tobytes())}
    r = requests.post('http://127.0.0.1:8000/', params=payload, files=files)
    task_id = r.json()
    time.sleep(10)
    r = requests.get('http://127.0.0.1:8000/', params=task_id, timeout=60)
    print("Result: ", r.json())
