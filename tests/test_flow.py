import pickle
import pytest
import requests

import tifffile as tiff


@pytest.mark.asyncio
async def test_flow_single_request(layout_name, crop_path):
    image = tiff.imread(crop_path)
    image_bytes = pickle.dumps(image)

    payload = {'layout_name': layout_name}
    files = {'file': ('filename', image_bytes)}
    r = requests.post('http://127.0.0.1:8000/', params=payload, files=files)
    task_id = r.json()
    r = requests.get('http://127.0.0.1:8000/', params=task_id, timeout=60)
    print("Result: ", r.json())
