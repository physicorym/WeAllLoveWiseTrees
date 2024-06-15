import pytest


@pytest.fixture
def layout_name():
    return "layout_2021-06-15.tif"


@pytest.fixture
def crop_path():
    return "./layouts/crop_0_0_0000.tif"
