from pathlib import Path

import pytest
import numpy as np


@pytest.fixture
def data_dir():
    return Path(__file__).parent / "data"


@pytest.fixture
def std_wl():
    return np.linspace(1.5, 1.6, 1000) * 1e-6
