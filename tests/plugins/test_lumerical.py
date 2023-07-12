from pathlib import Path

import pytest

from simphony.plugins.lumerical import load_sparams


@pytest.fixture
def sparam_file_dir(data_dir) -> Path:
    return data_dir / "lum_sparam_files"


def test_load_sparams1(sparam_file_dir):
    print((sparam_file_dir / "Ybranch_Thickness =220 width=500.sparam").exists())
    load_sparams(sparam_file_dir / "Ybranch_Thickness =220 width=500.sparam")


def test_load_sparams2(sparam_file_dir):
    load_sparams(
        sparam_file_dir
        / "te_ebeam_dc_halfring_straight_gap=170nm_radius=10um_width=480nm_thickness=230nm_CoupleLength=0um.dat"
    )
