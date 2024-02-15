"""Unit testing for the ``pacsman_data`` repository."""

import tempfile
from pathlib import Path

import numpy as np
import pytest
from _pytest.tmpdir import TempPathFactory
from nibabel.loadsave import save as nii_save
from nibabel.nifti1 import Nifti1Image

from pacsman_data.exceptions import DCM2NIIXError
from pacsman_data.generate_dummy_images import (
    create_pacsman_head_data,
    dicom_series_to_bids,
    nifti2dicom_1file,
)


@pytest.fixture(name="nii_img_path", scope="session")
def nifti_image_generator(tmp_path_factory: TempPathFactory) -> Path:
    """Generates a synthetic NIfTI image for unit testing."""
    nii_img_ = Nifti1Image(
        dataobj=np.random.randint(  # type: ignore
            0,
            255,
            size=(128, 128, 128),
            dtype=np.uint8,
        ),
        affine=np.eye(4),
        dtype=np.uint8,
    )
    img_path_ = tmp_path_factory.mktemp("test_data") / "test.nii"
    nii_save(nii_img_, img_path_)
    return img_path_


def test_create_pacsman_head_data() -> None:
    """Tests``create_pacsman_head_data`` function."""
    # Default run (no errors should be raised)
    create_pacsman_head_data()

    # ValueError should be raised when head_radius > image_size
    with pytest.raises(ValueError):
        create_pacsman_head_data(
            head_radius=200,
            image_size=128,
        )


def test_dicom_series_to_bids(
    nii_img_path: Path,
) -> None:
    """Tests ``dicom_series_to_bids`` function."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        output_dir_ = Path(tmp_dir) / "out"
        output_dir_.mkdir()

        nifti2dicom_1file(
            nifti_img_path=nii_img_path,
            output_dir=output_dir_,
        )
        dicom_series_to_bids(
            dicom_series_dir=Path(tmp_dir),
            output_dir=output_dir_,
        )

        with pytest.raises(DCM2NIIXError):
            # Inexistent directory should make `dcm2niix` raise an error
            dicom_series_to_bids(
                dicom_series_dir=Path(tmp_dir),
                output_dir=Path("foo"),
            )
