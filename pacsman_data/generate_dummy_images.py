"""Submodule 3D PACMAN image files in Nifti & DICOM formats for testing purposes.

The generated image files are originally aimed to be used for testing
PACSMAN commands, but they can be used for testing any other
DICOM / Nifti related tools.

"""

import subprocess
from pathlib import Path

import numpy as np
import pydicom
from loguru import logger
from nibabel.loadsave import (
    load as nib_load,
)
from numpy.typing import NDArray
from pydicom.dataset import FileMetaDataset
from pydicom.uid import generate_uid

from pacsman_data.exceptions import DCM2NIIXError


def create_pacsman_head_data(  # noqa: C901
    image_size: int = 128,
    head_radius: int = 60,
    eye_radius: int = 10,
) -> NDArray[np.float_]:
    """Create a labeled 3D numpy array defining a PACSMAN head.

    Voxels belonging to PACSMAN head are set to 2 and voxels belonging to the
    eyes are set to 1.

    Args:
        image_size: size of the output image in the three dimensions
        head_radius: radius of the PACSMAN head
        eye_radius: radius of the PACSMAN eyes

    Returns:
        img_data: 3D numpy array
    """
    # Check that the head radius is smaller than the image size divided by 2.
    if head_radius > image_size / 2:
        raise ValueError("Head radius should be smaller than image size divided by 2")

    # Define x,y,z coordinates of image voxels.
    # They are centered in 0 to facilitate definition of the
    # geometrical functions (head sphere, mouth planes) defining the PACSMAN head.
    x = np.arange(-int(image_size / 2) + 1, int(image_size / 2) + 1)
    y = np.arange(-int(image_size / 2) + 1, int(image_size / 2) + 1)
    z = np.arange(-int(image_size / 2) + 1, int(image_size / 2) + 1)

    # Initialize output array.
    img_data = np.zeros((len(x), len(y), len(z)))

    # Create the PACSMAN head.
    head_radius_sq = head_radius**2
    for i in x:
        for k in z:
            for j in y:
                if (i**2 + j**2 + k**2) <= head_radius_sq:  # Head sphere
                    if not (j + k > 0 and -j + k > 0):  # Mouth planes
                        img_data[i - x.min(), j - y.min(), k - z.min()] = 2.0

    # Swap axis to have correspondences between the generate head and
    # the R, L, A, P, I, S anatomical landmarks.
    img_data = np.swapaxes(img_data, 2, 1)

    # Create PACSMAN eyes.
    eye_radius_sq = eye_radius**2

    eye1_center = image_size * np.array([0.171875, 0.625, 0.78125]) + np.array(
        [x.min(), y.min(), z.min()]
    )
    eye2_center = image_size * np.array([0.828125, 0.625, 0.78125]) + np.array(
        [x.min(), y.min(), z.min()]
    )

    for i in x:
        for k in z:
            for j in y:
                if (
                    (
                        (i - eye1_center[0]) ** 2
                        + (j - eye1_center[1]) ** 2
                        + (k - eye1_center[2]) ** 2
                    )
                    <= eye_radius_sq
                ) or (
                    (
                        (i - eye2_center[0]) ** 2
                        + (j - eye2_center[1]) ** 2
                        + (k - eye2_center[2]) ** 2
                    )
                    <= eye_radius_sq
                ):
                    img_data[i - x.min(), j - y.min(), k - z.min()] = 1.0
    return img_data


def save_slice_as_dicom(  # noqa: PLR0915
    slice_arr: NDArray[np.float_],
    output_dir: Path,
    series_instance_uid: pydicom.uid.UID,
    study_instance_uid: pydicom.uid.UID,
    frame_of_reference_uid: pydicom.uid.UID,
    implementation_class_uid: pydicom.uid.UID,
    implementation_version_name: str,
    index: int = 0,
) -> None:
    """Save the index-th slice of a 3D image (last dimension) as a DICOM file.

    Args:
        slice_arr: 2D array (3D image slice)
        output_dir: output directory to store the dicom file
        series_instance_uid: Series Instance UID
        study_instance_uid: Study Instance UID
        frame_of_reference_uid: Frame Of Reference UID
        implementation_class_uid: Implementation Class UID
        implementation_version_name: Implementation Version Name
        index: index of the slice in the 3D image (last dimension)
    """
    # Initialize DICOM dataset.
    ds = pydicom.Dataset()

    # Populate required values for file meta information.
    sop_instance_uid = generate_uid()

    meta = FileMetaDataset()
    meta.MediaStorageSOPClassUID = pydicom.uid.MRImageStorage
    meta.MediaStorageSOPInstanceUID = sop_instance_uid
    meta.TransferSyntaxUID = pydicom.uid.ExplicitVRLittleEndian
    meta.ImplementationClassUID = implementation_class_uid
    meta.ImplementationVersionName = implementation_version_name

    ds.file_meta = meta

    # Populate required DICOM header fields.
    ds.is_little_endian = True
    ds.is_implicit_VR = False

    ds.SOPClassUID = pydicom.uid.MRImageStorage
    ds.SOPInstanceUID = sop_instance_uid
    ds.PatientName = "PACSMAN"
    ds.PatientID = "PACSMAN1"

    ds.Modality = "MR"
    ds.SeriesInstanceUID = series_instance_uid
    ds.StudyInstanceUID = study_instance_uid
    ds.FrameOfReferenceUID = frame_of_reference_uid

    ds.StudyDate = "20231016"
    ds.SeriesDescription = "pacsman_testing_dicom"

    ds.BitsStored = 16
    ds.BitsAllocated = 16
    ds.SamplesPerPixel = 1
    ds.HighBit = 15

    ds.Rows = slice_arr.shape[0]
    ds.Columns = slice_arr.shape[1]
    ds.InstanceNumber = index

    ds.PatientOrientation = r"L\R"
    ds.ImagePositionPatient = f"{-int(slice_arr.shape[0] / 2) + 1}\\0\\{index}"
    ds.ImageOrientationPatient = r"0\-1\0\1\0\0"
    ds.ImageType = r"ORIGINAL\PRIMARY\AXIAL"

    ds.RescaleIntercept = "0"
    ds.RescaleSlope = "1"
    ds.PixelSpacing = r"1\1"
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.PixelRepresentation = 1

    # Add PatientBirthDate for testing anonymization.
    ds.PatientBirthDate = "20231026"

    # Add Institution related tags for testing their anonymization.
    ds.InstitutionName = "CHUV TML"
    ds.InstitutionAddress = "Rue du Bugnon 46, 1011 Lausanne, Switzerland"

    # Add ReferencedPerformedProcedureStepSequence for testing anonymization.
    sub_ds = pydicom.Dataset()
    sub_ds.ReferencedSOPClassUID = pydicom.uid.MRImageStorage
    sub_ds.ReferencedSOPInstanceUID = sop_instance_uid
    ds.ReferencedPerformedProcedureStepSequence = pydicom.Sequence([sub_ds])

    # Add PatientID in SourcePatientGroupIdentificationSequence
    # to test anonymization of DICOM tags in nested DICOM tags, e.g. sequence.
    sub_ds = pydicom.Dataset()
    sub_ds.PatientID = "PACSMAN1"
    ds.SourcePatientGroupIdentificationSequence = pydicom.Sequence([sub_ds])

    # Validate the File Meta Information elements in ds.file_meta.
    pydicom.dataset.validate_file_meta(ds.file_meta, enforce_standard=True)

    # Make sure the array used uint16 data type and
    # assign the slice data.
    slice_arr = slice_arr.astype(np.uint16)
    ds.PixelData = slice_arr.tobytes()

    # Save the DICOM file.
    ds.save_as(
        output_dir / f"slice{index}.dcm",
        write_like_original=False,
        # To prevent to have the file not appearing to be DICOM error.
        # https://pydicom.github.io/pydicom/dev/reference/generated/pydicom.errors.InvalidDicomError.html
    )


def nifti2dicom_1file(
    nifti_img_path: Path,
    output_dir: Path,
) -> None:
    """Convert a nifti file into dicom series.

    Args:
        nifti_img_path (str): the path to a nifti file
        output_dir (str): the path to output the DICOM files
    """
    # Generate unique Series / Study Instance UID and  Frame Of Reference UID.
    series_instance_iud = generate_uid()
    study_instance_uid = generate_uid()
    frame_of_reference_uid = generate_uid()
    implementation_class_uid = generate_uid()
    implementation_version_name = f"pydicom {pydicom.__version__}"

    nifti_file = nib_load(nifti_img_path)
    nifti_array = nifti_file.get_fdata()  # type: ignore
    number_slices = nifti_array.shape[2]

    for slice_ in range(number_slices):
        save_slice_as_dicom(
            slice_arr=nifti_array[:, :, slice_],
            output_dir=output_dir,
            series_instance_uid=series_instance_iud,
            study_instance_uid=study_instance_uid,
            frame_of_reference_uid=frame_of_reference_uid,
            implementation_class_uid=implementation_class_uid,
            implementation_version_name=implementation_version_name,
            index=slice_,
        )


def dicom_series_to_bids(
    dicom_series_dir: Path,
    output_dir: Path,
) -> None:
    """Convert a DICOM series into a Nifti/JSON pair of files compliant to BIDS.

    It uses dcm2niix for the conversion. The participant label is set to 000001.
    This pair of files is saved in the specified `output_dir` and can be later used
    to create a dummy BIDS dataset for testing purposes.

    Args:
        dicom_series_dir: the path of the directory containing the DICOM series
        output_dir: the path to output the BIDS dataset
    """
    try:
        res = subprocess.run(
            [
                "dcm2niix",
                "-b",
                "y",
                "-z",
                "y",
                "-f",
                "sub-000001_T1w",
                "-o",
                output_dir,
                dicom_series_dir,
            ],
            check=True,
            capture_output=True,
        )
        logger.info(f"[dcm2niix] stdout:\n{res.stdout.decode('utf-8')}")
    except subprocess.CalledProcessError as e:
        logger.critical(e.output)
        raise DCM2NIIXError(
            f"Error during dcm2niix conversion: {e.returncode}\n{e.output}"
        )
