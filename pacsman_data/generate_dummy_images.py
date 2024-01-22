#!/usr/bin/env python
# coding: utf-8

# Copy

"""Script to generate dummy 3D PACMAN image files in Nifti and DICOM formats for testing purposes.

The generated image files are originally aimed to be used for testing PACSMAN commands,
but they can be used for testing any other DICOM / Nifti related tools.

"""

import os
import argparse
import shutil
import numpy as np
import nilearn.plotting
import nibabel as nib
import pydicom
from pprint import pprint


def create_pacsman_head_data(image_size=128, head_radius=60, eye_radius=10):
    """Create a labeled 3D numpy array defining a PACSMAN head.

    Voxels belonging to PACSMAN head are set to 2 and voxels belonging to the eyes are set to 1.

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


def convertNsave(
    slice_arr,
    output_dir,
    series_instance_uid,
    study_instance_uid,
    frame_of_reference_uid,
    implementation_class_uid,
    implementation_version_name,
    index=0,
):
    """Save the index-th slice of a 3D image (last dimension) in a DICOM file.

    Args:
        slice_arr: 2D array (3D image slice)
        output_dir: output directory to store the dicom file
        series_instance_uid: Series Instance UID
        study_instance_uid: Study Instance UID
        frame_of_reference_uid: Frame Of Reference UID
        index: index of the slice in the 3D image (last dimension)
    """
    # Initialize DICOM dataset.
    ds = pydicom.Dataset()

    # Populate required values for file meta information.
    sop_instance_uid = pydicom.uid.generate_uid()

    meta = pydicom.Dataset()
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
    ds.ImagePositionPatient = f"{-int(slice_arr.shape[0]/2)+1}\\0\\{index}"
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
        os.path.join(output_dir, f"slice{index}.dcm"),
        write_like_original=False  # To prevent to have the file not appearing to be DICOM error.
        # https://pydicom.github.io/pydicom/dev/reference/generated/pydicom.errors.InvalidDicomError.html
    )


def nifti2dicom_1file(nifti_path, output_dir):
    """Convert a nifti file into dicom series.

    Args:
        nifti_path (str): the path to a nifti file
        output_dir (str): the path to output the DICOM files
    """

    # Generate unique Series / Study Instance UID and  Frame Of Reference UID.
    series_instance_iud = pydicom.uid.generate_uid()
    study_instance_uid = pydicom.uid.generate_uid()
    frame_of_reference_uid = pydicom.uid.generate_uid()
    implementation_class_uid = pydicom.uid.generate_uid()
    implementation_version_name = f"pydicom {pydicom.__version__}"

    nifti_file = nib.load(nifti_path)
    nifti_array = nifti_file.get_fdata()
    number_slices = nifti_array.shape[2]

    for slice_ in range(number_slices):
        convertNsave(
            slice_arr=nifti_array[:, :, slice_],
            output_dir=output_dir,
            series_instance_uid=series_instance_iud,
            study_instance_uid=study_instance_uid,
            frame_of_reference_uid=frame_of_reference_uid,
            implementation_class_uid=implementation_class_uid,
            implementation_version_name=implementation_version_name,
            index=slice_,
        )


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--head_radius",
        type=int,
        default=60,
        help="Radius of the PACSMAN head",
    )
    parser.add_argument(
        "--eye_radius",
        type=int,
        default=10,
        help="Radius of the PACSMAN eyes",
    )
    parser.add_argument(
        "--image_size",
        type=int,
        default=128,
        help="Size of the generated image",
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        type=str,
        required=True,
        help="Directory to store the generated DICOM files",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="For overwriting the nifti, dicomseries, and png subdirectories "
        "in the output directory if they already exist.",
    )
    parser.add_argument(
        "--save_nifti_png",
        action="store_true",
        help="Save a PNG image of the generated Nifti image",
    )
    return parser


def main():
    # Create parser and parse arguments.
    parser = get_parser()
    args = parser.parse_args()

    # Create 3D numpy array defining the PACSMAN head and eyes.
    img_data = create_pacsman_head_data(
        head_radius=args.head_radius,
        eye_radius=args.eye_radius,
        image_size=args.image_size,
    )

    # Create and save the PACSMAN image in Nifti format.
    img = nib.Nifti1Image(img_data, np.eye(4))

    # Handle output directory.
    # If it already exists, remove all subdirectories if --force is set.
    # In this way, we keep the LICENSE file in the output directory.
    # Otherwise, raise an error.
    if os.path.exists(args.output_dir):
        if args.force:
            for sub_dir in ["nifti", "png", "dicomseries"]:
                if os.path.exists(os.path.join(args.output_dir, sub_dir)):
                    shutil.rmtree(os.path.join(args.output_dir, sub_dir))
        else:
            raise ValueError(
                f"Subdirectories nifti/, png/, and dicomseries/ already exist in {args.output_dir}. "
                "Please remove them or use --force."
            )

    # Create nifti output directory and save the Nifti image.
    output_nifti_dir = os.path.join(args.output_dir, "nifti")
    os.makedirs(output_nifti_dir)
    nifti_path = os.path.join(output_nifti_dir, "pacsman.nii.gz")
    nib.save(img, nifti_path)

    # Create png output directory and save a PNG image of the Nifti image
    # if --save_nifti_png is set.
    if args.save_nifti_png:
        output_png_dir = os.path.join(args.output_dir, "png")
        os.makedirs(output_png_dir)
        ratio = args.image_size / 128
        nilearn.plotting.plot_img(
            img, cut_coords=(100 * ratio, 80 * ratio, 100 * ratio)
        ).savefig(os.path.join(output_png_dir, "pacsman.png"))

    # Create dicomseries directory and generate the DICOM series
    # from the generated nifti image.
    output_dicom_dir = os.path.join(args.output_dir, "dicomseries")
    os.makedirs(output_dicom_dir)
    nifti2dicom_1file(
        nifti_path=nifti_path,
        output_dir=output_dicom_dir,
    )

    # Print the DICOM header of the middle slice.
    slice_index = int(img_data.shape[2] / 2)
    ods = pydicom.dcmread(f"{output_dicom_dir}/slice{slice_index}.dcm")
    pprint(ods)


if __name__ == "__main__":
    main()
