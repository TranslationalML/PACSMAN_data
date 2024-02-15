"""Command-line interface for generating dummy 3D PACMAN image files."""

import argparse
import shutil
from pathlib import Path

import numpy as np
import pydicom
from loguru import logger
from nibabel.loadsave import save as nib_save
from nibabel.nifti1 import Nifti1Image
from nilearn.plotting import plot_img

from pacsman_data.generate_dummy_images import (
    create_pacsman_head_data,
    dicom_series_to_bids,
    nifti2dicom_1file,
)

OUTPUT_SUBDIRS = ["nifti", "png", "dicomseries", "bids"]


def parse_args() -> argparse.Namespace:
    """Parse arguments from the command line."""
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
        help="For overwriting the nifti, dicomseries, bids, and png subdirectories "
        "in the output directory if they already exist.",
    )
    parser.add_argument(
        "--save_nifti_png",
        action="store_true",
        help="Save a PNG image of the generated Nifti image",
    )
    args_ = parser.parse_args()
    args_.output_dir = Path(args_.output_dir).expanduser()
    return args_


def main() -> None:
    """Generate dummy 3D PACMAN image files."""
    args = parse_args()

    # Create 3D numpy array defining the PACSMAN head and eyes.
    img_data = create_pacsman_head_data(
        head_radius=args.head_radius,
        eye_radius=args.eye_radius,
        image_size=args.image_size,
    )

    # Create and save the PACSMAN image in Nifti format.
    # Nifti1Image's argument dataobj is not typed correctly in nibabel.
    # Hence, the ignore declaration.
    img = Nifti1Image(img_data, np.eye(4), dtype=img_data.dtype)  # type: ignore

    # Handle output directory.
    # If it already exists, remove all subdirectories if --force is set.
    # In this way, we keep the LICENSE file in the output directory.
    # Otherwise, raise an error.
    if args.force:
        logger.warning(f"Running in force mode. Removing {args.output_dir}")
        shutil.rmtree(args.output_dir, ignore_errors=True)

    for sub_dir in OUTPUT_SUBDIRS:
        if (subdir_path := args.output_dir / sub_dir).exists():
            raise RuntimeError(
                "Some subdirectories in the output directory already exist. "
                "Please remove them or use the --force option."
            )
        subdir_path.mkdir(parents=True, exist_ok=True)

    # Create nifti output directory and save the Nifti image.
    output_nifti_dir = args.output_dir / "nifti"
    logger.info(f"Saving Nifti image in {output_nifti_dir}...")
    nifti_path = output_nifti_dir / "pacsman.nii.gz"
    nib_save(img, nifti_path)

    # Create png output directory and save a PNG image of the Nifti image
    # if --save_nifti_png is set.
    if args.save_nifti_png:
        output_png_dir = args.output_dir / "png"
        logger.info(f"Saving PNG image in {output_png_dir}")
        ratio = args.image_size / 128
        plot_img(
            img,
            cut_coords=(100 * ratio, 80 * ratio, 100 * ratio),
        ).savefig(output_png_dir / "pacsman.png")

    # Create dicomseries directory and generate the DICOM series
    # from the generated nifti image.
    output_dicom_dir = args.output_dir / "dicomseries"
    logger.info(f"Generating DICOM series in {output_dicom_dir}")
    nifti2dicom_1file(
        nifti_img_path=nifti_path,
        output_dir=output_dicom_dir,
    )

    # Print the DICOM header of the middle slice.
    slice_index = int(img_data.shape[2] / 2)
    ods = pydicom.dcmread(f"{output_dicom_dir}/slice{slice_index}.dcm")
    logger.info(f"DICOM header of the middle slice:\n{ods}")

    # Create BIDS output directory and generate the NIfTI/JSON pair of files
    # from the generated DICOM series.
    output_bids_dir = args.output_dir / "bids"
    logger.info(
        f"> Generating BIDS-compliant Nifti/JSON file pair in {output_bids_dir}"
    )
    dicom_series_to_bids(
        dicom_series_dir=output_dicom_dir,
        output_dir=output_bids_dir,
    )
    logger.success("Done!")
