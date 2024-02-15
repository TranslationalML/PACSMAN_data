"""Unit testing for the CLI corresponding to the ``generate_dummy_images`` command."""

import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from pacsman_data.cli import main, parse_args

CMD_NAME = "generate_dummy_images"
HEAD_RADIUS = 100
EYE_RADIUS = 20
IMAGE_SIZE = 256
OUTPUT_DIR = "/tmp/output_dir"


def test_parse_args() -> None:
    """Test the ``parse_args`` function."""
    test_args = [
        "--head_radius",
        str(HEAD_RADIUS),
        "--eye_radius",
        str(EYE_RADIUS),
        "--image_size",
        str(IMAGE_SIZE),
        "-o",
        OUTPUT_DIR,
        "--force",
        "--save_nifti_png",
    ]

    with patch("sys.argv", [CMD_NAME] + test_args):
        args = parse_args()
        assert args.head_radius == HEAD_RADIUS
        assert args.eye_radius == EYE_RADIUS
        assert args.image_size == IMAGE_SIZE
        assert args.output_dir == Path(OUTPUT_DIR).expanduser()
        assert args.force
        assert args.save_nifti_png


def test_main_force_mode() -> None:
    """Test the CLI main function in force mode."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        test_args = ["-o", tmp_dir, "--force", "--save_nifti_png"]
        with patch("sys.argv", [CMD_NAME] + test_args):
            main()

        # Rerun on existing directory (should not raise an error)
        test_args.remove("--force")
        with patch("sys.argv", [CMD_NAME] + test_args), pytest.raises(RuntimeError):
            main()
