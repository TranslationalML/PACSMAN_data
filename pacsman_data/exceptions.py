"""Exceptions submodule for the ``pacsman_data`` package."""


class DCM2NIIXError(Exception):
    """Exception raised when running subprocess ``dcm2niix``.

    This exception is raised when an error occurs during the
    conversion of DICOM files to NIfTI files using the
    ``dcm2niix`` command-line tool.
    """
