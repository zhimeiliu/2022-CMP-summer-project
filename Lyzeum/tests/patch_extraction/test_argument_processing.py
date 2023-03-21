"""Tests for the module `lyzeum.patch_extraction.argument_processing`."""
import os
import stat
from pathlib import Path

import pytest

from numpy import nan

from lyzeum.patch_extraction import argument_processing as ap


executable_path = Path("is.txt").resolve()
non_executable_path = Path("is_not.txt").resolve()


@pytest.fixture
def create_executable_files_to_test():
    """Create and delete executable and non-executable files to test with."""
    executable_path.touch()
    non_executable_path.touch()

    stat_result = os.stat(str(executable_path))
    os.chmod(str(executable_path), stat_result.st_mode | stat.S_IEXEC)

    yield

    executable_path.unlink()
    non_executable_path.unlink()


@pytest.fixture
def create_dummy_file_and_dir():
    """Create and clean up dummy file and directory."""
    dummy_file = Path("file.txt")
    dummy_dir = Path("dir/")

    dummy_file.touch()
    dummy_dir.mkdir()

    yield

    dummy_file.unlink()
    dummy_dir.rmdir()


def test_process_patch_size_values():
    """Test the values `ap.process_patch_size` accepts."""
    ap.process_patch_size(2)
    ap.process_patch_size(256)
    ap.process_patch_size(512)

    with pytest.raises(ValueError):
        ap.process_patch_size(-1)
    with pytest.raises(ValueError):
        ap.process_patch_size(0)
    with pytest.raises(ValueError):
        ap.process_patch_size(1)


def test_process_patch_size_types():
    """Test the types `ap.process_patch_size` accepts."""
    ap.process_patch_size(10)

    with pytest.raises(TypeError):
        ap.process_patch_size(10.0)
    with pytest.raises(TypeError):
        ap.process_patch_size(10.0j)
    with pytest.raises(TypeError):
        ap.process_patch_size([10, 10, 10])


def test_process_stride_values():
    """Test the values `ap.process_stride` accepts."""
    # Integers greater than or equal to one are allowed.
    ap.process_stride(1)
    ap.process_stride(16)

    with pytest.raises(ValueError):
        ap.process_stride(0)
    with pytest.raises(ValueError):
        ap.process_stride(-1)


def test_process_stride_types():
    """Test the types `ap.process_stride` accepts."""
    # Integers are allowed
    ap.process_stride(1)
    ap.process_stride(16)

    with pytest.raises(TypeError):
        ap.process_stride(1.0)
    with pytest.raises(TypeError):
        ap.process_stride(1.0j)
    with pytest.raises(TypeError):
        ap.process_stride([1, 2, 3])


def test_process_mag_var_values():
    """Test the values `ap.process_mag_var` accepts."""
    # Should aaccept values greater than or equal to 1.0
    ap.process_mag_var(1.0)
    ap.process_mag_var(2.0)
    ap.process_mag_var(40.0)

    # Should reject anything less than 1.0
    with pytest.raises(ValueError):
        ap.process_mag_var(0.99)
    with pytest.raises(ValueError):
        ap.process_mag_var(0.0)
    with pytest.raises(ValueError):
        ap.process_mag_var(-1.0)


def test_process_mag_var_types():
    """Test the types `ap.process_mag_var` accepts."""
    # Should accept floats
    ap.process_mag_var(1.0)
    ap.process_mag_var(2.0)
    ap.process_mag_var(40.0)

    # Should reject non-floats
    with pytest.raises(TypeError):
        ap.process_mag_var(1)
    with pytest.raises(TypeError):
        ap.process_mag_var(2j)
    with pytest.raises(TypeError):
        ap.process_mag_var([1, 2, 3, 4])


def test_process_min_max_mags():
    """Test the function `ap.process_min_max_mags`."""
    # Should accept floats greater than 1.0
    ap.process_min_max_mags(1.0, 2.0)

    # Should reject magnifcation less than 1.0
    with pytest.raises(ValueError):
        ap.process_min_max_mags(1.0, 0.99)
    with pytest.raises(ValueError):
        ap.process_min_max_mags(0.99, 1.0)
    with pytest.raises(ValueError):
        ap.process_min_max_mags(0.0, 1.0)
    with pytest.raises(ValueError):
        ap.process_min_max_mags(1.0, 0.0)
    with pytest.raises(ValueError):
        ap.process_min_max_mags(-1.0, 1.0)
    with pytest.raises(ValueError):
        ap.process_min_max_mags(1.0, -1.0)

    # Should reject min_mag > max_mag
    with pytest.raises(ValueError):
        ap.process_min_max_mags(2.0, 1.0)


def test_process_background_tol():
    """Test the function `ap.process_background_tol`."""
    # Should accept floats on (0, 1)
    ap.process_background_tol(0.0)
    ap.process_background_tol(0.5)
    ap.process_background_tol(1.0)

    # Should reject values outwith (0, 1)
    with pytest.raises(ValueError):
        ap.process_background_tol(-1e-3)
    with pytest.raises(ValueError):
        ap.process_background_tol(-1.0)
    with pytest.raises(ValueError):
        ap.process_background_tol(1.0 + 1e-3)
    with pytest.raises(ValueError):
        ap.process_background_tol(2.0)

    # Should reject non-floats
    with pytest.raises(TypeError):
        ap.process_background_tol(1)
    with pytest.raises(TypeError):
        ap.process_background_tol(2)
    with pytest.raises(TypeError):
        ap.process_background_tol(0.5j)
    with pytest.raises(TypeError):
        ap.process_background_tol([1, 2, 3])


def test_process_qupath_binary(create_executable_files_to_test):
    """Test `ap.process_qupath_binary`."""
    # Should work with a proper executable file (str or Path)
    ap.process_qupath_binary(executable_path)
    ap.process_qupath_binary(str(executable_path))

    # Should not work with a non executable path
    with pytest.raises(FileNotFoundError):
        ap.process_qupath_binary(non_executable_path)

    # Should reject anything which isn't a str or Path
    with pytest.raises(TypeError):
        ap.process_qupath_binary(123)
    with pytest.raises(TypeError):
        ap.process_qupath_binary(["hello.txt"])


def test_check_path_like():
    """Test `ap.check_path_like`."""
    # Should work with str or Path
    ap.check_path_like(Path("batman"))
    ap.check_path_like("batman")

    # Should reject anything that isn't str or Path
    with pytest.raises(TypeError):
        ap.check_path_like(123)
    with pytest.raises(TypeError):
        ap.check_path_like(["hello"])
    with pytest.raises(TypeError):
        ap.check_path_like(1j)


def test_process_cleanup_workers():
    """Test `ap.process_cleanup_workers`."""
    # Should work for positive integers
    ap.process_cleanup_workers(1)
    ap.process_cleanup_workers(10)
    ap.process_cleanup_workers(20)

    # Should reject integers less than one
    with pytest.raises(ValueError):
        ap.process_cleanup_workers(0)
    with pytest.raises(ValueError):
        ap.process_cleanup_workers(-1)
    with pytest.raises(ValueError):
        ap.process_cleanup_workers(-10)

    # Should reject any non-integers
    with pytest.raises(TypeError):
        ap.process_cleanup_workers(1.0)
    with pytest.raises(TypeError):
        ap.process_cleanup_workers(0.0)
    with pytest.raises(TypeError):
        ap.process_cleanup_workers(1.0j)
    with pytest.raises(TypeError):
        ap.process_cleanup_workers([1, 2, 3])


def test_process_zip_patches():
    """Test `ap.process_zip_patches`."""
    # Should work with bool
    ap.process_zip_patches(True)
    ap.process_zip_patches(False)

    # Should break with anything else
    with pytest.raises(TypeError):
        ap.process_zip_patches(1)
    with pytest.raises(TypeError):
        ap.process_zip_patches(1.0)
    with pytest.raises(TypeError):
        ap.process_zip_patches("True")


def test_process_wsi_arg(create_dummy_file_and_dir):
    """Test `ap.process_wsi_arg`."""
    # Should work for files which exist (str or Path)
    ap.process_wsi_arg(Path("file.txt"))
    ap.process_wsi_arg("file.txt")

    # Should fail for non-str or non-path
    with pytest.raises(TypeError):
        ap.process_wsi_arg(123)
    with pytest.raises(TypeError):
        ap.process_wsi_arg(["hello"])

    # Should fail for files which don't exist
    with pytest.raises(FileNotFoundError):
        ap.process_wsi_arg(Path("file-not-exist.txt"))

    # Should fail for directories which either exist or don't
    with pytest.raises(IsADirectoryError):
        ap.process_wsi_arg(Path("dir/"))
    with pytest.raises(FileNotFoundError):
        ap.process_wsi_arg(Path("dir-not-exist/"))


def test_process_parent_dir_arg():
    """Test `ap.process_parent_dir_arg`."""
    # Should work for Path, str or None
    ap.process_parent_dir_arg(Path("bob/the/builder/"))
    ap.process_parent_dir_arg("bob")
    ap.process_parent_dir_arg(None)

    # Should reject anything else
    with pytest.raises(TypeError):
        ap.process_parent_dir_arg(1234)
    with pytest.raises(TypeError):
        ap.process_parent_dir_arg(["/path/in/a/list/"])


def test_process_region_arg():
    """Test `ap.process_region_arg`."""
    correct_keys = ["left", "top", "width", "height"]
    # Should work with Dict[str, int] where the values zero or more and the
    # keys are correct, in any order
    ap.process_region_arg(dict(zip(correct_keys, 4 * [0])))
    ap.process_region_arg(dict(zip(correct_keys[::-1], 4 * [0])))

    # Should also work with None
    ap.process_region_arg(None)

    # Should break if region is not a dict
    with pytest.raises(TypeError):
        ap.process_region_arg([0, 0, 0, 0])

    # Should break if the keys in region are not all str
    with pytest.raises(TypeError):
        ap.process_region_arg({1: 0, "bob": 1})

    # Should break if the values are not all ints
    with pytest.raises(TypeError):
        ap.process_region_arg(dict(zip(correct_keys, [0, 0, 1.0, 0])))

    # Should break if the keys don't match `correct_keys`.
    with pytest.raises(ValueError):
        ap.process_region_arg({"not": 0, "correct": 0, "keys": 0})

    # Should break if any of the values are negative
    with pytest.raises(ValueError):
        ap.process_region_arg(dict(zip(correct_keys, [0, 1, -1, 0])))


def test_process_user_supplied_mag():
    """Test `ap.process_user_supplied_mag`."""
    # Should work with float greater than or equal to 1.0, or None
    assert ap.process_user_supplied_mag(1.0) == 1.0
    assert ap.process_user_supplied_mag(2.0) == 2.0
    assert ap.process_user_supplied_mag(4.0) == 4.0
    assert ap.process_user_supplied_mag(None) is None
    assert ap.process_user_supplied_mag(nan) is None

    # Should break with anything other than float or None
    with pytest.raises(TypeError):
        ap.process_user_supplied_mag(1)
    with pytest.raises(TypeError):
        ap.process_user_supplied_mag("1")

    # Should break with floats less than 1
    with pytest.raises(ValueError):
        ap.process_user_supplied_mag(0.999)
    with pytest.raises(ValueError):
        ap.process_user_supplied_mag(0.0)
    with pytest.raises(ValueError):
        ap.process_user_supplied_mag(-1.0)


def test_process_generate_patches_arg():
    """Test `ap.process_generate_patches_arg`."""
    # Should work with bool
    ap.process_generate_patches_arg(True)
    ap.process_generate_patches_arg(False)

    # Should break with anything else
    with pytest.raises(TypeError):
        ap.process_generate_patches_arg(1)
    with pytest.raises(TypeError):
        ap.process_generate_patches_arg("1")
    with pytest.raises(TypeError):
        ap.process_generate_patches_arg([1])
