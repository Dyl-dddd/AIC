from __future__ import annotations

from pathlib import Path
import sys
import zipfile

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.extract_test_data import inspect_test_archive


def _write_image(path: Path) -> None:
    Image.new("L", (32, 24), color=128).save(path, format="JPEG")


def test_image_only_archive_extracts_to_flat_normalized_directory(tmp_path):
    image = tmp_path / "sample.jpg"
    _write_image(image)
    archive_path = tmp_path / "test.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.write(image, "nested/sample.jpg")

    output = tmp_path / "output"
    report = inspect_test_archive(archive_path, output)
    assert report["images"] == 1
    assert report["dimensions"] == {"32x24": 1}
    assert report["modes"] == {"L": 1}
    assert (output / "sample.jpg").is_file()


def test_image_only_archive_rejects_non_image_members(tmp_path):
    image = tmp_path / "sample.jpg"
    _write_image(image)
    archive_path = tmp_path / "test.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.write(image, "sample.jpg")
        archive.writestr("instructions.txt", "ignore me")

    try:
        inspect_test_archive(archive_path)
    except ValueError as exc:
        assert "Unexpected non-image" in str(exc)
    else:
        raise AssertionError("non-image archive member was accepted")


def test_existing_empty_staging_directory_is_supported(tmp_path):
    image = tmp_path / "sample.jpg"
    _write_image(image)
    archive_path = tmp_path / "test.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.write(image, "sample.jpg")

    staging = tmp_path / "output.staging"
    staging.mkdir()
    report = inspect_test_archive(archive_path, staging)
    assert report["images"] == 1
    assert (staging / "sample.jpg").is_file()
