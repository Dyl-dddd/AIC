from __future__ import annotations

from pathlib import Path
import sys
import xml.etree.ElementTree as ET
import zipfile

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.extract_official_data import inspect_archives


def _write_image(path: Path) -> None:
    Image.new("L", (32, 24), color=128).save(path, format="JPEG")


def test_official_zip_normalization_repairs_metadata_and_box(tmp_path):
    image_path = tmp_path / "sample.jpg"
    _write_image(image_path)
    xml = """<annotation>
      <filename>wrong.jpg</filename>
      <size><width>32</width><height>24</height><depth>3</depth></size>
      <object><name>qilie</name><bndbox>
        <xmin>1</xmin><ymin>2</ymin><xmax>10</xmax><ymax>25</ymax>
      </bndbox></object>
    </annotation>"""
    train_zip = tmp_path / "train.zip"
    with zipfile.ZipFile(train_zip, "w") as archive:
        archive.write(image_path, "nested/sample.jpg")
        archive.writestr("nested/sample.xml", xml)
    test_image = tmp_path / "test.jpg"
    _write_image(test_image)
    test_zip = tmp_path / "test.zip"
    with zipfile.ZipFile(test_zip, "w") as archive:
        archive.write(test_image, "test/test.jpg")

    output = tmp_path / "normalized"
    report = inspect_archives(train_zip, test_zip, output)
    assert report["train"]["repairs"] == {
        "box_clipped": 1,
        "depth_corrected": 1,
        "filename_stem_corrected": 1,
    }
    root = ET.parse(output / "train" / "sample.xml").getroot()
    assert root.findtext("filename") == "sample.jpg"
    assert root.findtext("size/depth") == "1"
    assert root.findtext("object/bndbox/ymax") == "24"
    assert (output / "test" / "test.jpg").is_file()


def test_official_zip_rejects_path_traversal(tmp_path):
    train_zip = tmp_path / "train.zip"
    with zipfile.ZipFile(train_zip, "w") as archive:
        archive.writestr("../escape.jpg", b"not-an-image")
    test_zip = tmp_path / "test.zip"
    with zipfile.ZipFile(test_zip, "w") as archive:
        archive.writestr("test.jpg", b"not-an-image")
    try:
        inspect_archives(train_zip, test_zip)
    except ValueError as exc:
        assert "Unsafe" in str(exc)
    else:
        raise AssertionError("unsafe ZIP entry was accepted")


def test_official_zip_rejects_backslash_path_traversal(tmp_path):
    train_zip = tmp_path / "train.zip"
    with zipfile.ZipFile(train_zip, "w") as archive:
        archive.writestr("..\\escape.jpg", b"not-an-image")
    test_zip = tmp_path / "test.zip"
    with zipfile.ZipFile(test_zip, "w") as archive:
        archive.writestr("test.jpg", b"not-an-image")
    try:
        inspect_archives(train_zip, test_zip)
    except ValueError as exc:
        assert "Unsafe" in str(exc)
    else:
        raise AssertionError("unsafe backslash ZIP entry was accepted")
