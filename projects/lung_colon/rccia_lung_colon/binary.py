"""Binary benign/malignant mapping for the LC25000 classes."""

from __future__ import annotations


BINARY_CLASSES = ("benign", "malignant")

BINARY_CLASS_MAPPING = {
    "colon_adenocarcinoma": "malignant",
    "colon_benign": "benign",
    "lung_adenocarcinoma": "malignant",
    "lung_benign": "benign",
    "lung_squamous_cell_carcinoma": "malignant",
}

SOURCE_CLASSES_BY_BINARY = {
    binary_class: tuple(
        source_class
        for source_class, mapped_binary_class in BINARY_CLASS_MAPPING.items()
        if mapped_binary_class == binary_class
    )
    for binary_class in BINARY_CLASSES
}


def binary_label_for_source(source_class: str) -> str:
    try:
        return BINARY_CLASS_MAPPING[source_class]
    except KeyError as exc:
        expected = ", ".join(sorted(BINARY_CLASS_MAPPING))
        raise ValueError(f"Unknown LC25000 class '{source_class}'. Expected one of: {expected}.") from exc


def source_classes_for_binary_label(binary_label: str) -> tuple[str, ...]:
    try:
        return SOURCE_CLASSES_BY_BINARY[binary_label]
    except KeyError as exc:
        expected = ", ".join(BINARY_CLASSES)
        raise ValueError(f"Unknown binary label '{binary_label}'. Expected one of: {expected}.") from exc
