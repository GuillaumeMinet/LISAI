# lisai/data/preprocess/sources/folder.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Protocol


# item class
@dataclass(frozen=True)
class Item:
    """
    Represents one logical dataset element to be processed by a pipeline.

    An Item contains:
    - key: a stable identifier used for alignment and naming (usually the filename stem).
    - paths: one or more file paths associated with this element.

    For simple datasets, `paths` contains a single file.
    For paired or multi-source datasets, it may contain multiple aligned files.
    """
    key: str                 # stable alignment key (filename stem usually)
    paths: tuple[Path, ...]  # one or multiple paths depending on source

# base class
class Source(Protocol):
    """
    Abstract source interface for dataset discovery.

    A Source is responsible for enumerating the raw input data and yielding
    `Item` objects. Each Item represents one logical unit to be processed
    by a pipeline.

    Pipelines use a Source to remain independent from filesystem traversal
    details.
    """
    def iter_items(self) -> Iterable[Item]:
        ...

# FolderSource class
@dataclass(frozen=True)
class FolderSource(Source):
    """
    Simple source that reads files from a folder.

    It scans `root` for files matching the given extensions and yields one
    Item per file. If `combine_subfolders` is True, all immediate subfolders
    of `root` are scanned and combined into a single stream.

    This source is suitable for single-image datasets or simple stacks
    stored as individual files.
    """
    root: Path
    exts: tuple[str, ...]
    combine_subfolders: bool = False

    def iter_items(self) -> Iterable[Item]:
        roots = [self.root]
        if self.combine_subfolders:
            roots = sorted((p for p in self.root.iterdir() if p.is_dir()), key=lambda p: p.name)

        files: list[Path] = []
        for r in roots:
            for p in sorted(r.iterdir()):
                if p.is_file() and p.suffix.lower() in self.exts:
                    files.append(p)

        for p in files:
            yield Item(key=p.stem, paths=(p,))
