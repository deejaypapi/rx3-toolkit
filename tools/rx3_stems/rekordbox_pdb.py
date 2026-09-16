"""
RX3 Stem Studio - Rekordbox USB/PDB bridge.

Reads a Rekordbox export.pdb directly from a USB/SSD and exposes:
- tracks
- artists
- albums
- genres
- labels
- keys
- playlist folders
- playlists and their track IDs

It can also generate a Rekordbox-compatible XML representation for
a selected playlist.

This module intentionally uses only Python standard-library modules.
"""

from __future__ import annotations

import html
import os
import struct
from dataclasses import dataclass, field
from enum import IntEnum
from pathlib import Path
from typing import Callable
from urllib.parse import quote


# ---------------------------------------------------------------------------
# DeviceSQL / PDB constants
# ---------------------------------------------------------------------------

PAGE_HEADER_SIZE = 0x28
ROW_GROUP_SIZE = 0x24


class TableType(IntEnum):
    TRACKS = 0
    GENRES = 1
    ARTISTS = 2
    ALBUMS = 3
    LABELS = 4
    KEYS = 5
    COLORS = 6
    PLAYLIST_TREE = 7
    PLAYLIST_ENTRIES = 8
    UNKNOWN_9 = 9
    UNKNOWN_10 = 10
    HISTORY_PLAYLISTS = 11
    HISTORY_ENTRIES = 12
    ARTWORK = 13
    UNKNOWN_14 = 14
    UNKNOWN_15 = 15
    COLUMNS = 16
    UNKNOWN_17 = 17
    UNKNOWN_18 = 18
    HISTORY = 19


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------

def _u16(buf: bytes, off: int) -> int:
    return struct.unpack_from("<H", buf, off)[0]


def _u32(buf: bytes, off: int) -> int:
    return struct.unpack_from("<I", buf, off)[0]


def _safe_decode_ascii(data: bytes) -> str:
    return data.decode("ascii", errors="replace")


def _safe_decode_utf16(data: bytes) -> str:
    return data.decode("utf-16-le", errors="replace")


def decode_device_sql_string(buf: bytes, off: int) -> str:
    """
    Decode a Rekordbox DeviceSQL string.

    Short ASCII:
        one odd length byte followed by ASCII data.

    Long ASCII:
        0x40 + u16 total length + reserved byte + payload.

    Long UTF-16LE:
        0x90 + u16 total length + reserved byte + UTF-16LE payload.
    """
    if off < 0 or off >= len(buf):
        return ""

    b0 = buf[off]

    # Short ASCII string.
    if b0 & 1:
        length = (b0 >> 1) - 1
        if length < 0:
            return ""
        end = min(off + 1 + length, len(buf))
        return _safe_decode_ascii(buf[off + 1:end])

    # Long string.
    if off + 4 > len(buf):
        return ""

    total = _u16(buf, off + 1)

    if total < 4:
        return ""

    end = min(off + total, len(buf))
    payload = buf[off + 4:end]

    if b0 == 0x40:
        return _safe_decode_ascii(payload)

    if b0 == 0x90:
        return _safe_decode_utf16(payload)

    # Some damaged/older exports can contain unexpected values.
    # Do not crash the entire library over one optional string.
    return ""


# ---------------------------------------------------------------------------
# Parsed row models
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PdbTable:
    type: int
    empty_candidate: int
    first_page: int
    last_page: int


_TRACK_FIXED = struct.Struct("<HHIIIIIHH12I5H HBBHH")


@dataclass(frozen=True)
class PdbTrack:
    index_shift: int
    bitmask: int
    sample_rate: int
    composer_id: int
    file_size: int
    artwork_id: int
    key_id: int
    original_artist_id: int
    label_id: int
    remixer_id: int
    bitrate: int
    track_number: int
    tempo: int
    genre_id: int
    album_id: int
    artist_id: int
    id: int
    disc_number: int
    play_count: int
    year: int
    sample_depth: int
    duration: int
    color_id: int
    rating: int
    strings: tuple[str, ...]

    @property
    def bpm(self) -> float:
        return self.tempo / 100.0

    @property
    def title(self) -> str:
        return self.strings[17] if len(self.strings) > 17 else ""

    @property
    def filename(self) -> str:
        return self.strings[19] if len(self.strings) > 19 else ""

    @property
    def file_path(self) -> str:
        return self.strings[20] if len(self.strings) > 20 else ""

    @property
    def comment(self) -> str:
        return self.strings[16] if len(self.strings) > 16 else ""

    @property
    def mix_name(self) -> str:
        return self.strings[12] if len(self.strings) > 12 else ""

    @property
    def date_added(self) -> str:
        return self.strings[10] if len(self.strings) > 10 else ""


@dataclass(frozen=True)
class PdbGenre:
    id: int
    name: str


@dataclass(frozen=True)
class PdbArtist:
    id: int
    name: str
    index_shift: int


@dataclass(frozen=True)
class PdbAlbum:
    id: int
    name: str
    artist_id: int
    index_shift: int


@dataclass(frozen=True)
class PdbLabel:
    id: int
    name: str


@dataclass(frozen=True)
class PdbKey:
    id: int
    name: str


@dataclass(frozen=True)
class PdbColor:
    id: int
    name: str


@dataclass(frozen=True)
class PdbPlaylistTreeNode:
    id: int
    parent_id: int
    sort_order: int
    name: str
    is_folder: bool


@dataclass(frozen=True)
class PdbPlaylistEntry:
    entry_index: int
    track_id: int
    playlist_id: int


@dataclass(frozen=True)
class PdbUnknownRow:
    page_index: int
    offset: int


# ---------------------------------------------------------------------------
# Row parsers
# ---------------------------------------------------------------------------

def parse_track(page: bytes, row: int) -> PdbTrack:
    f = _TRACK_FIXED.unpack_from(page, row)

    (
        _,
        index_shift,
        bitmask,
        sample_rate,
        composer_id,
        file_size,
        _,
        _,
        _,
        artwork_id,
        key_id,
        original_artist_id,
        label_id,
        remixer_id,
        bitrate,
        track_number,
        tempo,
        genre_id,
        album_id,
        artist_id,
        track_id,
        disc_number,
        play_count,
        year,
        sample_depth,
        duration,
        _,
        color_id,
        rating,
        _,
        _,
    ) = f

    string_offsets = struct.unpack_from(
        "<21H",
        page,
        row + _TRACK_FIXED.size,
    )

    strings = tuple(
        decode_device_sql_string(page, row + offset)
        for offset in string_offsets
    )

    return PdbTrack(
        index_shift=index_shift,
        bitmask=bitmask,
        sample_rate=sample_rate,
        composer_id=composer_id,
        file_size=file_size,
        artwork_id=artwork_id,
        key_id=key_id,
        original_artist_id=original_artist_id,
        label_id=label_id,
        remixer_id=remixer_id,
        bitrate=bitrate,
        track_number=track_number,
        tempo=tempo,
        genre_id=genre_id,
        album_id=album_id,
        artist_id=artist_id,
        id=track_id,
        disc_number=disc_number,
        play_count=play_count,
        year=year,
        sample_depth=sample_depth,
        duration=duration,
        color_id=color_id,
        rating=rating,
        strings=strings,
    )


def parse_genre(page: bytes, row: int) -> PdbGenre:
    return PdbGenre(
        id=_u32(page, row),
        name=decode_device_sql_string(page, row + 4),
    )


def parse_label(page: bytes, row: int) -> PdbLabel:
    return PdbLabel(
        id=_u32(page, row),
        name=decode_device_sql_string(page, row + 4),
    )


def parse_artist(page: bytes, row: int) -> PdbArtist:
    subtype = _u16(page, row)
    index_shift = _u16(page, row + 2)
    artist_id = _u32(page, row + 4)

    if subtype == 0x64:
        name_offset = _u16(page, row + 0x0A)
    else:
        name_offset = page[row + 9]

    return PdbArtist(
        id=artist_id,
        name=decode_device_sql_string(page, row + name_offset),
        index_shift=index_shift,
    )


def parse_album(page: bytes, row: int) -> PdbAlbum:
    index_shift = _u16(page, row + 2)
    artist_id = _u32(page, row + 8)
    album_id = _u32(page, row + 12)
    name_offset = page[row + 0x15]

    return PdbAlbum(
        id=album_id,
        name=decode_device_sql_string(page, row + name_offset),
        artist_id=artist_id,
        index_shift=index_shift,
    )


def parse_key(page: bytes, row: int) -> PdbKey:
    return PdbKey(
        id=_u32(page, row),
        name=decode_device_sql_string(page, row + 8),
    )


def parse_color(page: bytes, row: int) -> PdbColor:
    return PdbColor(
        id=_u16(page, row + 5),
        name=decode_device_sql_string(page, row + 8),
    )


def parse_playlist_tree(page: bytes, row: int) -> PdbPlaylistTreeNode:
    return PdbPlaylistTreeNode(
        parent_id=_u32(page, row),
        sort_order=_u32(page, row + 8),
        id=_u32(page, row + 12),
        is_folder=_u32(page, row + 16) != 0,
        name=decode_device_sql_string(page, row + 20),
    )


def parse_playlist_entry(page: bytes, row: int) -> PdbPlaylistEntry:
    return PdbPlaylistEntry(
        entry_index=_u32(page, row),
        track_id=_u32(page, row + 4),
        playlist_id=_u32(page, row + 8),
    )


# ---------------------------------------------------------------------------
# Parser dispatch
# ---------------------------------------------------------------------------

_ROW_PARSERS: dict[int, Callable[[bytes, int], object]] = {
    TableType.TRACKS: parse_track,
    TableType.GENRES: parse_genre,
    TableType.ARTISTS: parse_artist,
    TableType.ALBUMS: parse_album,
    TableType.LABELS: parse_label,
    TableType.KEYS: parse_key,
    TableType.COLORS: parse_color,
    TableType.PLAYLIST_TREE: parse_playlist_tree,
    TableType.PLAYLIST_ENTRIES: parse_playlist_entry,
}


# ---------------------------------------------------------------------------
# PDB database
# ---------------------------------------------------------------------------

class RekordboxPdb:
    """
    Read-only Rekordbox export.pdb parser.

    This class deliberately does not modify the USB database.
    """

    def __init__(self, data: bytes):
        if len(data) < 0x20:
            raise ValueError("export.pdb is too small to be a valid Rekordbox PDB.")

        self._data = data
        self.page_size = _u32(data, 4)

        if self.page_size <= 0:
            raise ValueError("Invalid Rekordbox PDB page size.")

        if self.page_size != 4096:
            raise ValueError(
                f"Unsupported Rekordbox PDB page size: {self.page_size}"
            )

        num_tables = _u32(data, 8)

        if num_tables <= 0 or num_tables > 64:
            raise ValueError(
                f"Invalid Rekordbox PDB table count: {num_tables}"
            )

        directory_end = 0x1C + num_tables * 16

        if directory_end > len(data):
            raise ValueError("Rekordbox PDB table directory is truncated.")

        self.tables: list[PdbTable] = []

        for index in range(num_tables):
            base = 0x1C + index * 16

            self.tables.append(
                PdbTable(
                    type=_u32(data, base),
                    empty_candidate=_u32(data, base + 4),
                    first_page=_u32(data, base + 8),
                    last_page=_u32(data, base + 12),
                )
            )

        self._rows: dict[int, list[object]] = {}

        for table in self.tables:
            self._rows[table.type] = self._parse_table(table)

    @classmethod
    def from_file(cls, path: str | Path) -> "RekordboxPdb":
        path = Path(path)

        if not path.is_file():
            raise FileNotFoundError(f"Rekordbox PDB not found: {path}")

        return cls(path.read_bytes())

    def rows(self, table_type: int) -> list[object]:
        return self._rows.get(int(table_type), [])

    @property
    def tracks(self) -> list[PdbTrack]:
        return self.rows(TableType.TRACKS)  # type: ignore[return-value]

    @property
    def genres(self) -> list[PdbGenre]:
        return self.rows(TableType.GENRES)  # type: ignore[return-value]

    @property
    def artists(self) -> list[PdbArtist]:
        return self.rows(TableType.ARTISTS)  # type: ignore[return-value]

    @property
    def albums(self) -> list[PdbAlbum]:
        return self.rows(TableType.ALBUMS)  # type: ignore[return-value]

    @property
    def labels(self) -> list[PdbLabel]:
        return self.rows(TableType.LABELS)  # type: ignore[return-value]

    @property
    def keys(self) -> list[PdbKey]:
        return self.rows(TableType.KEYS)  # type: ignore[return-value]

    @property
    def colors(self) -> list[PdbColor]:
        return self.rows(TableType.COLORS)  # type: ignore[return-value]

    @property
    def playlist_tree(self) -> list[PdbPlaylistTreeNode]:
        return self.rows(TableType.PLAYLIST_TREE)  # type: ignore[return-value]

    @property
    def playlist_entries(self) -> list[PdbPlaylistEntry]:
        return self.rows(TableType.PLAYLIST_ENTRIES)  # type: ignore[return-value]

    def _page(self, index: int) -> bytes:
        start = index * self.page_size
        end = start + self.page_size

        if start < 0 or end > len(self._data):
            raise ValueError(
                f"Rekordbox PDB references invalid page {index}."
            )

        return self._data[start:end]

    def _parse_table(self, table: PdbTable) -> list[object]:
        parser = _ROW_PARSERS.get(table.type)

        rows: list[object] = []

        page_index = table.first_page
        visited_pages: set[int] = set()

        while True:
            if page_index in visited_pages:
                raise ValueError(
                    f"Loop detected in Rekordbox PDB table {table.type}."
                )

            visited_pages.add(page_index)

            page = self._page(page_index)

            for row_offset in self._row_offsets(page):
                if parser is not None:
                    try:
                        rows.append(parser(page, row_offset))
                    except (IndexError, struct.error, UnicodeError):
                        continue

            if page_index == table.last_page:
                break

            next_page = _u32(page, 12)

            if next_page == page_index:
                raise ValueError(
                    f"Invalid page chain in Rekordbox PDB table {table.type}."
                )

            page_index = next_page

        return rows

    def _row_offsets(self, page: bytes) -> list[int]:
        page_flags = page[27]

        if page_flags & 0x40:
            return []

        num_rows = page[24] + 0x100 * (page[25] & 1)

        if num_rows == 0:
            return []

        offsets: list[int] = []

        num_groups = (num_rows - 1) // 16 + 1

        for group in range(num_groups):
            base = self.page_size - group * ROW_GROUP_SIZE

            presence_flags = _u16(page, base - 4)

            in_group = (
                16
                if group < num_groups - 1
                else (num_rows - 1) % 16 + 1
            )

            for index in range(in_group):
                if (presence_flags >> index) & 1:
                    offset = PAGE_HEADER_SIZE + _u16(
                        page,
                        base - 6 - 2 * index,
                    )

                    if PAGE_HEADER_SIZE <= offset < self.page_size:
                        offsets.append(offset)

        return offsets


# ---------------------------------------------------------------------------
# USB discovery
# ---------------------------------------------------------------------------

def find_export_pdb(root: str | Path) -> Path:
    """
    Locate PIONEER/rekordbox/export.pdb below a USB/SSD root.
    """
    root = Path(root)

    if not root.is_dir():
        raise NotADirectoryError(f"USB/SSD path is not a directory: {root}")

    candidates = [
        root / "PIONEER" / "rekordbox" / "export.pdb",
        root / "PIONEER" / "rekordbox" / "exportPDB.pdb",
        root / "PIONEER" / "REKORDBOX" / "export.pdb",
    ]

    for candidate in candidates:
        if candidate.is_file():
            return candidate

    pioneer = None

    for child in root.iterdir():
        if child.is_dir() and child.name.lower() == "pioneer":
            pioneer = child
            break

    if pioneer is None:
        raise FileNotFoundError(
            "PIONEER folder was not found on the selected USB/SSD."
        )

    rekordbox_dir = None

    for child in pioneer.iterdir():
        if child.is_dir() and child.name.lower() == "rekordbox":
            rekordbox_dir = child
            break

    if rekordbox_dir is None:
        raise FileNotFoundError(
            "PIONEER\\rekordbox folder was not found on the selected USB/SSD."
        )

    for child in rekordbox_dir.iterdir():
        if child.is_file() and child.name.lower() == "export.pdb":
            return child

    raise FileNotFoundError(
        "PIONEER\\rekordbox\\export.pdb was not found on the selected USB/SSD."
    )


def load_from_usb(root: str | Path) -> RekordboxPdb:
    """
    Discover and load export.pdb from a selected USB/SSD root.
    """
    return RekordboxPdb.from_file(find_export_pdb(root))


# ---------------------------------------------------------------------------
# Friendly playlist model
# ---------------------------------------------------------------------------

@dataclass
class UsbPlaylist:
    id: int
    name: str
    parent_id: int | None
    is_folder: bool
    sort_order: int = 0
    children: list["UsbPlaylist"] = field(default_factory=list)
    track_ids: list[int] = field(default_factory=list)

    @property
    def is_playlist(self) -> bool:
        return not self.is_folder


def build_playlists(db: RekordboxPdb) -> list[UsbPlaylist]:
    """
    Convert the raw PDB playlist tree + entries into a clean hierarchy.
    """
    nodes = list(db.playlist_tree)
    entries = list(db.playlist_entries)

    by_id: dict[int, UsbPlaylist] = {}

    for node in nodes:
        by_id[node.id] = UsbPlaylist(
            id=node.id,
            name=node.name or "(Unnamed)",
            parent_id=node.parent_id if node.parent_id != 0 else None,
            is_folder=node.is_folder,
            sort_order=node.sort_order,
        )

    for entry in entries:
        playlist = by_id.get(entry.playlist_id)

        if playlist is not None:
            playlist.track_ids.append(entry.track_id)

    roots: list[UsbPlaylist] = []

    for playlist in by_id.values():
        parent = by_id.get(playlist.parent_id or 0)

        if parent is None:
            roots.append(playlist)
        else:
            parent.children.append(playlist)

    def sort_tree(items: list[UsbPlaylist]) -> None:
        items.sort(key=lambda item: (item.sort_order, item.name.lower()))

        for item in items:
            sort_tree(item.children)

    sort_tree(roots)

    return roots


def flatten_playlists(
    playlists: list[UsbPlaylist],
    include_folders: bool = False,
) -> list[UsbPlaylist]:
    """
    Flatten playlist hierarchy while preserving display order.
    """
    result: list[UsbPlaylist] = []

    def visit(items: list[UsbPlaylist], prefix: str = "") -> None:
        for item in items:
            display_name = (
                f"{prefix} / {item.name}"
                if prefix
                else item.name
            )

            if item.is_playlist or include_folders:
                copy = UsbPlaylist(
                    id=item.id,
                    name=display_name,
                    parent_id=item.parent_id,
                    is_folder=item.is_folder,
                    sort_order=item.sort_order,
                    children=item.children,
                    track_ids=list(item.track_ids),
                )

                result.append(copy)

            if item.children:
                next_prefix = display_name if item.is_folder else prefix
                visit(item.children, next_prefix)

    visit(playlists)

    return result


def playlist_by_name(
    playlists: list[UsbPlaylist],
    name: str,
) -> UsbPlaylist | None:
    wanted = name.strip().casefold()

    for playlist in flatten_playlists(playlists):
        if playlist.name.casefold() == wanted:
            return playlist

    return None


# ---------------------------------------------------------------------------
# Metadata lookup
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ResolvedTrack:
    id: int
    title: str
    artist: str
    album: str
    genre: str
    key: str
    label: str
    bpm: float
    duration: int
    bitrate: int
    sample_rate: int
    rating: int
    year: int
    track_number: int
    disc_number: int
    play_count: int
    comment: str
    mix_name: str
    file_path: str


def resolve_tracks(
    db: RekordboxPdb,
    track_ids: set[int] | None = None,
) -> list[ResolvedTrack]:
    """
    Resolve foreign-key IDs from the PDB into human-readable metadata.
    """
    artists = {item.id: item.name for item in db.artists}
    albums = {item.id: item.name for item in db.albums}
    genres = {item.id: item.name for item in db.genres}
    labels = {item.id: item.name for item in db.labels}
    keys = {item.id: item.name for item in db.keys}

    result: list[ResolvedTrack] = []

    for track in db.tracks:
        if track_ids is not None and track.id not in track_ids:
            continue

        result.append(
            ResolvedTrack(
                id=track.id,
                title=track.title,
                artist=artists.get(track.artist_id) or "Unknown Artist",
                album=albums.get(track.album_id) or "Unknown Album",
                genre=genres.get(track.genre_id, ""),
                key=keys.get(track.key_id, ""),
                label=labels.get(track.label_id, ""),
                bpm=track.bpm,
                duration=track.duration,
                bitrate=track.bitrate,
                sample_rate=track.sample_rate,
                rating=track.rating,
                year=track.year,
                track_number=track.track_number,
                disc_number=track.disc_number,
                play_count=track.play_count,
                comment=track.comment,
                mix_name=track.mix_name,
                file_path=track.file_path,
            )
        )

    return result


# ---------------------------------------------------------------------------
# Windows path / Rekordbox XML location
# ---------------------------------------------------------------------------

def _normalise_pdb_path(value: str) -> str:
    value = (value or "").strip()

    if not value:
        return ""

    value = value.replace("\\", "/")

    while "//" in value:
        value = value.replace("//", "/")

    return value


def resolve_track_path(
    usb_root: str | Path,
    pdb_file_path: str,
) -> Path:
    """
    Convert the path stored by Rekordbox into an actual Windows path.

    Rekordbox exports normally store a media-relative path such as:
        /Contents/Artist/Track.mp3

    If the PDB already contains an absolute Windows path, preserve it.
    """
    root = Path(usb_root).resolve()
    raw = _normalise_pdb_path(pdb_file_path)

    if not raw:
        return root

    # Already a Windows absolute path, e.g. D:/Music/file.mp3
    if len(raw) >= 3 and raw[1:3] == ":/":
        return Path(raw)

    # UNC path.
    if raw.startswith("//"):
        return Path(raw.replace("/", os.sep))

    relative = raw.lstrip("/")

    return root / Path(relative)


def file_path_to_rekordbox_url(
    path: str | Path,
) -> str:
    """
    Convert a Windows filesystem path to:
        file://localhost/C:/...
    with Rekordbox-compatible percent encoding.
    """
    resolved = Path(path).resolve()

    drive = resolved.drive

    if drive:
        drive_letter = drive.rstrip(":").upper()
        tail = resolved.as_posix()

        if tail.startswith(f"{drive}/"):
            tail = tail[len(drive) + 1:]

        tail = "/" + tail.lstrip("/")

        return (
            "file://localhost/"
            + drive_letter
            + ":"
            + quote(tail, safe="/:")
        )

    return (
        "file://localhost/"
        + quote(
            resolved.as_posix().lstrip("/"),
            safe="/:",
        )
    )


# ---------------------------------------------------------------------------
# XML generation
# ---------------------------------------------------------------------------

def _xml_escape(value: object) -> str:
    return html.escape(
        str(value if value is not None else ""),
        quote=True,
    )


def _xml_location(
    usb_root: str | Path,
    pdb_file_path: str,
) -> str:
    actual = resolve_track_path(
        usb_root,
        pdb_file_path,
    )

    return file_path_to_rekordbox_url(actual)


def _track_xml(
    track: ResolvedTrack,
    usb_root: str | Path,
) -> str:
    extension = Path(
        track.file_path
    ).suffix.lower()

    kinds = {
        ".mp3": "MP3 File",
        ".wav": "WAV File",
        ".aif": "AIFF File",
        ".aiff": "AIFF File",
        ".m4a": "AAC File",
        ".aac": "AAC File",
        ".flac": "FLAC File",
        ".alac": "ALAC File",
        ".ogg": "Ogg File",
    }

    kind = kinds.get(
        extension,
        "Unknown",
    )

    return (
        "      <TRACK "
        f'TrackID="{_xml_escape(track.id)}" '
        f'Name="{_xml_escape(track.title)}" '
        f'Artist="{_xml_escape(track.artist)}" '
        f'Composer="" '
        f'Album="{_xml_escape(track.album)}" '
        f'Grouping="" '
        f'Genre="{_xml_escape(track.genre)}" '
        f'Kind="{_xml_escape(kind)}" '
        f'TotalTime="{_xml_escape(track.duration)}" '
        f'DiscNumber="{_xml_escape(track.disc_number)}" '
        f'TrackNumber="{_xml_escape(track.track_number)}" '
        f'Year="{_xml_escape(track.year)}" '
        f'AverageBpm="{track.bpm:.2f}" '
        f'BitRate="{_xml_escape(track.bitrate)}" '
        f'SampleRate="{_xml_escape(track.sample_rate or 44100)}" '
        f'Comments="{_xml_escape(track.comment)}" '
        f'PlayCount="{_xml_escape(track.play_count)}" '
        f'Rating="{_xml_escape(track.rating)}" '
        f'Location="{_xml_escape(_xml_location(usb_root, track.file_path))}" '
        f'Remixer="" '
        f'Tonality="{_xml_escape(track.key)}" '
        f'Label="{_xml_escape(track.label)}" '
        f'Mix="{_xml_escape(track.mix_name)}" '
        "/>\n"
    )


def generate_playlist_xml(
    db: RekordboxPdb,
    usb_root: str | Path,
    playlist: UsbPlaylist,
) -> str:
    """
    Generate a Rekordbox XML representation matching the original
    Rekordbox XML structure for the selected playlist.
    """
    wanted_ids = set(playlist.track_ids)

    tracks = resolve_tracks(
        db,
        wanted_ids,
    )

    track_map = {
        track.id: track
        for track in tracks
    }

    lines: list[str] = []

    # XML declaration.
    lines.append(
        '<?xml version="1.0" encoding="UTF-8"?>\n'
    )

    # Root.
    lines.append(
        '<DJ_PLAYLISTS Version="1.0.0">\n'
    )

    # Match the Rekordbox reference XML.
    lines.append(
        '  <PRODUCT Name="rekordbox" '
        'Version="6.8.5" '
        'Company="AlphaTheta"/>\n'
    )

    # Collection.
    lines.append(
        '  <COLLECTION Entries="'
        + str(len(tracks))
        + '">\n'
    )

    for track in tracks:
        lines.append(
            _track_xml(
                track,
                usb_root,
            )
        )

    lines.append(
        "  </COLLECTION>\n"
    )

    # Playlist tree.
    lines.append(
        "  <PLAYLISTS>\n"
    )

    lines.append(
        '    <NODE Type="0" Name="ROOT" Count="1">\n'
    )

    # Match original Rekordbox playlist NODE attributes.
    lines.append(
        "      <NODE "
        f'Name="{_xml_escape(playlist.name)}" '
        'Type="1" '
        'KeyType="0" '
        f'Entries="{len(playlist.track_ids)}">\n'
    )

    # Preserve exact playlist track order.
    for track_id in playlist.track_ids:
        if track_id not in track_map:
            continue

        lines.append(
            "        <TRACK "
            f'Key="{_xml_escape(track_id)}"'
            "/>\n"
        )

    lines.append(
        "      </NODE>\n"
    )

    lines.append(
        "    </NODE>\n"
    )

    lines.append(
        "  </PLAYLISTS>\n"
    )

    lines.append(
        "</DJ_PLAYLISTS>\n"
    )

    return "".join(lines)


def write_playlist_xml(
    db: RekordboxPdb,
    usb_root: str | Path,
    playlist: UsbPlaylist,
    destination: str | Path,
) -> Path:
    """
    Write the generated XML to a temporary or requested destination.
    """
    destination = Path(destination)

    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    xml = generate_playlist_xml(
        db,
        usb_root,
        playlist,
    )

    destination.write_text(
        xml,
        encoding="utf-8",
        newline="\n",
    )

    return destination


# ---------------------------------------------------------------------------
# High-level convenience API for Stem Studio
# ---------------------------------------------------------------------------

@dataclass
class UsbRekordboxLibrary:
    usb_root: Path
    pdb_path: Path
    database: RekordboxPdb
    playlists: list[UsbPlaylist]

    @classmethod
    def open(
        cls,
        usb_root: str | Path,
    ) -> "UsbRekordboxLibrary":
        root = Path(usb_root).resolve()

        pdb_path = find_export_pdb(root)
        database = RekordboxPdb.from_file(pdb_path)
        playlists = build_playlists(database)

        return cls(
            usb_root=root,
            pdb_path=pdb_path,
            database=database,
            playlists=playlists,
        )

    def playlist_list(self) -> list[UsbPlaylist]:
        return flatten_playlists(
            self.playlists
        )

    def find_playlist(
        self,
        name: str,
    ) -> UsbPlaylist | None:
        return playlist_by_name(
            self.playlists,
            name,
        )

    def tracks_for_playlist(
        self,
        playlist: UsbPlaylist,
    ) -> list[ResolvedTrack]:
        return resolve_tracks(
            self.database,
            set(playlist.track_ids),
        )

    def export_playlist_xml(
        self,
        playlist: UsbPlaylist,
        destination: str | Path,
    ) -> Path:
        return write_playlist_xml(
            self.database,
            self.usb_root,
            playlist,
            destination,
        )


# ---------------------------------------------------------------------------
# Command-line self-test
# ---------------------------------------------------------------------------

def _main() -> int:
    import argparse

    parser = argparse.ArgumentParser(
        description="Inspect a Rekordbox USB export.pdb."
    )

    parser.add_argument(
        "usb",
        help="USB/SSD root containing the PIONEER folder",
    )

    parser.add_argument(
        "--playlist",
        help="Playlist name to export to XML",
    )

    parser.add_argument(
        "--xml",
        help="Destination XML file",
    )

    args = parser.parse_args()

    try:
        library = UsbRekordboxLibrary.open(
            args.usb
        )
    except Exception as error:
        print(f"ERROR: {error}")
        return 1

    print(
        f"PDB: {library.pdb_path}"
    )

    print(
        f"Tracks: {len(library.database.tracks)}"
    )

    print("Playlists:")

    playlists = library.playlist_list()

    for item in playlists:
        print(
            f"  {item.name} "
            f"({len(item.track_ids)} tracks)"
        )

    if not args.playlist:
        return 0

    playlist = library.find_playlist(
        args.playlist
    )

    if playlist is None:
        print(
            f"ERROR: playlist not found: {args.playlist}"
        )
        return 2

    tracks = library.tracks_for_playlist(
        playlist
    )

    print(
        f"Selected playlist: {playlist.name}"
    )

    print(
        f"Selected tracks: {len(tracks)}"
    )

    if args.xml:
        destination = library.export_playlist_xml(
            playlist,
            args.xml,
        )

        print(
            f"XML: {destination}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(_main())