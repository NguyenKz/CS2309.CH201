"""PIE-Bench helpers (cure-lab/PnPInversion format)."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

PIEBENCH_FORM_URL = "https://forms.gle/hVMkTABb4uvZVjme9"
PIEBENCH_REPO_URL = "https://github.com/cure-lab/PnPInversion"


def strip_prompt_brackets(prompt: str) -> str:
    return prompt.replace("[", "").replace("]", "")


def mask_decode(encoded_mask: list[int], image_shape: tuple[int, int] = (512, 512)) -> "np.ndarray":
    import numpy as np

    length = image_shape[0] * image_shape[1]
    mask_array = np.zeros((length,), dtype=np.float32)
    for i in range(0, len(encoded_mask), 2):
        start = encoded_mask[i]
        run_len = min(encoded_mask[i + 1], length - start)
        mask_array[start : start + run_len] = 1
    mask_array = mask_array.reshape(image_shape)
    mask_array[0, :] = 1
    mask_array[-1, :] = 1
    mask_array[:, 0] = 1
    mask_array[:, -1] = 1
    return mask_array


def find_piebench_root(start: Path) -> Path | None:
    candidates = [
        start,
        start / "PIE-Bench",
        start / "PIE-Bench_v1",
        start / "data",
        start.parent / "PIE-Bench",
        start.parent / "data",
    ]
    for root in candidates:
        if (root / "mapping_file.json").is_file() and (root / "annotation_images").is_dir():
            return root
    return None


def resolve_piebench_dir(project_root: Path, piebench_dir: str | Path | None = None) -> Path:
    if piebench_dir:
        root = Path(piebench_dir)
        if (root / "mapping_file.json").is_file():
            return root
        found = find_piebench_root(root)
        if found:
            return found
        raise FileNotFoundError(f"Không thấy mapping_file.json trong {root}")

    for env_key in ("PIEBENCH_DIR",):
        import os

        if os.environ.get(env_key):
            root = Path(os.environ[env_key])
            found = find_piebench_root(root) or (
                root if (root / "mapping_file.json").is_file() else None
            )
            if found:
                return found

    defaults = [
        project_root / "data" / "PIE-Bench",
        project_root / "data" / "PIE-Bench_v1",
        project_root / "data" / "PIE-Bench-smoke",
        Path("/content/PIE-Bench"),
        Path("/content/data/PIE-Bench"),
        Path("/content/PIE-Bench-smoke"),
    ]
    for p in defaults:
        found = find_piebench_root(p)
        if found:
            return found

    raise FileNotFoundError(
        "Chưa có PIE-Bench.\n"
        f"1. Tải từ Google Form: {PIEBENCH_FORM_URL}\n"
        f"2. Giải nén vào data/PIE-Bench/ (cần mapping_file.json + annotation_images/)\n"
        f"   hoặc: bash scripts/download_piebench.sh /path/to/PIE-Bench.zip\n"
        f"Tham khảo: {PIEBENCH_REPO_URL}"
    )


def extract_piebench_zip(zip_path: Path, dest_dir: Path) -> Path:
    dest_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(dest_dir)
    found = find_piebench_root(dest_dir)
    if not found:
        raise FileNotFoundError(
            f"Giải nén {zip_path} xong nhưng không thấy mapping_file.json trong {dest_dir}"
        )
    return found


def load_mapping(piebench_root: Path) -> dict:
    mapping_path = piebench_root / "mapping_file.json"
    with mapping_path.open(encoding="utf-8") as f:
        return json.load(f)


def select_samples(
    mapping: dict,
    *,
    edit_categories: list[str] | None = None,
    max_samples: int | None = None,
    sample_ids: list[str] | None = None,
) -> list[tuple[str, dict]]:
    if sample_ids:
        return [(sid, mapping[sid]) for sid in sample_ids if sid in mapping]

    cats = set(edit_categories or [])
    items: list[tuple[str, dict]] = []
    for key in sorted(mapping.keys()):
        item = mapping[key]
        if cats and item.get("editing_type_id") not in cats:
            continue
        items.append((key, item))

    if max_samples is None or len(items) <= max_samples:
        return items

    # Cân bằng theo editing_type_id
    by_type: dict[str, list[tuple[str, dict]]] = {}
    for pair in items:
        by_type.setdefault(pair[1].get("editing_type_id", "?"), []).append(pair)

    per_type = max(1, max_samples // max(len(by_type), 1))
    selected: list[tuple[str, dict]] = []
    for type_id in sorted(by_type.keys()):
        selected.extend(by_type[type_id][:per_type])
    if len(selected) < max_samples:
        seen = {k for k, _ in selected}
        for pair in items:
            if pair[0] not in seen:
                selected.append(pair)
                seen.add(pair[0])
            if len(selected) >= max_samples:
                break
    return selected[:max_samples]
