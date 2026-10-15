#!/usr/bin/env python3
"""Fetch CUFED albums from the Hugging Face mirror into data/cufed_<Event>_<album_id>/.

The mirror (Shawn-Huang/CUFED-AlbumBench) holds 133 albums as images/<album_id>/<photo_id>.jpg
plus event_type.json and image_importance.json. Each fetched album gets a manifest.json
in the same shape as the hand-made ones from 2026-10-06: photos in the importance index's
(chronological) order, index-prefixed filenames so that order survives without EXIF, and
the per-photo importance score that Stage 0 metrics read.

Licences: the mirror's collection tag is cc-by-nc-2.0, but each photo carries its own
Flickr licence. --check-licences scrapes each Flickr page (best effort); without it the
manifest says "unchecked" and the album must not be reproduced in the thesis.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import re
import shutil
from pathlib import Path

import httpx

logger = logging.getLogger(__name__)

MIRROR_REPO = "Shawn-Huang/CUFED-AlbumBench"
MIRROR_URL = f"https://huggingface.co/datasets/{MIRROR_REPO}"
SOURCE_NOTE = f"CUFED (Wang et al., BMVC 2017), mirror: {MIRROR_URL}"
FLICKR_PAGE = "https://www.flickr.com/photos/{nsid}/{photo_id}"
FLICKR_TIMEOUT_S = 20.0
_CC_LICENCE_RE = re.compile(r"creativecommons\.org/licenses/([a-z-]+)/([0-9.]+)")
_OG_TITLE_RE = re.compile(r'<meta property="og:title" content="([^"]*)"')


def plan_albums(
    event_types: dict[str, list[str]], wanted: dict[str, int], existing_ids: set[str]
) -> list[tuple[str, str]]:
    """(album_id, event) pairs to fetch so each wanted event reaches its quota, counting
    albums already on disk; albums are taken in id order for reproducibility."""
    plan: list[tuple[str, str]] = []
    taken = set(existing_ids)
    for event, quota in wanted.items():
        matching = sorted(aid for aid, types in event_types.items() if event in types)
        # Albums already on disk or planned under an earlier event count toward this quota
        # but are not fetched again.
        have = sum(aid in taken for aid in matching)
        for album_id in matching:
            if have >= quota:
                break
            if album_id not in taken:
                plan.append((album_id, event))
                taken.add(album_id)
                have += 1
    return plan


def build_manifest(
    album_id: str, event_types: list[str], importance_rows: list[list], *, license_note: str
) -> dict:
    nsid = album_id.split("_", 1)[1]
    photos = []
    for index, (image_url, key, score) in enumerate(importance_rows, start=1):
        photo_id = key.split("/")[-1]
        photos.append({
            "index": index,
            "file": f"{index:03d}_{photo_id}.jpg",
            "flickr_photo_id": photo_id,
            "flickr_page": FLICKR_PAGE.format(nsid=nsid, photo_id=photo_id),
            "flickr_image_url": image_url,
            "importance_score": float(score),
            "license": license_note,
        })
    return {
        "source_dataset": SOURCE_NOTE,
        "album_id": album_id,
        "flickr_user_nsid": nsid,
        "event_types": list(event_types),
        "order": "image_importance.json order (chronological per original CUFED readme); index prefix preserves it",
        "collection_license_tag_on_mirror": "cc-by-nc-2.0 (per-photo CC variant must be checked)",
        "license_checked": "no" if license_note == "unchecked" else dt.date.today().isoformat(),
        "photos": photos,
    }


def check_licence(flickr_page: str) -> dict[str, str]:
    """Best-effort licence and title from the public Flickr page; empty fields when unknown."""
    try:
        html = httpx.get(flickr_page, follow_redirects=True, timeout=FLICKR_TIMEOUT_S,
                         headers={"User-Agent": "Mozilla/5.0"}).text
    except httpx.HTTPError as e:
        logger.warning("Flickr page fetch failed", extra={"page": flickr_page, "error": str(e)})
        return {"license": "unchecked (fetch failed)"}
    match = _CC_LICENCE_RE.search(html)
    licence = f"CC {match.group(1).upper()} {match.group(2)}" if match else (
        "All rights reserved" if '"license":0' in html else "unchecked (not found on page)")
    title = _OG_TITLE_RE.search(html)
    return {"license": licence, "title": title.group(1) if title else ""}


def existing_album_ids(data_dir: Path) -> set[str]:
    # Folder names are cufed_<Event>_<album_id>; album ids themselves contain underscores.
    return {d.name.split("_", 2)[2] for d in data_dir.glob("cufed_*") if d.is_dir()}


def fetch_album(album_id: str, event: str, importance_rows: list[list], data_dir: Path,
                *, event_types: list[str], check_licences: bool) -> Path:
    from huggingface_hub import hf_hub_download

    album_dir = data_dir / f"cufed_{event}_{album_id}"
    album_dir.mkdir(parents=True, exist_ok=True)
    manifest = build_manifest(album_id, event_types, importance_rows,
                              license_note="unchecked" if not check_licences else "pending")
    for photo in manifest["photos"]:
        cached = hf_hub_download(MIRROR_REPO, f"images/{album_id}/{photo['flickr_photo_id']}.jpg",
                                 repo_type="dataset")
        shutil.copyfile(cached, album_dir / photo["file"])
        if check_licences:
            photo.update(check_licence(photo["flickr_page"]))
    (album_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    logger.info("Fetched album", extra={"album": album_dir.name, "photos": len(manifest["photos"])})
    return album_dir


def parse_wanted(spec: str) -> dict[str, int]:
    """'Zoo=4,Wedding=4' -> {'Zoo': 4, 'Wedding': 4}."""
    wanted: dict[str, int] = {}
    for item in filter(None, spec.split(",")):
        event, _, count = item.partition("=")
        wanted[event.strip()] = int(count)
    return wanted


def main() -> None:
    from huggingface_hub import hf_hub_download

    parser = argparse.ArgumentParser(description="Fetch CUFED albums from the HF mirror")
    parser.add_argument("--wanted", required=True, help="Quota per event, e.g. 'Zoo=4,Wedding=4'")
    parser.add_argument("--data", default="data")
    parser.add_argument("--check-licences", action="store_true", help="Scrape each Flickr page for its licence")
    parser.add_argument("--dry-run", action="store_true", help="Only print the plan")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-8s %(name)s %(message)s")

    event_types = json.loads(Path(hf_hub_download(MIRROR_REPO, "event_type.json", repo_type="dataset")).read_text())
    importance = json.loads(Path(hf_hub_download(MIRROR_REPO, "image_importance.json", repo_type="dataset")).read_text())
    data_dir = Path(args.data)
    plan = plan_albums(event_types, parse_wanted(args.wanted), existing_album_ids(data_dir))
    for album_id, event in plan:
        print(f"{'would fetch' if args.dry_run else 'fetching'}: {event:12s} {album_id}  ({len(importance[album_id])} photos)")
    if args.dry_run:
        return
    for album_id, event in plan:
        fetch_album(album_id, event, importance[album_id], data_dir,
                    event_types=event_types[album_id], check_licences=args.check_licences)
    print(f"fetched {len(plan)} album(s) into {data_dir}")


if __name__ == "__main__":
    main()
