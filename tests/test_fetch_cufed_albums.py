"""Choosing which CUFED albums to fetch, and the manifest written for each."""

from __future__ import annotations

from scripts.fetch_cufed_albums import build_manifest, plan_albums


EVENT_TYPES = {
    "0_a@N00": ["Zoo"], "1_b@N00": ["Zoo"], "2_c@N00": ["Zoo"],
    "3_d@N00": ["Wedding"], "4_e@N00": ["Wedding", "GroupActivity"],
}


def test_plan_albums_fills_each_event_up_to_the_wanted_count_skipping_existing():
    # One Zoo and one Wedding album already exist, so each quota of 2 needs one more;
    # 4_e counts as a Wedding album even though Wedding is not its only event type.
    plan = plan_albums(EVENT_TYPES, wanted={"Zoo": 2, "Wedding": 2}, existing_ids={"0_a@N00", "3_d@N00"})

    assert plan == [("1_b@N00", "Zoo"), ("4_e@N00", "Wedding")]


def test_plan_albums_counts_existing_albums_toward_the_quota():
    plan = plan_albums(EVENT_TYPES, wanted={"Zoo": 2}, existing_ids={"0_a@N00"})

    assert plan == [("1_b@N00", "Zoo")]


def test_build_manifest_orders_photos_as_the_importance_index_does():
    importance = [
        ["http://farm/x/111_ab.jpg", "0_a@N00/111", 0.5],
        ["http://farm/x/222_cd.jpg", "0_a@N00/222", 1.4],
    ]

    manifest = build_manifest("0_a@N00", ["Zoo"], importance, license_note="unchecked")

    assert manifest["album_id"] == "0_a@N00"
    assert manifest["flickr_user_nsid"] == "a@N00"
    assert manifest["event_types"] == ["Zoo"]
    assert [p["file"] for p in manifest["photos"]] == ["001_111.jpg", "002_222.jpg"]
    assert manifest["photos"][1] == {
        "index": 2,
        "file": "002_222.jpg",
        "flickr_photo_id": "222",
        "flickr_page": "https://www.flickr.com/photos/a@N00/222",
        "flickr_image_url": "http://farm/x/222_cd.jpg",
        "importance_score": 1.4,
        "license": "unchecked",
    }


def test_plan_albums_never_lists_the_same_album_under_two_events():
    """An album tagged with two events counts toward the first quota it fills and is not
    fetched again under the second; it still counts toward the second quota."""
    event_types = {"0_a@N00": ["Zoo", "NatureTrip"], "1_b@N00": ["NatureTrip"], "2_c@N00": ["NatureTrip"]}

    plan = plan_albums(event_types, wanted={"Zoo": 1, "NatureTrip": 2}, existing_ids=set())

    assert plan == [("0_a@N00", "Zoo"), ("1_b@N00", "NatureTrip")]
