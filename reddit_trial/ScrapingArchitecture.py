"""
Architecture skeleton for an r/AmItheAsshole scraping project.

This file is based on the structure of the original main.py:
- requests / BeautifulSoup collection layer
- JSON and CSV output
- a main() entry point

The live scraping details are intentionally left as clearly marked placeholders.
Everything around the scraping layer is implemented:
- three independent stages
- command-line interface
- runtime limits
- automatic resume
- atomic saving
- judgement extraction
- update/edit detection
- unique-user reconstruction
- user activity queue
- validation of saved records

Stages:
    python main_architecture.py --stage threads
    python main_architecture.py --stage users
    python main_architecture.py --stage validation
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import statistics
import tempfile
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from tqdm import tqdm


# =============================================================================
# 1. PROJECT CONSTANTS
# =============================================================================

DEFAULT_SUBREDDIT = "AmItheAsshole"
DEFAULT_OUTPUT_DIRECTORY = "aita_scrape_output"

JUDGEMENT_LABELS = (
    "YWNBTA",
    "YWBTA",
    "NTA",
    "YTA",
    "ESH",
    "NAH",
    "INFO",
)

JUDGEMENT_PATTERN = re.compile(
    r"\b(" + "|".join(map(re.escape, JUDGEMENT_LABELS)) + r")\b",
    flags=re.IGNORECASE,
)

UPDATE_OR_EDIT_PATTERN = re.compile(
    r"(?im)^\s*(?:#{1,6}\s*)?"
    r"(FINAL\s+UPDATE|UPDATED|UPDATE|EDITED\s+TO\s+ADD|EDIT|ETA)"
    r"\s*(?::|-)"
)

EXCLUDED_JUDGEMENT_USERS = {
    "automoderator",
    "judgement_bot_aita",
    "aita_mod",
}


# =============================================================================
# 2. PATHS AND RUN CONFIGURATION
# =============================================================================

@dataclass(frozen=True)
class OutputPaths:
    output_directory: Path
    thread_records_directory: Path
    user_activity_records_directory: Path
    posts_csv_path: Path
    comments_csv_path: Path
    judgement_summary_csv_path: Path
    unique_users_json_path: Path
    unique_users_csv_path: Path
    user_activity_json_path: Path
    user_activity_csv_path: Path
    validation_summary_json_path: Path
    validation_by_thread_csv_path: Path
    run_state_json_path: Path


def build_output_paths(output_directory: str | Path) -> OutputPaths:
    """Create and return all output paths used by the project."""
    base_directory = Path(output_directory)

    paths = OutputPaths(
        output_directory=base_directory,
        thread_records_directory=base_directory / "thread_records",
        user_activity_records_directory=base_directory / "user_activity_records",
        posts_csv_path=base_directory / "posts.csv",
        comments_csv_path=base_directory / "comments.csv",
        judgement_summary_csv_path=base_directory / "judgement_summary.csv",
        unique_users_json_path=base_directory / "unique_users.json",
        unique_users_csv_path=base_directory / "unique_users.csv",
        user_activity_json_path=base_directory / "user_activity.json",
        user_activity_csv_path=base_directory / "user_activity.csv",
        validation_summary_json_path=base_directory / "scrape_validation_summary.json",
        validation_by_thread_csv_path=base_directory / "scrape_validation_by_thread.csv",
        run_state_json_path=base_directory / "run_state.json",
    )

    paths.output_directory.mkdir(parents=True, exist_ok=True)
    paths.thread_records_directory.mkdir(parents=True, exist_ok=True)
    paths.user_activity_records_directory.mkdir(parents=True, exist_ok=True)

    return paths


def parse_runtime_seconds(runtime_text: str | None) -> float | None:
    """
    Parse runtime values such as:
    - 10m
    - 2h
    - 90  -> interpreted as minutes
    """
    if runtime_text is None:
        return None

    cleaned = runtime_text.strip().lower()

    if cleaned.endswith("h"):
        return float(cleaned[:-1]) * 60 * 60

    if cleaned.endswith("m"):
        return float(cleaned[:-1]) * 60

    return float(cleaned) * 60


def utc_now_iso() -> str:
    """Return the current UTC time as an ISO-8601 string."""
    return datetime.now(timezone.utc).isoformat()


def runtime_limit_reached(
    stage_started_at_monotonic: float,
    maximum_runtime_seconds: float | None,
) -> bool:
    """Return True when the configured stage runtime has been reached."""
    if maximum_runtime_seconds is None:
        return False

    elapsed_seconds = time.monotonic() - stage_started_at_monotonic
    return elapsed_seconds >= maximum_runtime_seconds


# =============================================================================
# 3. ATOMIC FILE SAVING
# =============================================================================

def atomic_write_text(destination_path: Path, text: str) -> None:
    """
    Write text without risking replacement of a valid file by a partial file.

    The new file is written to a temporary location first and then moved into place.
    """
    destination_path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        delete=False,
        dir=destination_path.parent,
        suffix=".tmp",
    ) as temporary_file:
        temporary_file.write(text)
        temporary_file.flush()
        os.fsync(temporary_file.fileno())
        temporary_path = Path(temporary_file.name)

    if destination_path.exists():
        backup_path = destination_path.with_suffix(destination_path.suffix + ".bak")
        shutil.copy2(destination_path, backup_path)

    temporary_path.replace(destination_path)


def atomic_write_json(destination_path: Path, value: Any) -> None:
    """Serialize a Python value to JSON using an atomic replacement."""
    serialized = json.dumps(value, indent=2, ensure_ascii=False)
    atomic_write_text(destination_path, serialized)


def atomic_write_csv(
    destination_path: Path,
    rows: list[dict[str, Any]],
    fieldnames: list[str],
) -> None:
    """Write a CSV file atomically."""
    destination_path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.NamedTemporaryFile(
        mode="w",
        newline="",
        encoding="utf-8-sig",
        delete=False,
        dir=destination_path.parent,
        suffix=".tmp",
    ) as temporary_file:
        writer = csv.DictWriter(
            temporary_file,
            fieldnames=fieldnames,
            extrasaction="ignore",
        )
        writer.writeheader()

        for row in rows:
            csv_row = {
                fieldname: serialize_csv_value(row.get(fieldname))
                for fieldname in fieldnames
            }
            writer.writerow(csv_row)

        temporary_file.flush()
        os.fsync(temporary_file.fileno())
        temporary_path = Path(temporary_file.name)

    if destination_path.exists():
        backup_path = destination_path.with_suffix(destination_path.suffix + ".bak")
        shutil.copy2(destination_path, backup_path)

    temporary_path.replace(destination_path)


def serialize_csv_value(value: Any) -> Any:
    """Serialize nested values as JSON strings inside CSV cells."""
    if isinstance(value, (dict, list, tuple, set)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return value


# =============================================================================
# 4. LOADING SAVED RECORDS AND AUTOMATIC RESUME
# =============================================================================

def load_json_file(json_path: Path, default_value: Any) -> Any:
    """Load JSON safely and return a default value when the file is missing."""
    if not json_path.exists():
        return default_value

    try:
        with json_path.open("r", encoding="utf-8") as file:
            return json.load(file)
    except (json.JSONDecodeError, OSError):
        return default_value


def load_thread_records(paths: OutputPaths) -> list[dict[str, Any]]:
    """Load every completed thread record from the durable record directory."""
    thread_records: list[dict[str, Any]] = []

    for record_path in sorted(paths.thread_records_directory.glob("*.json")):
        record = load_json_file(record_path, default_value=None)
        if isinstance(record, dict):
            thread_records.append(record)

    return thread_records


def load_completed_thread_ids(paths: OutputPaths) -> set[str]:
    """Return IDs of all thread records that were already saved."""
    return {
        str(record.get("post_id"))
        for record in load_thread_records(paths)
        if record.get("post_id")
    }


def load_user_activity_records(paths: OutputPaths) -> list[dict[str, Any]]:
    """Load all completed user activity records."""
    activity_records: list[dict[str, Any]] = []

    for record_path in sorted(paths.user_activity_records_directory.glob("*.json")):
        record = load_json_file(record_path, default_value=None)
        if isinstance(record, dict):
            activity_records.append(record)

    return activity_records


def load_completed_usernames(paths: OutputPaths) -> set[str]:
    """Return usernames whose activity record was already saved."""
    return {
        str(record.get("user_username")).lower()
        for record in load_user_activity_records(paths)
        if record.get("user_username")
    }


# =============================================================================
# 5. JUDGEMENT EXTRACTION
# =============================================================================

def remove_quoted_and_code_text(comment_text: str) -> str:
    """
    Remove quoted Markdown lines and fenced code blocks before judgement detection.
    """
    cleaned_lines: list[str] = []
    inside_code_block = False

    for line in comment_text.splitlines():
        stripped_line = line.lstrip()

        if stripped_line.startswith("```"):
            inside_code_block = not inside_code_block
            continue

        if inside_code_block:
            continue

        if stripped_line.startswith(">"):
            continue

        cleaned_lines.append(line)

    return "\n".join(cleaned_lines)


def extract_comment_judgements(comment_text: str) -> list[str]:
    """Return unique AITA judgement labels in the order they appear."""
    cleaned_text = remove_quoted_and_code_text(comment_text)

    detected_judgements: list[str] = []
    already_seen: set[str] = set()

    for match in JUDGEMENT_PATTERN.finditer(cleaned_text):
        judgement = match.group(1).upper()

        if judgement not in already_seen:
            already_seen.add(judgement)
            detected_judgements.append(judgement)

    return detected_judgements


def detect_post_update_or_edit(
    post_title: str,
    post_text: str,
) -> tuple[bool, list[str]]:
    """Detect explicit update or edit headings in a post."""
    combined_text = f"{post_title}\n{post_text}"

    detected_markers = [
        match.group(1).upper()
        for match in UPDATE_OR_EDIT_PATTERN.finditer(combined_text)
    ]

    unique_markers = list(dict.fromkeys(detected_markers))
    return bool(unique_markers), unique_markers


# =============================================================================
# 6. COMMUNITY JUDGEMENT CALCULATION
# =============================================================================

def build_judgement_result(judgement_counts: Counter[str]) -> dict[str, Any]:
    """Create counts, percentages, winner, and tie information."""
    total_votes = sum(judgement_counts.values())

    complete_counts = {
        judgement: int(judgement_counts.get(judgement, 0))
        for judgement in JUDGEMENT_LABELS
    }

    if total_votes == 0:
        return {
            "total_votes": 0,
            "judgement_counts": complete_counts,
            "judgement_percentages": {
                judgement: 0.0 for judgement in JUDGEMENT_LABELS
            },
            "winning_judgement": None,
            "is_tie": False,
            "tied_judgements": [],
        }

    highest_vote_count = max(complete_counts.values())
    tied_judgements = [
        judgement
        for judgement, count in complete_counts.items()
        if count == highest_vote_count
    ]

    return {
        "total_votes": total_votes,
        "judgement_counts": complete_counts,
        "judgement_percentages": {
            judgement: round(count / total_votes * 100, 4)
            for judgement, count in complete_counts.items()
        },
        "winning_judgement": (
            tied_judgements[0] if len(tied_judgements) == 1 else "TIE"
        ),
        "is_tie": len(tied_judgements) > 1,
        "tied_judgements": (
            tied_judgements if len(tied_judgements) > 1 else []
        ),
    }


def calculate_community_judgement(
    comments: list[dict[str, Any]],
) -> dict[str, Any]:
    """Calculate both comment-level and unique-user-level community judgements."""
    usable_judgement_comments = [
        comment
        for comment in comments
        if comment.get("comment_is_usable")
        and comment.get("comment_primary_judgement")
        and comment.get("comment_author_username")
        and comment["comment_author_username"].lower()
        not in EXCLUDED_JUDGEMENT_USERS
    ]

    comment_level_counts: Counter[str] = Counter(
        comment["comment_primary_judgement"]
        for comment in usable_judgement_comments
    )

    judgements_by_user: dict[str, list[str]] = defaultdict(list)

    for comment in usable_judgement_comments:
        username = comment["comment_author_username"]
        judgement = comment["comment_primary_judgement"]
        judgements_by_user[username].append(judgement)

    one_judgement_per_user: dict[str, str] = {}

    for username, user_judgements in judgements_by_user.items():
        user_counts = Counter(user_judgements)
        first_position = {
            judgement: user_judgements.index(judgement)
            for judgement in user_counts
        }

        selected_judgement = max(
            user_counts,
            key=lambda judgement: (
                user_counts[judgement],
                -first_position[judgement],
            ),
        )
        one_judgement_per_user[username] = selected_judgement

    unique_user_counts = Counter(one_judgement_per_user.values())

    return {
        "comment_level_judgement": build_judgement_result(comment_level_counts),
        "unique_user_level_judgement": build_judgement_result(unique_user_counts),
    }


# =============================================================================
# 7. RECORD SCHEMAS
# =============================================================================

def create_empty_thread_record(post_url: str) -> dict[str, Any]:
    """
    Create a clearly named thread record.

    The loading and HTML parsing functions should fill this structure.
    """
    return {
        "post_id": None,
        "post_url": post_url,
        "post_title": None,
        "post_author_username": None,
        "post_text": None,
        "post_created_at": None,
        "post_score": None,
        "post_upvote_ratio": None,
        "post_reported_comment_count": None,
        "post_scraped_comment_count": 0,
        "post_unique_user_count": 0,
        "post_flair_text": None,
        "post_contains_update_or_edit": False,
        "post_update_or_edit_markers": [],
        "post_scrape_status": "not_started",
        "post_scrape_error": None,
        "comments": [],
        "community_judgement_summary": {},
    }


def create_empty_comment_record() -> dict[str, Any]:
    """Create the expected structure for one parsed comment."""
    return {
        "comment_id": None,
        "parent_comment_id": None,
        "comment_depth": None,
        "comment_author_username": None,
        "comment_text": None,
        "comment_created_at": None,
        "comment_score": None,
        "comment_is_original_poster": False,
        "comment_author_flair_text": None,
        "comment_scrape_status": "not_started",
        "comment_is_usable": False,
        "comment_detected_judgements": [],
        "comment_primary_judgement": None,
        "comment_contains_multiple_judgements": False,
    }


def create_empty_user_activity_record(username: str) -> dict[str, Any]:
    """Create the expected structure for one user's sampled activity."""
    return {
        "user_username": username,
        "user_profile_url": f"https://www.reddit.com/user/{username}/",
        "user_link_karma": None,
        "user_comment_karma": None,
        "user_total_karma": None,
        "user_sampled_post_count": None,
        "user_sampled_comment_count": None,
        "user_sampled_aita_post_count": None,
        "user_sampled_aita_comment_count": None,
        "user_sampled_aita_post_percentage": None,
        "user_sampled_aita_comment_percentage": None,
        "user_history_sample_limit": None,
        "user_post_history_may_be_incomplete": None,
        "user_comment_history_may_be_incomplete": None,
        "user_activity_scrape_status": "not_started",
        "user_activity_scrape_error": None,
    }


# =============================================================================
# 8. LIVE PAGE LOADING PLACEHOLDERS
# =============================================================================

def discover_post_urls_from_subreddit(
    subreddit_name: str,
    post_limit: int,
) -> list[str]:
    """
    PLACEHOLDER FOR THE SUBREDDIT LISTING SCRAPER.

    Loading the subreddit listing page goes here.

    Expected future steps:
    1. Request the r/AmItheAsshole listing page.
    2. Parse the returned page with BeautifulSoup.
    3. Find links containing '/comments/'.
    4. Convert relative URLs into absolute URLs.
    5. Remove duplicate URLs.
    6. Return at most post_limit URLs.

    This architecture file intentionally performs no network request.
    """
    tqdm.write(
        "Thread URL discovery is a placeholder. "
        "Add the listing-page loading and parsing logic here."
    )
    return []


def load_and_parse_thread_page(post_url: str) -> dict[str, Any] | None:
    """
    PLACEHOLDER FOR LOADING AND PARSING ONE DISCUSSION THREAD.

    Loading the post page goes here.

    Expected future steps:
    1. Request the post URL.
    2. Validate the HTTP response.
    3. Create BeautifulSoup from the returned HTML.
    4. Extract the post ID, title, author, text, score, ratio, and flair.
    5. Find comment containers.
    6. Parse comment text, author, score, parent ID, and depth.
    7. Mark unusable comments.
    8. Run extract_comment_judgements() on usable comments.
    9. Detect updates with detect_post_update_or_edit().
    10. Calculate community judgement.
    11. Return the complete thread record.

    Return None when no valid record can be produced.
    """
    tqdm.write(
        f"Thread page loading is a placeholder for: {post_url}"
    )
    return None


def load_and_parse_user_activity(
    username: str,
    history_sample_limit: int,
) -> dict[str, Any] | None:
    """
    PLACEHOLDER FOR LOADING AND PARSING A USER'S PUBLIC ACTIVITY.

    Loading the user profile and activity pages goes here.

    Expected future steps:
    1. Load the user's profile page.
    2. Parse visible karma fields.
    3. Load a limited sample of the user's posts.
    4. Load a limited sample of the user's comments.
    5. Count total sampled items.
    6. Count sampled items from r/AmItheAsshole.
    7. Calculate AITA percentages.
    8. Set incomplete-history flags when the sample limit is reached.
    9. Return the user activity record.

    Return None when no valid activity record can be produced.
    """
    tqdm.write(
        f"User activity loading is a placeholder for: {username}"
    )
    return None


# =============================================================================
# 9. THREAD RECORD SAVING AND AGGREGATE OUTPUTS
# =============================================================================

def save_thread_record(
    thread_record: dict[str, Any],
    paths: OutputPaths,
) -> None:
    """Save one completed thread immediately."""
    post_id = thread_record.get("post_id")

    if not post_id:
        raise ValueError("Cannot save a thread record without post_id.")

    record_path = paths.thread_records_directory / f"{post_id}.json"
    atomic_write_json(record_path, thread_record)


def rebuild_thread_outputs(paths: OutputPaths) -> None:
    """Rebuild posts, comments, judgements, and unique-user outputs."""
    thread_records = load_thread_records(paths)

    post_rows: list[dict[str, Any]] = []
    comment_rows: list[dict[str, Any]] = []
    judgement_rows: list[dict[str, Any]] = []

    for thread_record in thread_records:
        post_row = {
            key: value
            for key, value in thread_record.items()
            if key not in {"comments", "community_judgement_summary"}
        }
        post_rows.append(post_row)

        post_id = thread_record.get("post_id")

        for comment in thread_record.get("comments", []):
            comment_rows.append(
                {
                    "post_id": post_id,
                    **comment,
                }
            )

        judgement_summary = thread_record.get(
            "community_judgement_summary",
            {},
        )

        for summary_name, summary_value in judgement_summary.items():
            judgement_rows.append(
                {
                    "post_id": post_id,
                    "judgement_summary_type": summary_name,
                    **summary_value,
                }
            )

    atomic_write_csv(
        paths.posts_csv_path,
        post_rows,
        fieldnames=[
            "post_id",
            "post_url",
            "post_title",
            "post_author_username",
            "post_text",
            "post_created_at",
            "post_score",
            "post_upvote_ratio",
            "post_reported_comment_count",
            "post_scraped_comment_count",
            "post_unique_user_count",
            "post_flair_text",
            "post_contains_update_or_edit",
            "post_update_or_edit_markers",
            "post_scrape_status",
            "post_scrape_error",
        ],
    )

    atomic_write_csv(
        paths.comments_csv_path,
        comment_rows,
        fieldnames=[
            "post_id",
            "comment_id",
            "parent_comment_id",
            "comment_depth",
            "comment_author_username",
            "comment_text",
            "comment_created_at",
            "comment_score",
            "comment_is_original_poster",
            "comment_author_flair_text",
            "comment_scrape_status",
            "comment_is_usable",
            "comment_detected_judgements",
            "comment_primary_judgement",
            "comment_contains_multiple_judgements",
        ],
    )

    atomic_write_csv(
        paths.judgement_summary_csv_path,
        judgement_rows,
        fieldnames=[
            "post_id",
            "judgement_summary_type",
            "total_votes",
            "judgement_counts",
            "judgement_percentages",
            "winning_judgement",
            "is_tie",
            "tied_judgements",
        ],
    )

    unique_users = build_unique_user_records(thread_records)
    atomic_write_json(paths.unique_users_json_path, unique_users)

    atomic_write_csv(
        paths.unique_users_csv_path,
        unique_users,
        fieldnames=[
            "user_username",
            "user_profile_url",
            "user_is_original_poster",
            "user_original_post_ids",
            "user_thread_ids",
            "user_comment_count_in_scraped_threads",
            "user_judgement_comment_count_in_scraped_threads",
            "user_observed_aita_flairs",
            "user_activity_scrape_completed",
            "user_activity_scrape_status",
            "user_activity_scrape_error",
        ],
    )


# =============================================================================
# 10. UNIQUE-USER RECONSTRUCTION
# =============================================================================

def build_unique_user_records(
    thread_records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Create one aggregated record per unique username."""
    users_by_lowercase_username: dict[str, dict[str, Any]] = {}

    for thread_record in thread_records:
        post_id = thread_record.get("post_id")
        post_author_username = thread_record.get("post_author_username")

        if post_author_username:
            lowercase_username = post_author_username.lower()

            user_record = users_by_lowercase_username.setdefault(
                lowercase_username,
                create_unique_user_record(post_author_username),
            )

            user_record["user_is_original_poster"] = True
            add_unique_value(
                user_record["user_original_post_ids"],
                post_id,
            )
            add_unique_value(
                user_record["user_thread_ids"],
                post_id,
            )

        for comment in thread_record.get("comments", []):
            username = comment.get("comment_author_username")

            if not username:
                continue

            lowercase_username = username.lower()

            user_record = users_by_lowercase_username.setdefault(
                lowercase_username,
                create_unique_user_record(username),
            )

            add_unique_value(
                user_record["user_thread_ids"],
                post_id,
            )

            user_record["user_comment_count_in_scraped_threads"] += 1

            if comment.get("comment_primary_judgement"):
                user_record[
                    "user_judgement_comment_count_in_scraped_threads"
                ] += 1

            observed_flair = comment.get("comment_author_flair_text")
            if observed_flair:
                add_unique_value(
                    user_record["user_observed_aita_flairs"],
                    observed_flair,
                )

    return sorted(
        users_by_lowercase_username.values(),
        key=lambda record: record["user_username"].lower(),
    )


def create_unique_user_record(username: str) -> dict[str, Any]:
    """Create one clearly named unique-user record."""
    return {
        "user_username": username,
        "user_profile_url": f"https://www.reddit.com/user/{username}/",
        "user_is_original_poster": False,
        "user_original_post_ids": [],
        "user_thread_ids": [],
        "user_comment_count_in_scraped_threads": 0,
        "user_judgement_comment_count_in_scraped_threads": 0,
        "user_observed_aita_flairs": [],
        "user_activity_scrape_completed": False,
        "user_activity_scrape_status": "not_started",
        "user_activity_scrape_error": None,
    }


def add_unique_value(values: list[Any], value: Any) -> None:
    """Append a value only when it is not already present."""
    if value is not None and value not in values:
        values.append(value)


# =============================================================================
# 11. USER ACTIVITY RECORD SAVING AND OUTPUTS
# =============================================================================

def save_user_activity_record(
    user_activity_record: dict[str, Any],
    paths: OutputPaths,
) -> None:
    """Save one user's completed activity record."""
    username = user_activity_record.get("user_username")

    if not username:
        raise ValueError("Cannot save user activity without user_username.")

    safe_filename = re.sub(r"[^A-Za-z0-9_.-]", "_", username)
    record_path = paths.user_activity_records_directory / f"{safe_filename}.json"
    atomic_write_json(record_path, user_activity_record)


def rebuild_user_activity_outputs(paths: OutputPaths) -> None:
    """Rebuild combined JSON and CSV user activity outputs."""
    activity_records = load_user_activity_records(paths)

    atomic_write_json(paths.user_activity_json_path, activity_records)

    atomic_write_csv(
        paths.user_activity_csv_path,
        activity_records,
        fieldnames=[
            "user_username",
            "user_profile_url",
            "user_link_karma",
            "user_comment_karma",
            "user_total_karma",
            "user_sampled_post_count",
            "user_sampled_comment_count",
            "user_sampled_aita_post_count",
            "user_sampled_aita_comment_count",
            "user_sampled_aita_post_percentage",
            "user_sampled_aita_comment_percentage",
            "user_history_sample_limit",
            "user_post_history_may_be_incomplete",
            "user_comment_history_may_be_incomplete",
            "user_activity_scrape_status",
            "user_activity_scrape_error",
        ],
    )


# =============================================================================
# 12. STOP INPUT AND WAITING
# =============================================================================

def wait_with_optional_stop(
    delay_seconds: float,
    prompt_text: str,
) -> bool:
    """
    Wait for a short interval.

    Returns True when the user requests a stop.

    This basic architecture uses a simple timed loop. The operating-system-specific
    non-blocking input implementation should be added here.
    """
    tqdm.write(prompt_text)

    # TODO: Add non-blocking console input here.
    # Expected accepted commands: q, quit, stop.
    # No input should allow the timer to expire normally.

    wait_started_at = time.monotonic()

    while time.monotonic() - wait_started_at < delay_seconds:
        time.sleep(min(0.25, delay_seconds))

    return False


# =============================================================================
# 13. STAGE 1: THREAD SCRAPING ORCHESTRATION
# =============================================================================

def run_thread_scraping_stage(
    arguments: argparse.Namespace,
    paths: OutputPaths,
) -> None:
    """
    Orchestrate thread collection.

    The orchestration, saving, resume, judgement, and output layers are implemented.
    Only page loading and BeautifulSoup parsing remain placeholders.
    """
    stage_started_at = time.monotonic()
    maximum_runtime_seconds = parse_runtime_seconds(arguments.max_runtime)

    completed_thread_ids = load_completed_thread_ids(paths)

    if arguments.post_url:
        candidate_post_urls = list(dict.fromkeys(arguments.post_url))
    else:
        candidate_post_urls = discover_post_urls_from_subreddit(
            subreddit_name=arguments.subreddit,
            post_limit=arguments.post_limit,
        )

    posts_saved_this_run = 0

    for post_url in tqdm(
        candidate_post_urls,
        desc="Thread scraping stage",
        unit="thread",
    ):
        if runtime_limit_reached(
            stage_started_at,
            maximum_runtime_seconds,
        ):
            tqdm.write("Maximum runtime reached.")
            break

        if (
            arguments.max_posts is not None
            and posts_saved_this_run >= arguments.max_posts
        ):
            tqdm.write("Maximum post count reached.")
            break

        # TODO: Extract a post ID from the URL before requesting the page.
        # Then skip it when it already exists in completed_thread_ids.

        thread_record = load_and_parse_thread_page(post_url)

        if thread_record is None:
            tqdm.write(
                f"No thread record was produced for {post_url}. "
                "Nothing was saved."
            )
            continue

        post_id = thread_record.get("post_id")

        if not post_id:
            tqdm.write(
                f"Skipping record without a post ID: {post_url}"
            )
            continue

        if str(post_id) in completed_thread_ids:
            tqdm.write(f"Skipping completed post: {post_id}")
            continue

        comments = thread_record.get("comments", [])

        thread_record["post_scraped_comment_count"] = len(comments)
        thread_record["post_unique_user_count"] = len(
            {
                comment.get("comment_author_username")
                for comment in comments
                if comment.get("comment_author_username")
            }
        )

        update_found, update_markers = detect_post_update_or_edit(
            thread_record.get("post_title") or "",
            thread_record.get("post_text") or "",
        )

        thread_record["post_contains_update_or_edit"] = update_found
        thread_record["post_update_or_edit_markers"] = update_markers
        thread_record["community_judgement_summary"] = (
            calculate_community_judgement(comments)
        )

        save_thread_record(thread_record, paths)
        completed_thread_ids.add(str(post_id))
        posts_saved_this_run += 1

        rebuild_thread_outputs(paths)

        update_run_state(
            paths=paths,
            stage_name="threads",
            status="running",
            completed_items_this_run=posts_saved_this_run,
        )

        should_stop = wait_with_optional_stop(
            delay_seconds=arguments.between_post_delay,
            prompt_text=(
                "Type q, quit, or stop during the wait to finish safely. "
                "No input means continue."
            ),
        )

        if should_stop:
            tqdm.write("Manual stop requested.")
            break

    rebuild_thread_outputs(paths)

    update_run_state(
        paths=paths,
        stage_name="threads",
        status="finished",
        completed_items_this_run=posts_saved_this_run,
    )


# =============================================================================
# 14. STAGE 2: USER ACTIVITY ORCHESTRATION
# =============================================================================

def run_user_activity_stage(
    arguments: argparse.Namespace,
    paths: OutputPaths,
) -> None:
    """Process unfinished users from the unique-user queue."""
    stage_started_at = time.monotonic()
    maximum_runtime_seconds = parse_runtime_seconds(arguments.max_runtime)

    unique_users = load_json_file(
        paths.unique_users_json_path,
        default_value=[],
    )

    if not unique_users:
        tqdm.write(
            "No unique-user queue was found. "
            "Run the thread stage first."
        )
        return

    completed_usernames = load_completed_usernames(paths)
    users_saved_this_run = 0

    unfinished_users = [
        user_record
        for user_record in unique_users
        if str(user_record.get("user_username", "")).lower()
        not in completed_usernames
    ]

    for unique_user_record in tqdm(
        unfinished_users,
        desc="User activity stage",
        unit="user",
    ):
        if runtime_limit_reached(
            stage_started_at,
            maximum_runtime_seconds,
        ):
            tqdm.write("Maximum runtime reached.")
            break

        if (
            arguments.max_users is not None
            and users_saved_this_run >= arguments.max_users
        ):
            tqdm.write("Maximum user count reached.")
            break

        username = unique_user_record.get("user_username")

        if not username:
            continue

        user_activity_record = load_and_parse_user_activity(
            username=username,
            history_sample_limit=arguments.user_history_limit,
        )

        if user_activity_record is None:
            tqdm.write(
                f"No user activity record was produced for {username}. "
                "Nothing was saved."
            )
            continue

        save_user_activity_record(user_activity_record, paths)
        completed_usernames.add(username.lower())
        users_saved_this_run += 1

        rebuild_user_activity_outputs(paths)

        update_run_state(
            paths=paths,
            stage_name="users",
            status="running",
            completed_items_this_run=users_saved_this_run,
        )

        should_stop = wait_with_optional_stop(
            delay_seconds=arguments.between_user_delay,
            prompt_text=(
                "Type q, quit, or stop during the wait to finish safely. "
                "No input means continue."
            ),
        )

        if should_stop:
            tqdm.write("Manual stop requested.")
            break

    rebuild_user_activity_outputs(paths)

    update_run_state(
        paths=paths,
        stage_name="users",
        status="finished",
        completed_items_this_run=users_saved_this_run,
    )


# =============================================================================
# 15. STAGE 3: VALIDATION
# =============================================================================

def classify_saved_thread(
    thread_record: dict[str, Any],
) -> dict[str, Any]:
    """Classify one saved thread and calculate its usable-comment metrics."""
    post_url = str(thread_record.get("post_url") or "")
    post_title = str(thread_record.get("post_title") or "").strip()
    comments = thread_record.get("comments", [])

    usable_comments = [
        comment
        for comment in comments
        if comment.get("comment_is_usable")
    ]

    judgement_comments = [
        comment
        for comment in usable_comments
        if comment.get("comment_primary_judgement")
    ]

    reported_comment_count = thread_record.get(
        "post_reported_comment_count"
    )
    scraped_comment_count = len(comments)

    comment_coverage_percentage: float | None = None

    if (
        isinstance(reported_comment_count, (int, float))
        and reported_comment_count > 0
    ):
        comment_coverage_percentage = round(
            scraped_comment_count
            / reported_comment_count
            * 100,
            4,
        )

    looks_like_reddit_url = (
        "reddit.com" in post_url.lower()
        and "/comments/" in post_url.lower()
    )

    scrape_status = str(
        thread_record.get("post_scrape_status") or ""
    ).lower()

    if "blocked" in scrape_status:
        validation_classification = "blocked_response"
    elif "login" in scrape_status:
        validation_classification = "login_page"
    elif "captcha" in scrape_status:
        validation_classification = "captcha_page"
    elif "rate" in scrape_status:
        validation_classification = "rate_limited_response"
    elif "malformed" in scrape_status:
        validation_classification = "malformed_content"
    elif not looks_like_reddit_url:
        validation_classification = "non_reddit_content"
    elif not post_title:
        validation_classification = "unknown_junk"
    elif not comments:
        validation_classification = "reddit_post_without_usable_comments"
    elif usable_comments and comment_coverage_percentage is not None and comment_coverage_percentage < 80:
        validation_classification = "partial_reddit_thread"
    elif usable_comments:
        validation_classification = "usable_reddit_thread"
    else:
        validation_classification = "reddit_post_without_usable_comments"

    return {
        "post_id": thread_record.get("post_id"),
        "post_url": post_url,
        "post_title": post_title,
        "validation_classification": validation_classification,
        "validation_is_usable_thread": (
            validation_classification == "usable_reddit_thread"
        ),
        "validation_is_partial_thread": (
            validation_classification == "partial_reddit_thread"
        ),
        "validation_scraped_comment_count": scraped_comment_count,
        "validation_usable_comment_count": len(usable_comments),
        "validation_judgement_comment_count": len(judgement_comments),
        "validation_reported_comment_count": reported_comment_count,
        "validation_comment_coverage_percentage": comment_coverage_percentage,
    }


def run_validation_stage(
    arguments: argparse.Namespace,
    paths: OutputPaths,
) -> None:
    """Validate saved thread records without any network requests."""
    del arguments  # This stage currently needs only the shared output directory.

    thread_records = load_thread_records(paths)

    validation_rows = [
        classify_saved_thread(thread_record)
        for thread_record in tqdm(
            thread_records,
            desc="Validation stage",
            unit="thread",
        )
    ]

    total_thread_records = len(validation_rows)

    usable_thread_count = sum(
        row["validation_is_usable_thread"]
        for row in validation_rows
    )

    partial_thread_count = sum(
        row["validation_is_partial_thread"]
        for row in validation_rows
    )

    empty_or_junk_count = sum(
        row["validation_classification"]
        in {
            "reddit_post_without_usable_comments",
            "blocked_response",
            "login_page",
            "captcha_page",
            "rate_limited_response",
            "malformed_content",
            "non_reddit_content",
            "unknown_junk",
        }
        for row in validation_rows
    )

    total_scraped_comments = sum(
        row["validation_scraped_comment_count"]
        for row in validation_rows
    )

    total_usable_comments = sum(
        row["validation_usable_comment_count"]
        for row in validation_rows
    )

    total_judgement_comments = sum(
        row["validation_judgement_comment_count"]
        for row in validation_rows
    )

    coverage_values = [
        row["validation_comment_coverage_percentage"]
        for row in validation_rows
        if row["validation_comment_coverage_percentage"] is not None
    ]

    validation_summary = {
        "validation_total_thread_records": total_thread_records,
        "validation_usable_thread_count": usable_thread_count,
        "validation_usable_thread_percentage": safe_percentage(
            usable_thread_count,
            total_thread_records,
        ),
        "validation_partial_thread_count": partial_thread_count,
        "validation_partial_thread_percentage": safe_percentage(
            partial_thread_count,
            total_thread_records,
        ),
        "validation_empty_or_junk_count": empty_or_junk_count,
        "validation_empty_or_junk_percentage": safe_percentage(
            empty_or_junk_count,
            total_thread_records,
        ),
        "validation_total_scraped_comments": total_scraped_comments,
        "validation_usable_comment_count": total_usable_comments,
        "validation_usable_comment_percentage": safe_percentage(
            total_usable_comments,
            total_scraped_comments,
        ),
        "validation_judgement_comment_count": total_judgement_comments,
        "validation_judgement_comment_percentage": safe_percentage(
            total_judgement_comments,
            total_usable_comments,
        ),
        "validation_threads_with_zero_usable_comments": sum(
            row["validation_usable_comment_count"] == 0
            for row in validation_rows
        ),
        "validation_average_comment_coverage": (
            round(statistics.mean(coverage_values), 4)
            if coverage_values
            else None
        ),
        "validation_median_comment_coverage": (
            round(statistics.median(coverage_values), 4)
            if coverage_values
            else None
        ),
        "validation_completed_at_utc": utc_now_iso(),
    }

    atomic_write_json(
        paths.validation_summary_json_path,
        validation_summary,
    )

    atomic_write_csv(
        paths.validation_by_thread_csv_path,
        validation_rows,
        fieldnames=[
            "post_id",
            "post_url",
            "post_title",
            "validation_classification",
            "validation_is_usable_thread",
            "validation_is_partial_thread",
            "validation_scraped_comment_count",
            "validation_usable_comment_count",
            "validation_judgement_comment_count",
            "validation_reported_comment_count",
            "validation_comment_coverage_percentage",
        ],
    )

    print_validation_summary(validation_summary)


def safe_percentage(numerator: int | float, denominator: int | float) -> float:
    """Calculate a percentage without dividing by zero."""
    if denominator == 0:
        return 0.0
    return round(numerator / denominator * 100, 4)


def print_validation_summary(validation_summary: dict[str, Any]) -> None:
    """Print the main validation results."""
    print("\nValidation summary")
    print("-" * 60)
    print(
        "Total thread records:",
        validation_summary["validation_total_thread_records"],
    )
    print(
        "Usable Reddit threads:",
        validation_summary["validation_usable_thread_count"],
        f"({validation_summary['validation_usable_thread_percentage']}%)",
    )
    print(
        "Partial Reddit threads:",
        validation_summary["validation_partial_thread_count"],
        f"({validation_summary['validation_partial_thread_percentage']}%)",
    )
    print(
        "Empty or junk records:",
        validation_summary["validation_empty_or_junk_count"],
        f"({validation_summary['validation_empty_or_junk_percentage']}%)",
    )
    print(
        "Usable comments:",
        validation_summary["validation_usable_comment_count"],
        f"({validation_summary['validation_usable_comment_percentage']}%)",
    )
    print(
        "Judgement-bearing comments:",
        validation_summary["validation_judgement_comment_count"],
        f"({validation_summary['validation_judgement_comment_percentage']}%)",
    )
    print(
        "Average comment coverage:",
        validation_summary["validation_average_comment_coverage"],
    )
    print(
        "Median comment coverage:",
        validation_summary["validation_median_comment_coverage"],
    )


# =============================================================================
# 16. RUN STATE
# =============================================================================

def update_run_state(
    paths: OutputPaths,
    stage_name: str,
    status: str,
    completed_items_this_run: int,
) -> None:
    """Save a concise run-state checkpoint."""
    run_state = {
        "last_stage_name": stage_name,
        "last_stage_status": status,
        "completed_items_this_run": completed_items_this_run,
        "updated_at_utc": utc_now_iso(),
    }
    atomic_write_json(paths.run_state_json_path, run_state)


# =============================================================================
# 17. COMMAND-LINE INTERFACE
# =============================================================================

def parse_arguments() -> argparse.Namespace:
    """Build the command-line interface."""
    parser = argparse.ArgumentParser(
        description=(
            "Architecture skeleton for AITA thread scraping, "
            "user activity scraping, and validation."
        )
    )

    parser.add_argument(
        "--stage",
        choices=("threads", "users", "validation"),
        required=True,
    )
    parser.add_argument(
        "--post-url",
        action="append",
        help="Explicit post URL. Repeat for multiple URLs.",
    )
    parser.add_argument(
        "--subreddit",
        default=DEFAULT_SUBREDDIT,
    )
    parser.add_argument(
        "--post-limit",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--max-posts",
        type=int,
    )
    parser.add_argument(
        "--max-users",
        type=int,
    )
    parser.add_argument(
        "--max-runtime",
        help="Examples: 10m, 90m, 2h. Bare numbers mean minutes.",
    )
    parser.add_argument(
        "--user-history-limit",
        type=int,
        default=200,
    )
    parser.add_argument(
        "--between-post-delay",
        type=float,
        default=15.0,
    )
    parser.add_argument(
        "--between-user-delay",
        type=float,
        default=5.0,
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIRECTORY,
    )

    arguments = parser.parse_args()

    if arguments.post_limit < 1:
        parser.error("--post-limit must be at least 1")

    if arguments.user_history_limit < 1:
        parser.error("--user-history-limit must be at least 1")

    if arguments.between_post_delay < 0:
        parser.error("--between-post-delay cannot be negative")

    if arguments.between_user_delay < 0:
        parser.error("--between-user-delay cannot be negative")

    return arguments


# =============================================================================
# 18. MAIN ENTRY POINT
# =============================================================================

def main() -> None:
    """Run the requested project stage."""
    arguments = parse_arguments()
    paths = build_output_paths(arguments.output_dir)

    try:
        if arguments.stage == "threads":
            run_thread_scraping_stage(arguments, paths)

        elif arguments.stage == "users":
            run_user_activity_stage(arguments, paths)

        elif arguments.stage == "validation":
            run_validation_stage(arguments, paths)

    except KeyboardInterrupt:
        tqdm.write("\nCtrl+C received. Rebuilding saved outputs before exit.")

        rebuild_thread_outputs(paths)
        rebuild_user_activity_outputs(paths)

        update_run_state(
            paths=paths,
            stage_name=arguments.stage,
            status="interrupted",
            completed_items_this_run=0,
        )

    print(f"\nOutput directory: {paths.output_directory.resolve()}")


if __name__ == "__main__":
    main()
