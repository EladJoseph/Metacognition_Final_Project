"""
AmItheAsshole Reddit research scraper with two separate, resumable stages.

Install:
    pip install -r requirements_reddit_scraper.txt

Set these environment variables for a Reddit "script" API application:
    REDDIT_CLIENT_ID
    REDDIT_CLIENT_SECRET
    REDDIT_USER_AGENT

Recommended workflow:

    # Stage 1: collect posts, complete comment trees, judgements, flair, and
    # a unique queue of users. This is the default and most important stage.
    python "Reddit Scraping.py" --stage threads --max-runtime 2h

    # Stage 2: later enrich the saved unique-user queue with karma and sampled
    # posting/commenting activity. It does not re-download the threads.
    python "Reddit Scraping.py" --stage users --max-runtime 2h \
        --user-history-limit 200

    # Scrape one or more explicit posts in the thread stage.
    python "Reddit Scraping.py" --stage threads --post-url \
        "https://www.reddit.com/r/AmItheAsshole/comments/POST_ID/example/"

Persistence and resume behavior:
- Resume is automatic. If output-dir already contains data, it is loaded and
  merged before any new request is made.
- Completed posts are stored individually in thread_records/<post_id>.json.
- Completed user profiles are stored individually in
  user_profile_records/<username>.json.
- Aggregate JSON/CSV files are rebuilt atomically from those durable records.
- Stopping with q/stop, Ctrl+C, or a runtime timeout preserves completed work.
- To start an unrelated dataset, use a different --output-dir. Existing output
  is never deliberately cleared by this script.

Important limitations:
- PRAW uses OAuth and follows Reddit's live response-header rate limits. This
  script also keeps a configurable request reserve.
- User activity is a sample from accessible newest listings, not a guaranteed
  lifetime archive.
- User flair is subreddit-specific. Flair values are collected from the AITA
  threads being scraped.
- Deleted, removed, private, quarantined, or otherwise unavailable content
  cannot be recovered.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import praw
from praw.models import Comment, Submission
from prawcore.exceptions import (
    Forbidden,
    NotFound,
    PrawcoreException,
    Redirect,
    ResponseException,
)
from tqdm import tqdm


AITA_SUBREDDIT = "AmItheAsshole"
REDDIT_BASE_URL = "https://www.reddit.com"
DEFAULT_USER_HISTORY_LIMIT = 200
DEFAULT_OUTPUT_DIR = "aita_scrape_output"
DEFAULT_LISTING_BATCH_SIZE = 25
DEFAULT_BETWEEN_POST_DELAY_SECONDS = 15.0
DEFAULT_EMPTY_POLL_DELAY_SECONDS = 60.0
DEFAULT_RATE_LIMIT_RESERVE = 10.0
DEFAULT_RATELIMIT_SECONDS = 900
DEFAULT_REQUEST_TIMEOUT_SECONDS = 30
THREAD_RECORDS_DIRNAME = "thread_records"
USER_PROFILE_RECORDS_DIRNAME = "user_profile_records"

# Informational only. Live response headers are the source of truth.
DOCUMENTED_FREE_OAUTH_QPM = 100

# Longer labels come first so YWNBTA is not accidentally reduced to NTA.
JUDGEMENT_LABELS = ("YWNBTA", "YWBTA", "NTA", "YTA", "ESH", "NAH", "INFO")
JUDGEMENT_PATTERN = re.compile(
    r"\b(" + "|".join(map(re.escape, JUDGEMENT_LABELS)) + r")\b",
    flags=re.IGNORECASE,
)

EXCLUDED_VOTERS = {
    "automoderator",
    "judgement_bot_aita",
    "aita_mod",
}

BODY_UPDATE_PATTERN = re.compile(
    r"(?im)^\s*(?:\*{0,2}|#{1,6}\s*|\[)?"
    r"(final\s+update|update|edited\s+to\s+add|edit|eta)"
    r"(?:\s*(?:#?\d+|[ivx]+))?"
    r"\s*(?:\]|\*{0,2})?\s*(?::|\-)\s*"
)
TITLE_UPDATE_PATTERN = re.compile(
    r"(?i)(?:^|\s|\[|\()(final\s+update|update|updated)(?:\]|\)|:|\s|$)"
)
DURATION_PATTERN = re.compile(
    r"^\s*(?P<value>\d+(?:\.\d+)?)\s*(?P<unit>[smh]?)\s*$",
    flags=re.IGNORECASE,
)


@dataclass
class RuntimeController:
    """Track the maximum run time and manual stop state."""

    max_runtime_seconds: float | None
    started_monotonic: float = 0.0
    stop_requested: bool = False
    stop_reason: str | None = None

    def __post_init__(self) -> None:
        if self.started_monotonic == 0.0:
            self.started_monotonic = time.monotonic()

    @property
    def elapsed_seconds(self) -> float:
        return time.monotonic() - self.started_monotonic

    @property
    def remaining_seconds(self) -> float | None:
        if self.max_runtime_seconds is None:
            return None
        return max(0.0, self.max_runtime_seconds - self.elapsed_seconds)

    def should_stop(self) -> bool:
        if self.stop_requested:
            return True
        if (
            self.max_runtime_seconds is not None
            and self.elapsed_seconds >= self.max_runtime_seconds
        ):
            self.stop_requested = True
            self.stop_reason = "maximum runtime reached"
            return True
        return False

    def request_stop(self, reason: str) -> None:
        self.stop_requested = True
        self.stop_reason = reason


def utc_iso(timestamp: float | int | None) -> str | None:
    """Convert a Unix timestamp to an ISO-8601 UTC string."""
    if timestamp is None:
        return None
    return datetime.fromtimestamp(float(timestamp), tz=timezone.utc).isoformat()


def parse_duration(value: str) -> float:
    """Parse 90m, 2h, or 30s. A number without a suffix means minutes."""
    match = DURATION_PATTERN.fullmatch(value)
    if not match:
        raise argparse.ArgumentTypeError(
            "Duration must look like 90m, 2h, or 30s. A bare number means minutes."
        )

    amount = float(match.group("value"))
    unit = match.group("unit").lower() or "m"
    seconds = amount * {"s": 1.0, "m": 60.0, "h": 3600.0}[unit]
    if seconds <= 0:
        raise argparse.ArgumentTypeError("Duration must be greater than zero.")
    return seconds


def format_duration(seconds: float | None) -> str:
    if seconds is None:
        return "unlimited"
    if seconds >= 3600:
        return f"{seconds / 3600:.2f}h"
    if seconds >= 60:
        return f"{seconds / 60:.1f}m"
    return f"{seconds:.1f}s"


def reddit_edited_value(value: Any) -> bool:
    """Reddit returns False when unedited and a timestamp when edited."""
    return bool(value)


def clean_comment_for_judgement(body: str) -> str:
    """Remove Markdown quote lines and fenced code before vote detection."""
    kept_lines: list[str] = []
    in_code_block = False

    for line in body.splitlines():
        stripped = line.lstrip()
        if stripped.startswith("```"):
            in_code_block = not in_code_block
            continue
        if in_code_block or stripped.startswith(">"):
            continue
        kept_lines.append(line)

    return "\n".join(kept_lines)


def extract_judgements(body: str) -> list[str]:
    """Return unique judgement labels in textual order."""
    cleaned = clean_comment_for_judgement(body)
    found: list[str] = []
    seen: set[str] = set()

    for match in JUDGEMENT_PATTERN.finditer(cleaned):
        label = match.group(1).upper()
        if label not in seen:
            seen.add(label)
            found.append(label)
    return found


def detect_update_or_edit(title: str, selftext: str, reddit_edited: Any) -> dict[str, Any]:
    """Detect API-level edits and explicit UPDATE/EDIT headings."""
    markers: list[str] = []

    for match in TITLE_UPDATE_PATTERN.finditer(title or ""):
        markers.append(f"title:{match.group(1).upper()}")
    for match in BODY_UPDATE_PATTERN.finditer(selftext or ""):
        markers.append(f"body:{match.group(1).upper()}")

    markers = list(dict.fromkeys(markers))
    api_edited = reddit_edited_value(reddit_edited)
    return {
        "has_update_or_edit": api_edited or bool(markers),
        "reddit_api_says_edited": api_edited,
        "detected_update_edit_markers": markers,
    }


def choose_winner(counts: Counter[str]) -> dict[str, Any]:
    total = sum(counts.values())
    if total == 0:
        return {
            "winner": None,
            "is_tie": False,
            "tied_labels": [],
            "votes": 0,
            "counts": {label: 0 for label in JUDGEMENT_LABELS},
            "shares": {label: 0.0 for label in JUDGEMENT_LABELS},
        }

    max_votes = max(counts.values())
    tied = sorted(label for label, count in counts.items() if count == max_votes)
    complete_counts = {label: int(counts.get(label, 0)) for label in JUDGEMENT_LABELS}
    return {
        "winner": tied[0] if len(tied) == 1 else "TIE",
        "is_tie": len(tied) > 1,
        "tied_labels": tied if len(tied) > 1 else [],
        "votes": total,
        "counts": complete_counts,
        "shares": {
            label: round(complete_counts[label] / total, 6)
            for label in JUDGEMENT_LABELS
        },
    }


def summarize_judgements(comments: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute comment-level and one-vote-per-unique-user sentiment."""
    eligible = [
        comment
        for comment in comments
        if comment.get("primary_judgement")
        and comment.get("author")
        and comment["author"].lower() not in EXCLUDED_VOTERS
    ]

    comment_counts: Counter[str] = Counter(
        comment["primary_judgement"] for comment in eligible
    )

    votes_by_user: dict[str, list[str]] = defaultdict(list)
    for comment in sorted(eligible, key=lambda item: item.get("created_utc") or 0):
        votes_by_user[comment["author"]].append(comment["primary_judgement"])

    user_vote: dict[str, str] = {}
    for username, votes in votes_by_user.items():
        counts = Counter(votes)
        first_position = {label: votes.index(label) for label in counts}
        user_vote[username] = max(
            counts,
            key=lambda label: (counts[label], -first_position[label]),
        )

    return {
        "comment_level": choose_winner(comment_counts),
        "unique_user_level": choose_winner(Counter(user_vote.values())),
        "judgement_bearing_comments": len(eligible),
        "unique_judgement_voters": len(user_vote),
        "user_vote_method": "modal judgement per user; earliest judgement breaks ties",
    }


def build_reddit_client(args: argparse.Namespace) -> praw.Reddit:
    """Create an OAuth-authenticated, read-only PRAW client."""
    client_id = args.client_id or os.getenv("REDDIT_CLIENT_ID")
    client_secret = args.client_secret or os.getenv("REDDIT_CLIENT_SECRET")
    user_agent = args.user_agent or os.getenv("REDDIT_USER_AGENT")

    missing = [
        name
        for name, value in {
            "REDDIT_CLIENT_ID": client_id,
            "REDDIT_CLIENT_SECRET": client_secret,
            "REDDIT_USER_AGENT": user_agent,
        }.items()
        if not value
    ]
    if missing:
        raise RuntimeError(
            "Missing Reddit API configuration: " + ", ".join(missing) + ". "
            "Set the environment variables or pass the matching command-line options."
        )

    reddit = praw.Reddit(
        client_id=client_id,
        client_secret=client_secret,
        user_agent=user_agent,
        check_for_async=False,
        ratelimit_seconds=args.ratelimit_seconds,
        timeout=args.request_timeout,
    )
    reddit.read_only = True
    return reddit


def normalize_limits(raw_limits: dict[str, Any] | None) -> dict[str, float | None]:
    raw_limits = raw_limits or {}
    lowered = {str(key).lower(): value for key, value in raw_limits.items()}

    def as_float(key: str) -> float | None:
        value = lowered.get(key)
        try:
            return float(value) if value is not None else None
        except (TypeError, ValueError):
            return None

    return {
        "used": as_float("used"),
        "remaining": as_float("remaining"),
        "reset_timestamp": as_float("reset_timestamp"),
    }


def get_rate_limit_snapshot(reddit: praw.Reddit) -> dict[str, float | None]:
    try:
        return normalize_limits(reddit.auth.limits)
    except Exception:
        return {"used": None, "remaining": None, "reset_timestamp": None}


def wait_for_console_or_timeout(
    seconds: float,
    controller: RuntimeController,
    reason: str,
    allow_manual_stop: bool = True,
) -> bool:
    """
    Wait while allowing a manual q/stop command.

    On Windows, pressing q is enough. On POSIX terminals, enter q and press Enter.
    Blank input means continue immediately. No input means continue after the delay.
    """
    if seconds <= 0:
        return controller.should_stop()

    if controller.remaining_seconds is not None:
        seconds = min(seconds, controller.remaining_seconds)
    if seconds <= 0:
        controller.should_stop()
        return True

    if allow_manual_stop and sys.stdin.isatty():
        tqdm.write(
            f"Waiting {format_duration(seconds)} ({reason}). "
            "Type q or stop to finish; no response means continue."
        )
    else:
        tqdm.write(f"Waiting {format_duration(seconds)} ({reason}).")

    deadline = time.monotonic() + seconds

    if not allow_manual_stop or not sys.stdin.isatty():
        while time.monotonic() < deadline:
            if controller.should_stop():
                return True
            time.sleep(min(0.25, max(0.0, deadline - time.monotonic())))
        return controller.should_stop()

    if os.name == "nt":
        import msvcrt

        typed = ""
        while time.monotonic() < deadline:
            if controller.should_stop():
                return True
            while msvcrt.kbhit():
                char = msvcrt.getwch()
                if char in {"\r", "\n"}:
                    command = typed.strip().lower()
                    typed = ""
                    if command in {"q", "quit", "stop", "exit"}:
                        controller.request_stop("manual console stop")
                        return True
                    return controller.should_stop()
                if char == "\b":
                    typed = typed[:-1]
                else:
                    typed += char
                    if typed.strip().lower() == "q":
                        controller.request_stop("manual console stop")
                        return True
            time.sleep(0.1)
    else:
        import select

        while time.monotonic() < deadline:
            if controller.should_stop():
                return True
            timeout = min(0.25, max(0.0, deadline - time.monotonic()))
            readable, _, _ = select.select([sys.stdin], [], [], timeout)
            if readable:
                command = sys.stdin.readline().strip().lower()
                if command in {"q", "quit", "stop", "exit"}:
                    controller.request_stop("manual console stop")
                    return True
                return controller.should_stop()

    return controller.should_stop()


def respect_rate_limit_budget(
    reddit: praw.Reddit,
    reserve: float,
    controller: RuntimeController,
) -> bool:
    """Keep a safety reserve in Reddit's current live API request window."""
    snapshot = get_rate_limit_snapshot(reddit)
    remaining = snapshot["remaining"]
    reset_timestamp = snapshot["reset_timestamp"]

    if remaining is None or reset_timestamp is None or remaining > reserve:
        return controller.should_stop()

    wait_seconds = max(0.0, reset_timestamp - time.time()) + 2.0
    tqdm.write(
        f"API safety reserve reached: {remaining:.1f} requests remain. "
        "Holding until the current window resets."
    )
    return wait_for_console_or_timeout(
        wait_seconds,
        controller,
        reason="Reddit API rate-limit reset",
        allow_manual_stop=True,
    )


def load_submission_from_url(reddit: praw.Reddit, url: str) -> Submission:
    submission = reddit.submission(url=url)
    _ = submission.title
    return submission


def fetch_listing_batch(reddit: praw.Reddit, args: argparse.Namespace) -> list[Submission]:
    subreddit = reddit.subreddit(args.subreddit)
    listing_method = getattr(subreddit, args.listing)
    if args.listing in {"top", "controversial"}:
        listing = listing_method(
            limit=args.listing_batch_size,
            time_filter=args.time_filter,
        )
    else:
        listing = listing_method(limit=args.listing_batch_size)
    return list(listing)


def validate_submission_subreddit(
    submission: Submission,
    args: argparse.Namespace,
) -> bool:
    if args.allow_other_subreddits:
        return True
    return submission.subreddit.display_name.lower() == args.subreddit.lower()


def resolve_full_comment_tree(submission: Submission, attempts: int = 5) -> list[Comment]:
    """Resolve MoreComments placeholders, retrying transient API failures."""
    submission.comment_limit = None

    for attempt in range(1, attempts + 1):
        try:
            submission.comments.replace_more(limit=None)
            return submission.comments.list()
        except (ResponseException, PrawcoreException) as exc:
            if attempt == attempts:
                raise
            delay = min(2 ** (attempt - 1), 30)
            tqdm.write(
                f"Comment-tree request failed for {submission.id} "
                f"({type(exc).__name__}); retrying in {delay}s "
                f"[{attempt}/{attempts}]."
            )
            time.sleep(delay)
    return []


def comment_to_record(comment: Comment, submission_id: str) -> dict[str, Any]:
    body = comment.body or ""
    judgements = extract_judgements(body)
    author = str(comment.author) if comment.author else None

    return {
        "submission_id": submission_id,
        "comment_id": comment.id,
        "parent_id": getattr(comment, "parent_id", None),
        "depth": getattr(comment, "depth", None),
        "author": author,
        "author_flair_text": getattr(comment, "author_flair_text", None),
        "author_flair_css_class": getattr(comment, "author_flair_css_class", None),
        "author_flair_template_id": getattr(comment, "author_flair_template_id", None),
        "body": body,
        "score": getattr(comment, "score", None),
        "created_utc": getattr(comment, "created_utc", None),
        "created_utc_iso": utc_iso(getattr(comment, "created_utc", None)),
        "edited": reddit_edited_value(getattr(comment, "edited", False)),
        "is_submitter": bool(getattr(comment, "is_submitter", False)),
        "distinguished": getattr(comment, "distinguished", None),
        "stickied": bool(getattr(comment, "stickied", False)),
        "permalink": REDDIT_BASE_URL + comment.permalink,
        "all_detected_judgements": judgements,
        "primary_judgement": judgements[0] if judgements else None,
        "contains_multiple_judgements": len(judgements) > 1,
    }


def scrape_submission(
    submission: Submission,
    comment_sort: str,
) -> tuple[dict[str, Any], set[str]]:
    """Scrape one post and every comment Reddit makes available."""
    submission.comment_sort = comment_sort
    tqdm.write(
        f"Resolving full comment tree for post {submission.id}: "
        f"{submission.title[:80]}"
    )
    flat_comments = resolve_full_comment_tree(submission)

    comments: list[dict[str, Any]] = []
    unique_users: set[str] = set()
    for comment in tqdm(
        flat_comments,
        desc=f"Comments {submission.id}",
        unit="comment",
        leave=False,
    ):
        record = comment_to_record(comment, submission.id)
        comments.append(record)
        if record["author"]:
            unique_users.add(record["author"])

    post_author = str(submission.author) if submission.author else None
    if post_author:
        unique_users.add(post_author)

    update_info = detect_update_or_edit(
        submission.title,
        submission.selftext or "",
        getattr(submission, "edited", False),
    )

    post_record: dict[str, Any] = {
        "submission_id": submission.id,
        "subreddit": submission.subreddit.display_name,
        "title": submission.title,
        "author": post_author,
        "author_flair_text": getattr(submission, "author_flair_text", None),
        "author_flair_css_class": getattr(submission, "author_flair_css_class", None),
        "author_flair_template_id": getattr(submission, "author_flair_template_id", None),
        "selftext": submission.selftext or "",
        "created_utc": submission.created_utc,
        "created_utc_iso": utc_iso(submission.created_utc),
        "scraped_at_utc": datetime.now(timezone.utc).isoformat(),
        "score": getattr(submission, "score", None),
        "upvote_ratio": getattr(submission, "upvote_ratio", None),
        "reported_num_comments": getattr(submission, "num_comments", None),
        "scraped_num_comments": len(comments),
        "unique_users_count": len(unique_users),
        "unique_users": sorted(unique_users, key=str.lower),
        "link_flair_text": getattr(submission, "link_flair_text", None),
        "is_self": bool(getattr(submission, "is_self", False)),
        "over_18": bool(getattr(submission, "over_18", False)),
        "locked": bool(getattr(submission, "locked", False)),
        "stickied": bool(getattr(submission, "stickied", False)),
        "permalink": REDDIT_BASE_URL + submission.permalink,
        "content_url": submission.url,
        "comments": comments,
        "judgement_summary": summarize_judgements(comments),
        **update_info,
    }
    return post_record, unique_users


def new_user_observation(username: str) -> dict[str, Any]:
    """Create mutable observations derived only from collected AITA threads."""
    return {
        "username": username,
        "submission_ids_seen": set(),
        "posts_authored_in_scraped_threads": 0,
        "is_original_poster_for_any_scraped_post": False,
        "op_submission_ids": set(),
        "aita_user_flair_texts_seen": set(),
        "aita_user_flair_css_classes_seen": set(),
        "aita_user_flair_template_ids_seen": set(),
        "comments_in_scraped_threads": 0,
        "judgement_comments_in_scraped_threads": 0,
        "is_submitter_comments_in_scraped_threads": 0,
    }


def add_nonempty(target: set[str], value: Any) -> None:
    if value is not None and str(value).strip():
        target.add(str(value))


def update_user_observations(
    observations: dict[str, dict[str, Any]],
    post: dict[str, Any],
) -> None:
    """Aggregate OP flags, thread participation, and observed AITA flair."""
    submission_id = post["submission_id"]
    post_author = post.get("author")

    if post_author:
        obs = observations.setdefault(post_author, new_user_observation(post_author))
        obs["submission_ids_seen"].add(submission_id)
        obs["posts_authored_in_scraped_threads"] += 1
        obs["is_original_poster_for_any_scraped_post"] = True
        obs["op_submission_ids"].add(submission_id)
        add_nonempty(obs["aita_user_flair_texts_seen"], post.get("author_flair_text"))
        add_nonempty(
            obs["aita_user_flair_css_classes_seen"],
            post.get("author_flair_css_class"),
        )
        add_nonempty(
            obs["aita_user_flair_template_ids_seen"],
            post.get("author_flair_template_id"),
        )

    for comment in post.get("comments", []):
        username = comment.get("author")
        if not username:
            continue

        obs = observations.setdefault(username, new_user_observation(username))
        obs["submission_ids_seen"].add(submission_id)
        obs["comments_in_scraped_threads"] += 1
        if comment.get("primary_judgement"):
            obs["judgement_comments_in_scraped_threads"] += 1
        if comment.get("is_submitter"):
            obs["is_submitter_comments_in_scraped_threads"] += 1
            obs["is_original_poster_for_any_scraped_post"] = True
            obs["op_submission_ids"].add(submission_id)

        add_nonempty(obs["aita_user_flair_texts_seen"], comment.get("author_flair_text"))
        add_nonempty(
            obs["aita_user_flair_css_classes_seen"],
            comment.get("author_flair_css_class"),
        )
        add_nonempty(
            obs["aita_user_flair_template_ids_seen"],
            comment.get("author_flair_template_id"),
        )


def serialize_user_observation(observation: dict[str, Any] | None) -> dict[str, Any]:
    """Convert observation sets to deterministic JSON-compatible lists."""
    if not observation:
        return {
            "submission_ids_seen": [],
            "posts_authored_in_scraped_threads": 0,
            "is_original_poster_for_any_scraped_post": False,
            "op_submission_ids": [],
            "aita_user_flair_texts_seen": [],
            "aita_user_flair_css_classes_seen": [],
            "aita_user_flair_template_ids_seen": [],
            "comments_in_scraped_threads": 0,
            "judgement_comments_in_scraped_threads": 0,
            "is_submitter_comments_in_scraped_threads": 0,
        }

    return {
        "submission_ids_seen": sorted(observation.get("submission_ids_seen", set())),
        "posts_authored_in_scraped_threads": int(
            observation.get("posts_authored_in_scraped_threads", 0)
        ),
        "is_original_poster_for_any_scraped_post": bool(
            observation.get("is_original_poster_for_any_scraped_post")
        ),
        "op_submission_ids": sorted(observation.get("op_submission_ids", set())),
        "aita_user_flair_texts_seen": sorted(
            observation.get("aita_user_flair_texts_seen", set()), key=str.lower
        ),
        "aita_user_flair_css_classes_seen": sorted(
            observation.get("aita_user_flair_css_classes_seen", set()), key=str.lower
        ),
        "aita_user_flair_template_ids_seen": sorted(
            observation.get("aita_user_flair_template_ids_seen", set()), key=str.lower
        ),
        "comments_in_scraped_threads": int(
            observation.get("comments_in_scraped_threads", 0)
        ),
        "judgement_comments_in_scraped_threads": int(
            observation.get("judgement_comments_in_scraped_threads", 0)
        ),
        "is_submitter_comments_in_scraped_threads": int(
            observation.get("is_submitter_comments_in_scraped_threads", 0)
        ),
    }


def reconstruct_user_observations(
    posts: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    observations: dict[str, dict[str, Any]] = {}
    for post in posts:
        update_user_observations(observations, post)
    return observations


def count_accessible_user_items(
    items: Iterable[Any],
    target_subreddit: str,
    history_limit: int,
) -> dict[str, Any]:
    """Count sampled accessible items overall and in the target subreddit."""
    total = 0
    in_target = 0

    for item in items:
        total += 1
        try:
            subreddit_name = item.subreddit.display_name
        except (AttributeError, Forbidden, NotFound, PrawcoreException):
            subreddit_name = None
        if subreddit_name and subreddit_name.lower() == target_subreddit.lower():
            in_target += 1

    return {
        "accessible_total": total,
        "accessible_in_aita": in_target,
        "aita_share": round(in_target / total, 6) if total else 0.0,
        "may_be_truncated": total >= history_limit,
    }


def scrape_user_profile(
    reddit: praw.Reddit,
    username: str,
    target_subreddit: str,
    history_limit: int,
) -> dict[str, Any]:
    """Scrape public karma and a configurable sample of user activity."""
    profile: dict[str, Any] = {
        "username": username,
        "profile_url": f"{REDDIT_BASE_URL}/user/{username}/",
        "profile_scraped_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "ok",
        "error": None,
        "link_karma": None,
        "comment_karma": None,
        "total_karma": None,
        "created_utc": None,
        "created_utc_iso": None,
        "account_age_days_at_scrape": None,
        "posts_accessible_total": None,
        "posts_accessible_in_aita": None,
        "posts_accessible_aita_share": None,
        "posts_may_be_truncated": None,
        "comments_accessible_total": None,
        "comments_accessible_in_aita": None,
        "comments_accessible_aita_share": None,
        "comments_may_be_truncated": None,
        "history_limit_requested_per_type": history_limit,
    }

    try:
        redditor = reddit.redditor(username)
        profile["link_karma"] = int(redditor.link_karma)
        profile["comment_karma"] = int(redditor.comment_karma)
        profile["total_karma"] = int(
            getattr(
                redditor,
                "total_karma",
                profile["link_karma"] + profile["comment_karma"],
            )
        )
        profile["created_utc"] = getattr(redditor, "created_utc", None)
        profile["created_utc_iso"] = utc_iso(profile["created_utc"])
        if profile["created_utc"] is not None:
            profile["account_age_days_at_scrape"] = round(
                (time.time() - float(profile["created_utc"])) / 86400,
                2,
            )

        post_counts = count_accessible_user_items(
            redditor.submissions.new(limit=history_limit),
            target_subreddit,
            history_limit,
        )
        comment_counts = count_accessible_user_items(
            redditor.comments.new(limit=history_limit),
            target_subreddit,
            history_limit,
        )
        profile.update(
            {
                "posts_accessible_total": post_counts["accessible_total"],
                "posts_accessible_in_aita": post_counts["accessible_in_aita"],
                "posts_accessible_aita_share": post_counts["aita_share"],
                "posts_may_be_truncated": post_counts["may_be_truncated"],
                "comments_accessible_total": comment_counts["accessible_total"],
                "comments_accessible_in_aita": comment_counts["accessible_in_aita"],
                "comments_accessible_aita_share": comment_counts["aita_share"],
                "comments_may_be_truncated": comment_counts["may_be_truncated"],
            }
        )
    except (NotFound, Forbidden, Redirect) as exc:
        profile["status"] = "unavailable_or_suspended"
        profile["error"] = f"{type(exc).__name__}: {exc}"
    except (ResponseException, PrawcoreException) as exc:
        profile["status"] = "api_error"
        profile["error"] = f"{type(exc).__name__}: {exc}"
    except Exception as exc:
        profile["status"] = "unexpected_error"
        profile["error"] = f"{type(exc).__name__}: {exc}"

    return profile


# ---------------------------------------------------------------------------
# Durable persistence
# ---------------------------------------------------------------------------


def safe_record_name(value: str) -> str:
    """Create a conservative filename for a Reddit ID or username."""
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._")
    return cleaned or "unknown"


def atomic_write_text(path: Path, text: str, keep_backup: bool = True) -> None:
    """Write a text file atomically, optionally retaining the prior version."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")

    with temporary.open("w", encoding="utf-8", newline="") as file:
        file.write(text)
        file.flush()
        os.fsync(file.fileno())

    if keep_backup and path.exists():
        backup = path.with_name(path.name + ".bak")
        try:
            shutil.copy2(path, backup)
        except OSError:
            pass

    os.replace(temporary, path)


def atomic_write_json(path: Path, value: Any, keep_backup: bool = True) -> None:
    atomic_write_text(
        path,
        json.dumps(value, indent=2, ensure_ascii=False),
        keep_backup=keep_backup,
    )


def atomic_write_csv(
    path: Path,
    rows: list[dict[str, Any]],
    fieldnames: list[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")

    with temporary.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {key: csv_value(row.get(key)) for key in fieldnames}
            )
        file.flush()
        os.fsync(file.fileno())

    if path.exists():
        backup = path.with_name(path.name + ".bak")
        try:
            shutil.copy2(path, backup)
        except OSError:
            pass
    os.replace(temporary, path)


def load_json_safely(path: Path) -> Any | None:
    """Load JSON, falling back to its .bak copy after an interrupted write."""
    candidates = (path, path.with_name(path.name + ".bak"))
    for candidate in candidates:
        if not candidate.exists():
            continue
        try:
            with candidate.open("r", encoding="utf-8") as file:
                return json.load(file)
        except (OSError, json.JSONDecodeError) as exc:
            tqdm.write(f"Could not read {candidate}: {type(exc).__name__}: {exc}")
    return None


def save_post_record(post: dict[str, Any], output_dir: Path) -> Path:
    """Persist a completed thread before rebuilding any aggregate files."""
    path = (
        output_dir
        / THREAD_RECORDS_DIRNAME
        / f"{safe_record_name(post['submission_id'])}.json"
    )
    atomic_write_json(path, post, keep_backup=True)
    return path


def save_user_profile_record(profile: dict[str, Any], output_dir: Path) -> Path:
    """Persist a completed user enrichment record independently."""
    path = (
        output_dir
        / USER_PROFILE_RECORDS_DIRNAME
        / f"{safe_record_name(profile['username'])}.json"
    )
    atomic_write_json(path, profile, keep_backup=True)
    return path


def profile_is_completed(profile: dict[str, Any] | None) -> bool:
    """Distinguish real enrichment results from old placeholder rows."""
    if not profile:
        return False
    return profile.get("status") not in {None, "pending", "profile_scrape_skipped"}


def materialize_durable_records(
    posts: list[dict[str, Any]],
    profiles_by_name: dict[str, dict[str, Any]],
    output_dir: Path,
) -> None:
    """Migrate records loaded from an older aggregate into per-item files."""
    for post in posts:
        path = (
            output_dir
            / THREAD_RECORDS_DIRNAME
            / f"{safe_record_name(post['submission_id'])}.json"
        )
        if not path.exists():
            atomic_write_json(path, post, keep_backup=False)

    for profile in profiles_by_name.values():
        path = (
            output_dir
            / USER_PROFILE_RECORDS_DIRNAME
            / f"{safe_record_name(profile['username'])}.json"
        )
        if not path.exists():
            atomic_write_json(path, profile, keep_backup=False)


def load_durable_data(
    output_dir: Path,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    """
    Recover data from individual records first, then merge older aggregate data.

    The individual files are the source of truth. The aggregate fallback keeps
    compatibility with datasets created by earlier versions of this script.
    """
    posts_by_id: dict[str, dict[str, Any]] = {}
    profiles_by_name: dict[str, dict[str, Any]] = {}

    thread_dir = output_dir / THREAD_RECORDS_DIRNAME
    if thread_dir.exists():
        for path in sorted(thread_dir.glob("*.json")):
            record = load_json_safely(path)
            if isinstance(record, dict) and record.get("submission_id"):
                posts_by_id[record["submission_id"]] = record

    profile_dir = output_dir / USER_PROFILE_RECORDS_DIRNAME
    if profile_dir.exists():
        for path in sorted(profile_dir.glob("*.json")):
            record = load_json_safely(path)
            if isinstance(record, dict) and record.get("username"):
                profiles_by_name[record["username"]] = record

    aggregate = load_json_safely(output_dir / "aita_scrape.json")
    if isinstance(aggregate, dict):
        for post in aggregate.get("posts", []):
            if isinstance(post, dict) and post.get("submission_id"):
                posts_by_id.setdefault(post["submission_id"], post)

        legacy_profiles = aggregate.get("user_profiles", aggregate.get("users", []))
        for profile in legacy_profiles:
            if (
                isinstance(profile, dict)
                and profile.get("username")
                and profile_is_completed(profile)
            ):
                profiles_by_name.setdefault(profile["username"], profile)

    posts = sorted(
        posts_by_id.values(),
        key=lambda post: (post.get("created_utc") or 0, post.get("submission_id") or ""),
    )
    return posts, profiles_by_name


def csv_value(value: Any) -> Any:
    if isinstance(value, (dict, list, tuple, set)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return value


def build_unique_user_rows(
    observations: dict[str, dict[str, Any]],
    profiles_by_name: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    """Create the separate unique-user queue produced by the thread stage."""
    rows: list[dict[str, Any]] = []
    for username in sorted(observations, key=str.lower):
        profile = profiles_by_name.get(username)
        completed = profile_is_completed(profile)
        rows.append(
            {
                "username": username,
                "profile_url": f"{REDDIT_BASE_URL}/user/{username}/",
                "profile_scraped": completed,
                "profile_status": profile.get("status") if completed else "pending",
                "profile_scraped_at_utc": (
                    profile.get("profile_scraped_at_utc") if completed else None
                ),
                **serialize_user_observation(observations[username]),
            }
        )
    return rows


def build_dataset(
    args: argparse.Namespace,
    posts: list[dict[str, Any]],
    profiles_by_name: dict[str, dict[str, Any]],
    observations: dict[str, dict[str, Any]],
    controller: RuntimeController,
    reddit: praw.Reddit,
) -> dict[str, Any]:
    unique_users = build_unique_user_rows(observations, profiles_by_name)
    profiles = sorted(
        profiles_by_name.values(),
        key=lambda item: item["username"].lower(),
    )

    return {
        "metadata": {
            "checkpointed_at_utc": datetime.now(timezone.utc).isoformat(),
            "active_stage": args.stage,
            "automatic_resume": True,
            "subreddit": args.subreddit,
            "post_count": len(posts),
            "unique_thread_users_count": len(unique_users),
            "completed_user_profiles_count": len(profiles),
            "pending_user_profiles_count": sum(
                1 for row in unique_users if not row["profile_scraped"]
            ),
            "user_history_limit_requested_per_type": args.user_history_limit,
            "elapsed_runtime_seconds": round(controller.elapsed_seconds, 3),
            "maximum_runtime_seconds": controller.max_runtime_seconds,
            "stop_requested": controller.stop_requested,
            "stop_reason": controller.stop_reason,
            "listing": args.listing,
            "listing_batch_size": args.listing_batch_size,
            "between_post_delay_seconds": args.between_post_delay,
            "empty_poll_delay_seconds": args.empty_poll_delay,
            "durable_sources": {
                "threads": THREAD_RECORDS_DIRNAME,
                "user_profiles": USER_PROFILE_RECORDS_DIRNAME,
            },
            "rate_limit": {
                "oauth_required": True,
                "documented_free_oauth_queries_per_minute": DOCUMENTED_FREE_OAUTH_QPM,
                "praw_automatic_header_throttling": True,
                "praw_ratelimit_seconds": args.ratelimit_seconds,
                "configured_request_reserve": args.rate_limit_reserve,
                "latest_live_header_snapshot": get_rate_limit_snapshot(reddit),
            },
            "judgement_labels": list(JUDGEMENT_LABELS),
            "community_sentiment_definition": {
                "comment_level": (
                    "first explicit judgement in every judgement-bearing comment"
                ),
                "unique_user_level": (
                    "modal judgement per user; earliest judgement breaks ties"
                ),
            },
            "limitations": [
                "Removed, deleted, private, quarantined, or otherwise unavailable content cannot be recovered.",
                "User activity counts are configurable samples of accessible newest listings, not guaranteed lifetime totals.",
                "User flair is subreddit-specific and is collected from observed AITA contributions.",
                "Judgement extraction is rule-based and may misclassify sarcasm, negation, or unusual formatting.",
            ],
        },
        "posts": posts,
        "unique_users": unique_users,
        "user_profiles": profiles,
        # Compatibility alias for notebooks built against the previous version.
        "users": profiles,
    }


def save_results(
    dataset: dict[str, Any],
    output_dir: Path,
    quiet: bool = False,
) -> None:
    """Rebuild aggregate JSON and CSV outputs atomically."""
    output_dir.mkdir(parents=True, exist_ok=True)

    posts = dataset["posts"]
    unique_users = dataset["unique_users"]
    profiles = dataset["user_profiles"]
    comments = [comment for post in posts for comment in post.get("comments", [])]

    post_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    for post in posts:
        post_rows.append({key: value for key, value in post.items() if key != "comments"})
        for method in ("comment_level", "unique_user_level"):
            summary = post["judgement_summary"][method]
            row: dict[str, Any] = {
                "submission_id": post["submission_id"],
                "title": post["title"],
                "method": method,
                "winner": summary["winner"],
                "is_tie": summary["is_tie"],
                "votes": summary["votes"],
            }
            for label in JUDGEMENT_LABELS:
                row[f"{label}_count"] = summary["counts"][label]
                row[f"{label}_share"] = summary["shares"][label]
            summary_rows.append(row)

    post_fields = [
        "submission_id",
        "subreddit",
        "title",
        "author",
        "author_flair_text",
        "author_flair_css_class",
        "author_flair_template_id",
        "selftext",
        "created_utc",
        "created_utc_iso",
        "scraped_at_utc",
        "score",
        "upvote_ratio",
        "reported_num_comments",
        "scraped_num_comments",
        "unique_users_count",
        "unique_users",
        "link_flair_text",
        "is_self",
        "over_18",
        "locked",
        "stickied",
        "permalink",
        "content_url",
        "has_update_or_edit",
        "reddit_api_says_edited",
        "detected_update_edit_markers",
        "judgement_summary",
    ]
    comment_fields = [
        "submission_id",
        "comment_id",
        "parent_id",
        "depth",
        "author",
        "author_flair_text",
        "author_flair_css_class",
        "author_flair_template_id",
        "body",
        "score",
        "created_utc",
        "created_utc_iso",
        "edited",
        "is_submitter",
        "distinguished",
        "stickied",
        "permalink",
        "all_detected_judgements",
        "primary_judgement",
        "contains_multiple_judgements",
    ]
    unique_user_fields = [
        "username",
        "profile_url",
        "profile_scraped",
        "profile_status",
        "profile_scraped_at_utc",
        "submission_ids_seen",
        "posts_authored_in_scraped_threads",
        "is_original_poster_for_any_scraped_post",
        "op_submission_ids",
        "aita_user_flair_texts_seen",
        "aita_user_flair_css_classes_seen",
        "aita_user_flair_template_ids_seen",
        "comments_in_scraped_threads",
        "judgement_comments_in_scraped_threads",
        "is_submitter_comments_in_scraped_threads",
    ]
    profile_fields = [
        "username",
        "profile_url",
        "profile_scraped_at_utc",
        "status",
        "error",
        "link_karma",
        "comment_karma",
        "total_karma",
        "created_utc",
        "created_utc_iso",
        "account_age_days_at_scrape",
        "posts_accessible_total",
        "posts_accessible_in_aita",
        "posts_accessible_aita_share",
        "posts_may_be_truncated",
        "comments_accessible_total",
        "comments_accessible_in_aita",
        "comments_accessible_aita_share",
        "comments_may_be_truncated",
        "history_limit_requested_per_type",
    ]
    summary_fields = [
        "submission_id",
        "title",
        "method",
        "winner",
        "is_tie",
        "votes",
        *[
            f"{label}_{suffix}"
            for label in JUDGEMENT_LABELS
            for suffix in ("count", "share")
        ],
    ]

    atomic_write_json(output_dir / "aita_scrape.json", dataset)
    atomic_write_json(output_dir / "unique_users.json", unique_users)
    atomic_write_json(output_dir / "user_profiles.json", profiles)
    atomic_write_json(
        output_dir / "run_state.json",
        {
            **dataset["metadata"],
            "completed_submission_ids": [post["submission_id"] for post in posts],
            "completed_usernames": [profile["username"] for profile in profiles],
        },
    )

    atomic_write_csv(output_dir / "posts.csv", post_rows, post_fields)
    atomic_write_csv(output_dir / "comments.csv", comments, comment_fields)
    atomic_write_csv(output_dir / "unique_users.csv", unique_users, unique_user_fields)
    atomic_write_csv(output_dir / "user_profiles.csv", profiles, profile_fields)
    # Compatibility alias retained from the previous version.
    atomic_write_csv(output_dir / "users.csv", profiles, profile_fields)
    atomic_write_csv(
        output_dir / "judgement_summary.csv",
        summary_rows,
        summary_fields,
    )

    if not quiet:
        print(f"Saved durable and aggregate outputs to: {output_dir.resolve()}")


def checkpoint(
    args: argparse.Namespace,
    posts: list[dict[str, Any]],
    profiles_by_name: dict[str, dict[str, Any]],
    observations: dict[str, dict[str, Any]],
    controller: RuntimeController,
    reddit: praw.Reddit,
    quiet: bool = True,
) -> None:
    dataset = build_dataset(
        args,
        posts,
        profiles_by_name,
        observations,
        controller,
        reddit,
    )
    save_results(dataset, Path(args.output_dir), quiet=quiet)


# ---------------------------------------------------------------------------
# Stage 1: threads
# ---------------------------------------------------------------------------


def process_submission_thread(
    submission: Submission,
    reddit: praw.Reddit,
    args: argparse.Namespace,
    controller: RuntimeController,
    posts: list[dict[str, Any]],
    observations: dict[str, dict[str, Any]],
    profiles_by_name: dict[str, dict[str, Any]],
) -> bool:
    """Scrape and durably commit one thread without requesting user histories."""
    if controller.should_stop():
        return False

    if not validate_submission_subreddit(submission, args):
        tqdm.write(
            f"Skipping {submission.id}: belongs to "
            f"r/{submission.subreddit.display_name}, not r/{args.subreddit}."
        )
        return False

    try:
        post_record, post_users = scrape_submission(submission, args.comment_sort)

        # Commit the thread itself first. If aggregate rebuilding is interrupted,
        # the next run recovers this file and reconstructs the queue automatically.
        record_path = save_post_record(post_record, Path(args.output_dir))
        posts.append(post_record)
        update_user_observations(observations, post_record)

        try:
            checkpoint(
                args,
                posts,
                profiles_by_name,
                observations,
                controller,
                reddit,
                quiet=True,
            )
        except Exception as exc:
            tqdm.write(
                "The thread record is safe, but aggregate export failed: "
                f"{type(exc).__name__}: {exc}. It will be rebuilt next run."
            )

        winner = post_record["judgement_summary"]["unique_user_level"]["winner"]
        votes = post_record["judgement_summary"]["unique_user_level"]["votes"]
        tqdm.write(
            f"Committed post {submission.id} to {record_path.name}: "
            f"{post_record['scraped_num_comments']} comments, "
            f"{len(post_users)} unique users, community judgement={winner} "
            f"({votes} unique-user votes)."
        )
        return True

    except (Forbidden, NotFound, Redirect, ResponseException, PrawcoreException) as exc:
        tqdm.write(f"Skipping post {submission.id} after Reddit API error: {exc}")
    except Exception as exc:
        tqdm.write(
            f"Skipping post {submission.id} after unexpected error: "
            f"{type(exc).__name__}: {exc}"
        )
    return False


def run_thread_stage(
    reddit: praw.Reddit,
    args: argparse.Namespace,
    controller: RuntimeController,
    posts: list[dict[str, Any]],
    profiles_by_name: dict[str, dict[str, Any]],
    observations: dict[str, dict[str, Any]],
) -> None:
    seen_submission_ids = {post["submission_id"] for post in posts}
    posts_at_start = len(posts)

    if args.post_url:
        for url in args.post_url:
            if controller.should_stop():
                break
            if args.max_posts is not None and len(posts) - posts_at_start >= args.max_posts:
                controller.request_stop("maximum post count reached")
                break
            if respect_rate_limit_budget(reddit, args.rate_limit_reserve, controller):
                break

            try:
                submission = load_submission_from_url(reddit, url)
            except Exception as exc:
                tqdm.write(f"Could not load {url}: {type(exc).__name__}: {exc}")
                continue

            if submission.id in seen_submission_ids:
                tqdm.write(f"Skipping already-committed post {submission.id}.")
                continue

            completed = process_submission_thread(
                submission,
                reddit,
                args,
                controller,
                posts,
                observations,
                profiles_by_name,
            )
            if completed:
                seen_submission_ids.add(submission.id)
                if wait_for_console_or_timeout(
                    args.between_post_delay,
                    controller,
                    reason="between-post safety delay",
                    allow_manual_stop=True,
                ):
                    break
        return

    while not controller.should_stop():
        if args.max_posts is not None and len(posts) - posts_at_start >= args.max_posts:
            controller.request_stop("maximum post count reached")
            break
        if respect_rate_limit_budget(reddit, args.rate_limit_reserve, controller):
            break

        try:
            batch = fetch_listing_batch(reddit, args)
        except (ResponseException, PrawcoreException) as exc:
            tqdm.write(f"Listing poll failed: {type(exc).__name__}: {exc}")
            if wait_for_console_or_timeout(
                args.empty_poll_delay,
                controller,
                reason="retry after listing error",
                allow_manual_stop=True,
            ):
                break
            continue

        unseen = [
            submission for submission in batch if submission.id not in seen_submission_ids
        ]
        unseen.reverse()  # Stable oldest-first processing within each fetched window.

        if not unseen:
            if wait_for_console_or_timeout(
                args.empty_poll_delay,
                controller,
                reason="no unseen posts in the latest listing poll",
                allow_manual_stop=True,
            ):
                break
            continue

        for submission in unseen:
            if controller.should_stop():
                break
            if args.max_posts is not None and len(posts) - posts_at_start >= args.max_posts:
                controller.request_stop("maximum post count reached")
                break
            if respect_rate_limit_budget(reddit, args.rate_limit_reserve, controller):
                break

            completed = process_submission_thread(
                submission,
                reddit,
                args,
                controller,
                posts,
                observations,
                profiles_by_name,
            )
            if not completed:
                continue

            seen_submission_ids.add(submission.id)
            if wait_for_console_or_timeout(
                args.between_post_delay,
                controller,
                reason="between-post safety delay",
                allow_manual_stop=True,
            ):
                break


# ---------------------------------------------------------------------------
# Stage 2: user enrichment
# ---------------------------------------------------------------------------


def profile_needs_scrape(
    profile: dict[str, Any] | None,
    retry_user_errors: bool,
) -> bool:
    if not profile_is_completed(profile):
        return True
    if retry_user_errors and profile.get("status") in {"api_error", "unexpected_error"}:
        return True
    return False


def run_user_stage(
    reddit: praw.Reddit,
    args: argparse.Namespace,
    controller: RuntimeController,
    posts: list[dict[str, Any]],
    profiles_by_name: dict[str, dict[str, Any]],
    observations: dict[str, dict[str, Any]],
) -> None:
    """Enrich only the saved unique-user queue. No thread is downloaded here."""
    pending = [
        username
        for username in sorted(observations, key=str.lower)
        if profile_needs_scrape(
            profiles_by_name.get(username),
            args.retry_user_errors,
        )
    ]

    if args.max_users is not None:
        pending = pending[: args.max_users]

    if not pending:
        tqdm.write(
            "No pending users were found. Run --stage threads first, or all saved "
            "users have already been enriched."
        )
        return

    tqdm.write(
        f"User stage loaded {len(observations)} unique thread users; "
        f"{len(pending)} require enrichment in this run."
    )

    for username in tqdm(pending, desc="User profiles", unit="user"):
        if controller.should_stop():
            break
        if respect_rate_limit_budget(reddit, args.rate_limit_reserve, controller):
            break

        profile = scrape_user_profile(
            reddit,
            username,
            args.subreddit,
            args.user_history_limit,
        )

        # Save the individual profile before rebuilding aggregate files.
        save_user_profile_record(profile, Path(args.output_dir))
        profiles_by_name[username] = profile

        try:
            checkpoint(
                args,
                posts,
                profiles_by_name,
                observations,
                controller,
                reddit,
                quiet=True,
            )
        except Exception as exc:
            tqdm.write(
                "The user record is safe, but aggregate export failed: "
                f"{type(exc).__name__}: {exc}. It will be rebuilt next run."
            )

        if args.user_delay > 0:
            if wait_for_console_or_timeout(
                args.user_delay,
                controller,
                reason="between-user delay",
                allow_manual_stop=True,
            ):
                break


# ---------------------------------------------------------------------------
# CLI and main
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Resumably scrape AITA threads and, in a separate stage, enrich the "
            "unique users found in those threads."
        )
    )
    parser.add_argument(
        "--stage",
        choices=("threads", "users"),
        default="threads",
        help=(
            "threads: collect posts/comments/judgements and build unique_users. "
            "users: enrich that saved queue with karma and activity samples."
        ),
    )
    parser.add_argument(
        "--post-url",
        action="append",
        help="Specific Reddit post URL. Repeat for multiple posts. Threads stage only.",
    )
    parser.add_argument("--subreddit", default=AITA_SUBREDDIT)
    parser.add_argument(
        "--listing-batch-size",
        "--post-limit",
        dest="listing_batch_size",
        type=int,
        default=DEFAULT_LISTING_BATCH_SIZE,
        help="Number of posts requested from each listing poll.",
    )
    parser.add_argument(
        "--max-posts",
        type=int,
        help="Optional number of new posts to collect in this thread-stage run.",
    )
    parser.add_argument(
        "--max-users",
        type=int,
        help="Optional number of pending users to enrich in this user-stage run.",
    )
    parser.add_argument(
        "--listing",
        choices=("new", "hot", "top", "controversial"),
        default="new",
    )
    parser.add_argument(
        "--time-filter",
        choices=("hour", "day", "week", "month", "year", "all"),
        default="month",
        help="Used only with top or controversial listings.",
    )
    parser.add_argument(
        "--comment-sort",
        choices=("confidence", "top", "new", "controversial", "old", "qa"),
        default="confidence",
    )
    parser.add_argument(
        "--user-history-limit",
        type=int,
        default=DEFAULT_USER_HISTORY_LIMIT,
        help="Activity sample size per user, per type. Users stage only.",
    )
    parser.add_argument(
        "--retry-user-errors",
        action="store_true",
        help="Retry saved api_error and unexpected_error user records.",
    )
    parser.add_argument(
        "--allow-other-subreddits",
        action="store_true",
        help="Allow explicit post URLs outside the configured subreddit.",
    )
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Deprecated compatibility flag. Resume is now always automatic.",
    )
    parser.add_argument(
        "--max-runtime",
        type=parse_duration,
        help="Maximum run time such as 90m or 2h. Bare numbers mean minutes.",
    )
    parser.add_argument(
        "--between-post-delay",
        type=float,
        default=DEFAULT_BETWEEN_POST_DELAY_SECONDS,
        help=(
            "Seconds after each completed post. During this wait, q/stop saves and "
            "ends the run; no input continues automatically."
        ),
    )
    parser.add_argument(
        "--empty-poll-delay",
        type=float,
        default=DEFAULT_EMPTY_POLL_DELAY_SECONDS,
        help="Seconds to wait when a listing poll has no unseen posts.",
    )
    parser.add_argument(
        "--rate-limit-reserve",
        type=float,
        default=DEFAULT_RATE_LIMIT_RESERVE,
        help="Pause when Reddit's remaining-request header reaches this reserve.",
    )
    parser.add_argument(
        "--ratelimit-seconds",
        type=int,
        default=DEFAULT_RATELIMIT_SECONDS,
        help="Maximum Reddit-requested wait PRAW handles automatically.",
    )
    parser.add_argument(
        "--request-timeout",
        type=int,
        default=DEFAULT_REQUEST_TIMEOUT_SECONDS,
        help="Network request timeout in seconds.",
    )
    parser.add_argument("--client-id")
    parser.add_argument("--client-secret")
    parser.add_argument("--user-agent")
    parser.add_argument(
        "--user-delay",
        type=float,
        default=0.0,
        help=(
            "Optional seconds between user enrichments. During this wait, q/stop "
            "ends cleanly."
        ),
    )

    args = parser.parse_args()

    if args.listing_batch_size < 1:
        parser.error("--listing-batch-size must be at least 1")
    if args.max_posts is not None and args.max_posts < 1:
        parser.error("--max-posts must be at least 1")
    if args.max_users is not None and args.max_users < 1:
        parser.error("--max-users must be at least 1")
    if args.user_history_limit < 1:
        parser.error("--user-history-limit must be at least 1")
    if args.user_delay < 0:
        parser.error("--user-delay cannot be negative")
    if args.between_post_delay < 0:
        parser.error("--between-post-delay cannot be negative")
    if args.empty_poll_delay < 0:
        parser.error("--empty-poll-delay cannot be negative")
    if args.rate_limit_reserve < 0:
        parser.error("--rate-limit-reserve cannot be negative")
    if args.ratelimit_seconds < 1:
        parser.error("--ratelimit-seconds must be at least 1")
    if args.request_timeout < 1:
        parser.error("--request-timeout must be at least 1")
    if args.stage == "users" and args.post_url:
        parser.error("--post-url is only valid with --stage threads")

    return args


def main() -> None:
    args = parse_args()
    controller = RuntimeController(args.max_runtime)
    reddit = build_reddit_client(args)
    output_dir = Path(args.output_dir)

    posts, profiles_by_name = load_durable_data(output_dir)
    materialize_durable_records(posts, profiles_by_name, output_dir)
    observations = reconstruct_user_observations(posts)
    posts_at_start = len(posts)
    profiles_at_start = len(profiles_by_name)

    if posts or profiles_by_name:
        tqdm.write(
            f"Automatic resume: loaded {len(posts)} committed threads and "
            f"{len(profiles_by_name)} completed user profiles from "
            f"{output_dir.resolve()}."
        )
    else:
        tqdm.write(
            f"No prior data found in {output_dir.resolve()}; starting a new durable dataset."
        )

    tqdm.write(
        "OAuth read-only mode enabled. PRAW follows Reddit's live rate-limit "
        f"headers; an additional {args.rate_limit_reserve:g}-request reserve is enabled."
    )
    tqdm.write(
        f"Stage: {args.stage}. Runtime limit: {format_duration(args.max_runtime)}. "
        "Completed records are committed before aggregate files are rebuilt."
    )

    try:
        if args.stage == "threads":
            run_thread_stage(
                reddit,
                args,
                controller,
                posts,
                profiles_by_name,
                observations,
            )
        else:
            run_user_stage(
                reddit,
                args,
                controller,
                posts,
                profiles_by_name,
                observations,
            )
    except KeyboardInterrupt:
        controller.request_stop("Ctrl+C keyboard interrupt")
        tqdm.write("Keyboard interrupt received. Preserving all completed records.")
    finally:
        # Rebuild aggregates one final time. If this fails, individual records
        # remain intact and are automatically recovered on the next run.
        try:
            checkpoint(
                args,
                posts,
                profiles_by_name,
                observations,
                controller,
                reddit,
                quiet=False,
            )
        except Exception as exc:
            print(
                "Final aggregate export failed, but individual completed records "
                f"remain safe: {type(exc).__name__}: {exc}",
                file=sys.stderr,
            )

    total_comments = sum(post.get("scraped_num_comments", 0) for post in posts)
    total_votes = sum(
        post.get("judgement_summary", {})
        .get("comment_level", {})
        .get("votes", 0)
        for post in posts
    )
    new_posts = len(posts) - posts_at_start
    new_profiles = len(profiles_by_name) - profiles_at_start
    pending_profiles = sum(
        1
        for username in observations
        if profile_needs_scrape(profiles_by_name.get(username), False)
    )

    print(
        f"Finished stage={args.stage}: {new_posts} new threads, "
        f"{new_profiles} new user profiles. Dataset totals: {len(posts)} threads, "
        f"{total_comments} comments, {total_votes} judgement-bearing comments, "
        f"{len(observations)} unique thread users, {len(profiles_by_name)} completed "
        f"profiles, {pending_profiles} pending profiles. Stop reason: "
        f"{controller.stop_reason or 'stage work completed'}."
    )


if __name__ == "__main__":
    main()
