"""
AmItheAsshole Reddit scraper.

Install:
    pip install praw tqdm

Create a Reddit API application, then set these environment variables:
    REDDIT_CLIENT_ID
    REDDIT_CLIENT_SECRET
    REDDIT_USER_AGENT

Examples:
    python "Reddit Scraping.py" --post-url \
        "https://www.reddit.com/r/AmItheAsshole/comments/POST_ID/example/"

    python "Reddit Scraping.py" --post-limit 10 --listing new

Notes:
- PRAW's replace_more(limit=None) is used to resolve the complete comment tree
  that Reddit makes available through its API.
- Deleted, removed, private, quarantined, or otherwise unavailable content cannot
  be recovered.
- Reddit user-history listings are finite. User activity fields therefore report
  accessible counts, not guaranteed lifetime totals. A truncation flag is saved.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import praw
from praw.models import Comment, Submission
from prawcore.exceptions import Forbidden, NotFound, PrawcoreException, Redirect, ResponseException
from tqdm import tqdm


AITA_SUBREDDIT = "AmItheAsshole"
REDDIT_BASE_URL = "https://www.reddit.com"
DEFAULT_USER_HISTORY_LIMIT = 1000
DEFAULT_OUTPUT_DIR = "aita_scrape_output"

# Longer labels come first so YWNBTA is not accidentally reduced to NTA, etc.
JUDGEMENT_LABELS = ("YWNBTA", "YWBTA", "NTA", "YTA", "ESH", "NAH", "INFO")
JUDGEMENT_PATTERN = re.compile(
    r"\b(" + "|".join(map(re.escape, JUDGEMENT_LABELS)) + r")\b",
    flags=re.IGNORECASE,
)

# These accounts are not useful as community voters.
EXCLUDED_VOTERS = {
    "automoderator",
    "judgement_bot_aita",
    "aita_mod",
}

# Detect explicit update/edit headings without flagging every ordinary use of
# the words "update" or "edit" in the body.
BODY_UPDATE_PATTERN = re.compile(
    r"(?im)^\s*(?:\*{0,2}|#{1,6}\s*|\[)?"
    r"(final\s+update|update|edited\s+to\s+add|edit|eta)"
    r"(?:\s*(?:#?\d+|[ivx]+))?"
    r"\s*(?:\]|\*{0,2})?\s*(?::|\-)\s*"
)
TITLE_UPDATE_PATTERN = re.compile(
    r"(?i)(?:^|\s|\[|\()(final\s+update|update|updated)(?:\]|\)|:|\s|$)"
)


def utc_iso(timestamp: float | int | None) -> str | None:
    """Convert a Unix timestamp to an ISO-8601 UTC string."""
    if timestamp is None:
        return None
    return datetime.fromtimestamp(float(timestamp), tz=timezone.utc).isoformat()


def reddit_edited_value(value: Any) -> bool:
    """Reddit returns False when unedited and a timestamp when edited."""
    return bool(value)


def clean_comment_for_judgement(body: str) -> str:
    """
    Remove Markdown quote lines and fenced code before judgement detection.

    This reduces false positives when a commenter quotes someone else's verdict.
    It cannot perfectly infer intent, so all detected labels are retained in the
    exported comment record for later inspection.
    """
    kept_lines: list[str] = []
    in_code_block = False

    for line in body.splitlines():
        stripped = line.lstrip()

        if stripped.startswith("```"):
            in_code_block = not in_code_block
            continue
        if in_code_block:
            continue
        if stripped.startswith(">"):
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
    """Return the leading judgement, vote count, shares, and tie information."""
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
    winner = tied[0] if len(tied) == 1 else "TIE"

    complete_counts = {label: int(counts.get(label, 0)) for label in JUDGEMENT_LABELS}
    shares = {
        label: round(complete_counts[label] / total, 6)
        for label in JUDGEMENT_LABELS
    }

    return {
        "winner": winner,
        "is_tie": len(tied) > 1,
        "tied_labels": tied if len(tied) > 1 else [],
        "votes": total,
        "counts": complete_counts,
        "shares": shares,
    }


def summarize_judgements(comments: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Compute both comment-level and unique-user-level community sentiment.

    Comment-level sentiment follows the requested definition: one vote for each
    judgement-bearing comment, using the first detected judgement in that comment.

    Unique-user sentiment prevents prolific commenters from voting repeatedly. For
    each user, their modal judgement is used; ties are broken by their earliest use.
    """
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
        selected = max(
            counts,
            key=lambda label: (counts[label], -first_position[label]),
        )
        user_vote[username] = selected

    unique_user_counts = Counter(user_vote.values())

    return {
        "comment_level": choose_winner(comment_counts),
        "unique_user_level": choose_winner(unique_user_counts),
        "judgement_bearing_comments": len(eligible),
        "unique_judgement_voters": len(user_vote),
        "user_vote_method": "modal judgement per user; earliest judgement breaks ties",
    }


def build_reddit_client(args: argparse.Namespace) -> praw.Reddit:
    """Create an authenticated, read-only PRAW client."""
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
    )
    reddit.read_only = True
    return reddit


def get_submission_candidates(
    reddit: praw.Reddit,
    args: argparse.Namespace,
) -> list[Submission]:
    """Load explicit post URLs or collect posts from an AITA listing."""
    submissions: list[Submission] = []

    if args.post_url:
        for url in args.post_url:
            submissions.append(reddit.submission(url=url))
    else:
        subreddit = reddit.subreddit(args.subreddit)
        listing_method = getattr(subreddit, args.listing)

        if args.listing in {"top", "controversial"}:
            listing = listing_method(limit=args.post_limit, time_filter=args.time_filter)
        else:
            listing = listing_method(limit=args.post_limit)

        submissions.extend(listing)

    # Trigger lazy loading, remove duplicates, and optionally enforce subreddit.
    unique: dict[str, Submission] = {}
    for submission in submissions:
        _ = submission.title
        if not args.allow_other_subreddits and submission.subreddit.display_name.lower() != args.subreddit.lower():
            tqdm.write(
                f"Skipping {submission.id}: belongs to r/{submission.subreddit.display_name}, "
                f"not r/{args.subreddit}."
            )
            continue
        unique[submission.id] = submission

    return list(unique.values())



def resolve_full_comment_tree(submission: Submission, attempts: int = 5) -> list[Comment]:
    """Resolve MoreComments placeholders, retrying transient Reddit API failures."""
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
    """Convert a PRAW Comment into a serializable dictionary."""
    body = comment.body or ""
    judgements = extract_judgements(body)
    author = str(comment.author) if comment.author else None

    return {
        "submission_id": submission_id,
        "comment_id": comment.id,
        "parent_id": getattr(comment, "parent_id", None),
        "depth": getattr(comment, "depth", None),
        "author": author,
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


def scrape_submission(submission: Submission, comment_sort: str) -> tuple[dict[str, Any], set[str]]:
    """Scrape one post and all comments Reddit makes available."""
    submission.comment_sort = comment_sort

    tqdm.write(f"Resolving full comment tree for post {submission.id}: {submission.title[:80]}")
    flat_comments = resolve_full_comment_tree(submission)

    comments: list[dict[str, Any]] = []
    unique_comment_authors: set[str] = set()

    for comment in tqdm(
        flat_comments,
        desc=f"Comments {submission.id}",
        unit="comment",
        leave=False,
    ):
        record = comment_to_record(comment, submission.id)
        comments.append(record)
        if record["author"]:
            unique_comment_authors.add(record["author"])

    update_info = detect_update_or_edit(
        submission.title,
        submission.selftext or "",
        getattr(submission, "edited", False),
    )

    post_record: dict[str, Any] = {
        "submission_id": submission.id,
        "subreddit": submission.subreddit.display_name,
        "title": submission.title,
        "author": str(submission.author) if submission.author else None,
        "selftext": submission.selftext or "",
        "created_utc": submission.created_utc,
        "created_utc_iso": utc_iso(submission.created_utc),
        "score": getattr(submission, "score", None),
        "upvote_ratio": getattr(submission, "upvote_ratio", None),
        "reported_num_comments": getattr(submission, "num_comments", None),
        "scraped_num_comments": len(comments),
        "unique_comment_authors_count": len(unique_comment_authors),
        "unique_comment_authors": sorted(unique_comment_authors, key=str.lower),
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

    return post_record, unique_comment_authors


def count_accessible_user_items(
    items: Iterable[Any],
    target_subreddit: str,
    history_limit: int,
) -> dict[str, Any]:
    """Count accessible items overall and in the target subreddit."""
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
        # Reaching the requested limit means additional history may exist.
        "may_be_truncated": total >= history_limit,
    }


def scrape_user_profile(
    reddit: praw.Reddit,
    username: str,
    target_subreddit: str,
    history_limit: int,
) -> dict[str, Any]:
    """Scrape public karma and accessible posting/commenting history."""
    profile: dict[str, Any] = {
        "username": username,
        "profile_url": f"{REDDIT_BASE_URL}/user/{username}/",
        "status": "ok",
        "error": None,
        "link_karma": None,
        "comment_karma": None,
        "total_karma": None,
        "created_utc": None,
        "created_utc_iso": None,
        "posts_accessible_total": None,
        "posts_accessible_in_aita": None,
        "posts_may_be_truncated": None,
        "comments_accessible_total": None,
        "comments_accessible_in_aita": None,
        "comments_may_be_truncated": None,
        "history_limit_requested_per_type": history_limit,
    }

    try:
        redditor = reddit.redditor(username)

        # Accessing these properties loads the profile and detects suspended users.
        profile["link_karma"] = int(redditor.link_karma)
        profile["comment_karma"] = int(redditor.comment_karma)
        profile["total_karma"] = int(
            getattr(redditor, "total_karma", profile["link_karma"] + profile["comment_karma"])
        )
        profile["created_utc"] = getattr(redditor, "created_utc", None)
        profile["created_utc_iso"] = utc_iso(profile["created_utc"])

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
                "posts_may_be_truncated": post_counts["may_be_truncated"],
                "comments_accessible_total": comment_counts["accessible_total"],
                "comments_accessible_in_aita": comment_counts["accessible_in_aita"],
                "comments_may_be_truncated": comment_counts["may_be_truncated"],
            }
        )

    except (NotFound, Forbidden, Redirect) as exc:
        profile["status"] = "unavailable_or_suspended"
        profile["error"] = f"{type(exc).__name__}: {exc}"
    except (ResponseException, PrawcoreException) as exc:
        profile["status"] = "api_error"
        profile["error"] = f"{type(exc).__name__}: {exc}"
    except Exception as exc:  # Keep one bad profile from aborting the dataset.
        profile["status"] = "unexpected_error"
        profile["error"] = f"{type(exc).__name__}: {exc}"

    return profile


def csv_value(value: Any) -> Any:
    """Serialize nested values cleanly inside CSV cells."""
    if isinstance(value, (dict, list, tuple, set)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return value


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    """Write dictionaries to UTF-8 CSV."""
    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: csv_value(row.get(key)) for key in fieldnames})


def save_results(dataset: dict[str, Any], output_dir: Path) -> None:
    """Save nested JSON plus normalized CSV tables."""
    output_dir.mkdir(parents=True, exist_ok=True)

    json_path = output_dir / "aita_scrape.json"
    with json_path.open("w", encoding="utf-8") as file:
        json.dump(dataset, file, indent=2, ensure_ascii=False)

    posts = dataset["posts"]
    users = dataset["users"]
    comments = [comment for post in posts for comment in post["comments"]]

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
        "selftext",
        "created_utc",
        "created_utc_iso",
        "score",
        "upvote_ratio",
        "reported_num_comments",
        "scraped_num_comments",
        "unique_comment_authors_count",
        "unique_comment_authors",
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
    user_fields = [
        "username",
        "profile_url",
        "status",
        "error",
        "link_karma",
        "comment_karma",
        "total_karma",
        "created_utc",
        "created_utc_iso",
        "posts_accessible_total",
        "posts_accessible_in_aita",
        "posts_may_be_truncated",
        "comments_accessible_total",
        "comments_accessible_in_aita",
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
        *[f"{label}_{suffix}" for label in JUDGEMENT_LABELS for suffix in ("count", "share")],
    ]

    write_csv(output_dir / "posts.csv", post_rows, post_fields)
    write_csv(output_dir / "comments.csv", comments, comment_fields)
    write_csv(output_dir / "users.csv", users, user_fields)
    write_csv(output_dir / "judgement_summary.csv", summary_rows, summary_fields)

    print(f"Saved JSON and CSV outputs to: {output_dir.resolve()}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Scrape AITA posts, complete comment trees, judgements, and public user activity."
    )
    parser.add_argument(
        "--post-url",
        action="append",
        help="Specific Reddit post URL. Repeat this option to scrape multiple posts.",
    )
    parser.add_argument("--subreddit", default=AITA_SUBREDDIT)
    parser.add_argument("--post-limit", type=int, default=10)
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
        help="Maximum accessible submissions and comments requested per user, per type.",
    )
    parser.add_argument(
        "--skip-user-profiles",
        action="store_true",
        help="Skip expensive per-user history collection.",
    )
    parser.add_argument(
        "--allow-other-subreddits",
        action="store_true",
        help="Allow explicit post URLs outside the configured subreddit.",
    )
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--client-id")
    parser.add_argument("--client-secret")
    parser.add_argument("--user-agent")
    parser.add_argument(
        "--user-delay",
        type=float,
        default=0.0,
        help="Optional extra delay in seconds after each user profile.",
    )

    args = parser.parse_args()

    if args.post_limit < 1:
        parser.error("--post-limit must be at least 1")
    if args.user_history_limit < 1:
        parser.error("--user-history-limit must be at least 1")
    if args.user_delay < 0:
        parser.error("--user-delay cannot be negative")

    return args


def main() -> None:
    args = parse_args()
    reddit = build_reddit_client(args)
    submissions = get_submission_candidates(reddit, args)

    if not submissions:
        print("No matching submissions were found.")
        return

    posts: list[dict[str, Any]] = []
    all_comment_authors: set[str] = set()

    for submission in tqdm(submissions, desc="Posts", unit="post"):
        try:
            post_record, comment_authors = scrape_submission(submission, args.comment_sort)
            posts.append(post_record)
            all_comment_authors.update(comment_authors)
        except (Forbidden, NotFound, Redirect, ResponseException, PrawcoreException) as exc:
            tqdm.write(f"Skipping post {submission.id} after Reddit API error: {exc}")
        except Exception as exc:
            tqdm.write(f"Skipping post {submission.id} after unexpected error: {type(exc).__name__}: {exc}")

    user_profiles: list[dict[str, Any]] = []
    if not args.skip_user_profiles:
        for username in tqdm(
            sorted(all_comment_authors, key=str.lower),
            desc="User profiles",
            unit="user",
        ):
            profile = scrape_user_profile(
                reddit,
                username,
                args.subreddit,
                args.user_history_limit,
            )
            user_profiles.append(profile)
            if args.user_delay:
                time.sleep(args.user_delay)

    dataset = {
        "metadata": {
            "scraped_at_utc": datetime.now(timezone.utc).isoformat(),
            "subreddit": args.subreddit,
            "post_count": len(posts),
            "unique_comment_authors_count": len(all_comment_authors),
            "user_profiles_scraped": not args.skip_user_profiles,
            "user_history_limit_requested_per_type": args.user_history_limit,
            "judgement_labels": list(JUDGEMENT_LABELS),
            "community_sentiment_definition": {
                "comment_level": "first explicit judgement in every judgement-bearing comment",
                "unique_user_level": "modal judgement per user; earliest judgement breaks ties",
            },
            "limitations": [
                "Removed, deleted, private, quarantined, or otherwise unavailable content cannot be recovered.",
                "User activity counts cover accessible Reddit listings and are not guaranteed lifetime totals.",
                "Judgement extraction is rule-based and may still misclassify sarcasm, negation, or unusual formatting.",
            ],
        },
        "posts": posts,
        "users": user_profiles,
    }

    save_results(dataset, Path(args.output_dir))

    total_comments = sum(post["scraped_num_comments"] for post in posts)
    total_votes = sum(
        post["judgement_summary"]["comment_level"]["votes"] for post in posts
    )
    print(
        f"Finished: {len(posts)} posts, {total_comments} comments, "
        f"{total_votes} judgement-bearing comments, {len(user_profiles)} user profiles."
    )


if __name__ == "__main__":
    main()
