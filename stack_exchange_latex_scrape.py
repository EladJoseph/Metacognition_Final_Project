import requests
import pandas as pd
from datetime import datetime
import time
import re
import html

# --- CONFIGURATION ---
SITE = "tex"
TARGET_QUESTIONS_PER_CHUNK = 1000 # 5000 Questions in total
PAGE_SIZE = 100

CUSTOM_QUESTION_FILTER = "!22Zfkj9PJFRtEtTc9YBFQ"

BASE_URL = "https://api.stackexchange.com/2.3"
COMMON_PARAMS = {"site": SITE, "pagesize": PAGE_SIZE}

# We split the timeline to bypass the 2500-question API limit
DATE_CHUNKS = [
    ("2000-01-01", "2015-12-31"),
    ("2016-01-01", "2018-12-31"),
    ("2019-01-01", "2021-12-31"),
    ("2022-01-01", "2023-12-31"),
    ("2024-01-01", "2026-12-31")
]


def to_unix(date_str):
    return int(datetime.strptime(date_str, "%Y-%m-%d").timestamp())


def fetch_data():
    questions_data = []
    question_ids = []

    print(
        f"1. Fetching {len(DATE_CHUNKS) * TARGET_QUESTIONS_PER_CHUNK} questions from {SITE}.stackexchange.com in date chunks...")

    current_timestamp = time.time()  # Used for Post Age calculation

    for start_date, end_date in DATE_CHUNKS:
        print(f"\n--- Fetching Top Questions from {start_date} to {end_date} ---")

        fromdate = to_unix(start_date)
        todate = to_unix(end_date)

        pages_to_fetch = TARGET_QUESTIONS_PER_CHUNK // PAGE_SIZE

        for page in range(1, pages_to_fetch + 1):
            params = {
                **COMMON_PARAMS,
                "filter": CUSTOM_QUESTION_FILTER,
                "page": page,
                "order": "desc",
                "sort": "votes",
                "fromdate": fromdate,
                "todate": todate
            }

            response = requests.get(f"{BASE_URL}/questions", params=params)

            if response.status_code != 200:
                print(f"API Error: {response.status_code} - {response.text}")
                break

            items = response.json().get('items', [])
            if not items:
                break

            for q in items:
                q_id = q['question_id']
                question_ids.append(q_id)

                # Extract HTML body
                body_html = q.get('body', '').lower()

                has_image = "<img" in body_html
                code_block_count = body_html.count("<pre")
                link_count = body_html.count("<a href")

                # Clean text for word counts
                clean_text = re.sub(r'<[^>]+>', ' ', body_html)
                word_count = len(clean_text.split())
                title_word_count = len(q.get('title', '').split())

                title = q.get('title', '')
                clean_title = html.unescape(title).strip()
                is_question_format = clean_title.endswith('?')

                # Count LaTeX comments
                code_blocks = re.findall(r'<pre.*?</pre>', body_html, flags=re.DOTALL)
                latex_comment_count = 0
                for block in code_blocks:
                    comments_in_block = re.findall(r'(?<!\\)%', block)
                    latex_comment_count += len(comments_in_block)

                # User Metadata
                owner_info = q.get('owner', {})
                user_id = owner_info.get('user_id', 'Unknown')
                user_reputation = owner_info.get('reputation', 0)
                user_type = owner_info.get('user_type', 'Unknown')
                accept_rate = owner_info.get('accept_rate', None)

                badges = owner_info.get('badge_counts', {})
                gold_badges = badges.get('gold', 0)
                silver_badges = badges.get('silver', 0)
                bronze_badges = badges.get('bronze', 0)

                profile_image = owner_info.get('profile_image', '')
                has_custom_avatar = "identicon" not in profile_image

                # Post Metadata
                q_creation = q.get('creation_date')
                post_age_days = round((current_timestamp - q_creation) / (60 * 60 * 24), 2)

                questions_data.append({
                    "Question_ID": q_id,
                    "User_ID": user_id,
                    "Creation_Date": q_creation,
                    "Tags": ", ".join(q.get('tags', [])),
                    "Tag_Count": len(q.get('tags', [])),

                    # --- INDEPENDENT VARIABLES / ATTRIBUTES ---
                    "Has_Image": has_image,
                    "Word_Count": word_count,
                    "Code_Block_Count": code_block_count,
                    "Link_Count": link_count,
                    "Title_Word_Count": title_word_count,
                    "LaTeX_Comment_Count": latex_comment_count,
                    "Is_Question_Format": is_question_format,

                    # --- USER ATTRIBUTES ---
                    "User_Reputation": user_reputation,
                    "User_Type": user_type,
                    "Gold_Badges": gold_badges,
                    "Silver_Badges": silver_badges,
                    "Bronze_Badges": bronze_badges,
                    "Has_Custom_Avatar": has_custom_avatar,
                    "Accept_Rate": accept_rate,

                    # --- POST METADATA ---
                    "Post_Age_Days": post_age_days,
                    "Subjective_Score": q.get('score', 0),
                    "Up_Vote_Count": q.get('up_vote_count', None),
                    "Down_Vote_Count": q.get('down_vote_count', None),
                    "Favorite_Count": q.get('favorite_count', 0),
                    "View_Count": q.get('view_count', 0),
                    "Answer_Count": q.get('answer_count', 0),
                    "Is_Answered": q.get('is_answered', False),
                    "Link": q.get('link')
                })

            print(f"   Gathered {len(questions_data)} total questions so far...")

            # Sleep to respect API rate limits
            time.sleep(1)

    print("\n2. Fetching answers for metadata and Time to First Answer...")
    answers_dict = {}

    for i in range(0, len(question_ids), 100):
        chunk = question_ids[i:i + 100]
        ids_string = ";".join(map(str, chunk))

        params = {**COMMON_PARAMS, "filter": "withbody"}
        response = requests.get(f"{BASE_URL}/questions/{ids_string}/answers", params=params)

        if response.status_code == 200:
            for ans in response.json().get('items', []):
                q_id = ans['question_id']
                if q_id not in answers_dict:
                    answers_dict[q_id] = {'dates': [], 'scores': [], 'is_accepted': False, 'total_words': 0}

                answers_dict[q_id]['dates'].append(ans['creation_date'])
                answers_dict[q_id]['scores'].append(ans.get('score', 0))

                if ans.get('is_accepted', False):
                    answers_dict[q_id]['is_accepted'] = True

                clean_ans = re.sub(r'<[^>]+>', ' ', ans.get('body', '')).strip()
                answers_dict[q_id]['total_words'] += len(clean_ans.split())
        time.sleep(1)

    print("3. Fetching comments for objective friction metrics...")
    comments_dict = {}

    for i in range(0, len(question_ids), 100):
        chunk = question_ids[i:i + 100]
        ids_string = ";".join(map(str, chunk))

        params = {**COMMON_PARAMS, "filter": "withbody"}
        response = requests.get(f"{BASE_URL}/questions/{ids_string}/comments", params=params)

        if response.status_code == 200:
            for comment in response.json().get('items', []):
                q_id = comment['post_id']
                if q_id not in comments_dict:
                    comments_dict[q_id] = {'count': 0, 'total_words': 0}

                comments_dict[q_id]['count'] += 1

                clean_comment = re.sub(r'<[^>]+>', ' ', comment.get('body', '')).strip()
                comments_dict[q_id]['total_words'] += len(clean_comment.split())
        time.sleep(1)

    print("\n4. Merging data and calculating final objective metrics...")
    for q in questions_data:
        q_id = q["Question_ID"]

        # --- Comments Merge ---
        c_data = comments_dict.get(q_id, {'count': 0, 'total_words': 0})
        q["Objective_Comment_Count"] = c_data['count']
        q["Objective_Total_Comment_Words"] = c_data['total_words']
        q["Objective_Avg_Comment_Words"] = round(c_data['total_words'] / c_data['count'], 2) if c_data[
                                                                                                    'count'] > 0 else 0

        # --- Answers Merge ---
        a_data = answers_dict.get(q_id, {'dates': [], 'scores': [], 'is_accepted': False, 'total_words': 0})

        if a_data['dates']:
            first_answer_time = min(a_data['dates'])
            time_diff_seconds = first_answer_time - q["Creation_Date"]
            q["Objective_Time_To_First_Answer_Mins"] = round(time_diff_seconds / 60, 2)
            q["Max_Answer_Score"] = max(a_data['scores'])
            q["Avg_Answer_Score"] = round(sum(a_data['scores']) / len(a_data['scores']), 2)
        else:
            q["Objective_Time_To_First_Answer_Mins"] = None
            q["Max_Answer_Score"] = 0
            q["Avg_Answer_Score"] = 0

        q["Has_Accepted_Answer"] = a_data['is_accepted']
        q["Total_Answer_Words"] = a_data['total_words']

        q["Creation_Date_Readable"] = datetime.fromtimestamp(q["Creation_Date"]).strftime('%Y-%m-%d %H:%M:%S')

    return questions_data


data = fetch_data()
df = pd.DataFrame(data)

columns_order = [
    "Question_ID", "User_ID", "User_Reputation", "User_Type",
    "Gold_Badges", "Silver_Badges", "Bronze_Badges", "Has_Custom_Avatar", "Accept_Rate",
    "Tags", "Tag_Count", "Creation_Date_Readable", "Post_Age_Days",

    # Cues
    "Has_Image", "Word_Count", "Code_Block_Count", "Link_Count",
    "Title_Word_Count", "LaTeX_Comment_Count", "Is_Question_Format",

    # Question Performance
    "Subjective_Score", "Up_Vote_Count", "Down_Vote_Count",
    "View_Count", "Is_Answered",

    # Comments & Friction
    "Objective_Comment_Count", "Objective_Total_Comment_Words", "Objective_Avg_Comment_Words",

    # Answers & Resolution
    "Objective_Time_To_First_Answer_Mins", "Answer_Count", "Has_Accepted_Answer",
    "Max_Answer_Score", "Avg_Answer_Score", "Total_Answer_Words", "Link"
]

df = df[columns_order]

output_file = "stackexchange_enhanced_dataset.csv"
df.to_csv(output_file, index=False, encoding="utf-8")
print(f"\nSuccess! Saved exactly {len(df)} enriched questions to {output_file}")