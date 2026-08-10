import requests
import pandas as pd
from datetime import datetime
import time
import re
import html

# --- CONFIGURATION ---
SITE = "tex"
TARGET_QUESTIONS = 5000
PAGE_SIZE = 100

BASE_URL = "https://api.stackexchange.com/2.3"
COMMON_PARAMS = {"site": SITE, "filter": "withbody", "pagesize": PAGE_SIZE}


def fetch_data():
    questions_data = []
    question_ids = []

    print(f"1. Fetching {TARGET_QUESTIONS} questions from {SITE}.stackexchange.com...")

    total_pages = (TARGET_QUESTIONS // PAGE_SIZE)

    for page in range(1, total_pages + 1):
        params = {**COMMON_PARAMS, "page": page, "order": "desc", "sort": "votes"}
        response = requests.get(f"{BASE_URL}/questions", params=params)

        if response.status_code != 200:
            print(f"API Error: {response.status_code}")
            break

        items = response.json().get('items', [])
        if not items:
            break

        for q in items:
            # Failsafe to ensure we don't exceed the target
            if len(questions_data) >= TARGET_QUESTIONS:
                break

            q_id = q['question_id']
            question_ids.append(q_id)

            # Extract HTML body
            body_html = q.get('body', '').lower()

            has_image = "<img" in body_html
            code_block_count = body_html.count("<pre")  # Stack Exchange wraps code in <pre> tags
            link_count = body_html.count("<a href")

            # Strip HTML tags to get a clean word count
            clean_text = re.sub(r'<[^>]+>', ' ', body_html)
            word_count = len(clean_text.split())
            title_word_count = len(q.get('title', '').split())

            title = q.get('title', '')
            clean_title = html.unescape(title).strip()

            # Check if the title ends with a question mark (returns True/False)
            is_question_format = clean_title.endswith('?')

            # Count LaTeX comments in code blocks
            code_blocks = re.findall(r'<pre.*?</pre>', body_html, flags=re.DOTALL)

            latex_comment_count = 0
            for block in code_blocks:
                # Count '%' characters that are not preceded by a backslash ('\%')
                comments_in_block = re.findall(r'(?<!\\)%', block)
                latex_comment_count += len(comments_in_block)

            # Get user metadata
            owner_info = q.get('owner', {})

            user_id = owner_info.get('user_id', 'Unknown')
            user_reputation = owner_info.get('reputation', 0)  # Defaults to 0 if the user was deleted
            user_type = owner_info.get('user_type', 'Unknown')  # e.g., 'registered', 'unregistered'

            # Extracting badges
            badges = owner_info.get('badge_counts', {})
            gold_badges = badges.get('gold', 0)
            silver_badges = badges.get('silver', 0)
            bronze_badges = badges.get('bronze', 0)

            # Checking if they have a custom profile image (Stack Exchange uses Gravatar defaults)
            profile_image = owner_info.get('profile_image', '')
            has_custom_avatar = "identicon" not in profile_image

            # Accept rate (Note: Stack Exchange sometimes limits this field, so default to 0 or None)
            accept_rate = owner_info.get('accept_rate', None)

            questions_data.append({
                "Question_ID": q_id,
                "User_ID": user_id,
                "Creation_Date": q.get('creation_date'),
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
                "User_Reputation": user_reputation,
                "User_Type": user_type,
                "Gold_Badges": gold_badges,
                "Silver_Badges": silver_badges,
                "Bronze_Badges": bronze_badges,
                "Has_Custom_Avatar": has_custom_avatar,
                "Accept_Rate": accept_rate,

                # --- DEPENDENT VARIABLES ---
                "Subjective_Score": q.get('score', 0),
                "Answer_Count": q.get('answer_count', 0),
                "View_Count": q.get('view_count', 0),
                "Link": q.get('link')
            })

        print(f"   Gathered {len(questions_data)} questions...")
        time.sleep(1)

    print("\n2. Fetching all answers to calculate Time to First Answer...")
    answers_dict = {}

    for i in range(0, len(question_ids), 100):
        chunk = question_ids[i:i + 100]
        ids_string = ";".join(map(str, chunk))

        response = requests.get(f"{BASE_URL}/questions/{ids_string}/answers", params=COMMON_PARAMS)
        if response.status_code == 200:
            for ans in response.json().get('items', []):
                q_id = ans['question_id']
                if q_id not in answers_dict:
                    answers_dict[q_id] = []
                answers_dict[q_id].append(ans['creation_date'])
        time.sleep(1)

    print("3. Fetching all comments on the questions...")
    comments_dict = {}

    for i in range(0, len(question_ids), 100):
        chunk = question_ids[i:i + 100]
        ids_string = ";".join(map(str, chunk))

        response = requests.get(f"{BASE_URL}/questions/{ids_string}/comments", params=COMMON_PARAMS)
        if response.status_code == 200:
            for comment in response.json().get('items', []):
                q_id = comment['post_id']
                comments_dict[q_id] = comments_dict.get(q_id, 0) + 1
        time.sleep(1)

    print("\n4. Merging data and calculating objective metrics...")
    for q in questions_data:
        q_id = q["Question_ID"]

        q["Objective_Comment_Count"] = comments_dict.get(q_id, 0)

        q_creation = q["Creation_Date"]
        answers = answers_dict.get(q_id, [])

        if answers:
            first_answer_time = min(answers)
            time_diff_seconds = first_answer_time - q_creation
            q["Objective_Time_To_First_Answer_Mins"] = round(time_diff_seconds / 60, 2)
        else:
            q["Objective_Time_To_First_Answer_Mins"] = None

        q["Creation_Date_Readable"] = datetime.fromtimestamp(q_creation).strftime('%Y-%m-%d %H:%M:%S')

    return questions_data


data = fetch_data()
df = pd.DataFrame(data)

columns_order = [
    "Question_ID", "User_ID", "User_Reputation", "User_Type",
    "Gold_Badges", "Silver_Badges", "Bronze_Badges", "Has_Custom_Avatar", "Accept_Rate",
    "Tags", "Tag_Count", "Creation_Date_Readable",
    "Has_Image", "Word_Count", "Code_Block_Count", "Link_Count",
    "Title_Word_Count", "LaTeX_Comment_Count", "Is_Question_Format",
    "Subjective_Score", "Objective_Comment_Count",
    "Objective_Time_To_First_Answer_Mins", "Answer_Count", "View_Count", "Link"
]
df = df[columns_order]

output_file = "stackexchange_enhanced_dataset.csv"
df.to_csv(output_file, index=False, encoding="utf-8")
print(f"\nSuccess! Saved exactly {len(df)} questions with new attributes to {output_file}")