import json
import os
from typing import List, Dict, Any
import google.generativeai as Client
from github import Github
import difflib
import requests
import fnmatch
from unidiff import Hunk, PatchedFile, PatchSet

GITHUB_TOKEN = os.environ["GITHUB_TOKEN"]

# Initialize GitHub and Gemini clients
gh = Github(GITHUB_TOKEN)
gemini_client = Client.configure(api_key=os.environ.get('GEMINI_API_KEY'))


class PRDetails:
    def __init__(self, owner: str, repo: str, pull_number: int, title: str, description: str):
        self.owner = owner
        self.repo = repo
        self.pull_number = pull_number
        self.title = title
        self.description = description


def get_pr_details() -> PRDetails:
    """Retrieves details of the pull request from GitHub Actions event payload."""
    with open(os.environ["GITHUB_EVENT_PATH"], "r") as f:
        event_data = json.load(f)

    # Handle comment trigger differently from direct PR events
    if "issue" in event_data and "pull_request" in event_data["issue"]:
        # For comment triggers, we need to get the PR number from the issue
        pull_number = event_data["issue"]["number"]
        repo_full_name = event_data["repository"]["full_name"]
    else:
        # Original logic for direct PR events
        pull_number = event_data["number"]
        repo_full_name = event_data["repository"]["full_name"]

    owner, repo = repo_full_name.split("/")

    repo = gh.get_repo(repo_full_name)
    pr = repo.get_pull(pull_number)

    return PRDetails(owner, repo.name, pull_number, pr.title, pr.body)


def get_diff(owner: str, repo: str, pull_number: int) -> str:
    """Fetches the diff of the pull request from GitHub API."""
    # Use the correct repository name format
    repo_name = f"{owner}/{repo}"
    print(f"Attempting to get diff for: {repo_name} PR#{pull_number}")

    repo = gh.get_repo(repo_name)
    pr = repo.get_pull(pull_number)

    # Use the GitHub API URL directly
    api_url = f"https://api.github.com/repos/{repo_name}/pulls/{pull_number}"

    headers = {
        'Authorization': f'Bearer {GITHUB_TOKEN}',  # Changed to Bearer format
        'Accept': 'application/vnd.github.v3.diff'
    }

    response = requests.get(f"{api_url}.diff", headers=headers)

    if response.status_code == 200:
        diff = response.text
        print(f"Retrieved diff length: {len(diff) if diff else 0}")
        return diff
    else:
        print(f"Failed to get diff. Status code: {response.status_code}")
        print(f"Response content: {response.text}")
        print(f"URL attempted: {api_url}.diff")
        return ""


def analyze_code(parsed_diff: List[Dict[str, Any]], pr_details: PRDetails) -> List[Dict[str, Any]]:
    """Analyzes the code changes using Gemini and generates review comments."""
    print("Starting analyze_code...")
    print(f"Number of files to analyze: {len(parsed_diff)}")
    comments = []
    #print(f"Initial comments list: {comments}")

    for file_data in parsed_diff:
        file_path = file_data.get('path', '')
        print(f"\nProcessing file: {file_path}")

        if not file_path or file_path == "/dev/null":
            continue

        class FileInfo:
            def __init__(self, path):
                self.path = path

        file_info = FileInfo(file_path)

        hunks = file_data.get('hunks', [])
        print(f"Hunks in file: {len(hunks)}")

        for hunk_data in hunks:
            print(f"\nHunk content: {json.dumps(hunk_data, indent=2)}")
            hunk_lines = hunk_data.get('lines', [])
            print(f"Number of lines in hunk: {len(hunk_lines)}")

            if not hunk_lines:
                continue

            hunk = Hunk()
            hunk.source_start = 1
            hunk.source_length = len(hunk_lines)
            hunk.target_start = 1
            hunk.target_length = len(hunk_lines)
            hunk.content = '\n'.join(hunk_lines)

            prompt = create_prompt(file_info, hunk, pr_details)
            print("Sending prompt to Gemini...")
            ai_response = get_ai_response(prompt)
            print(f"AI response received: {ai_response}")

            if ai_response:
                new_comments = create_comment(file_info, hunk, ai_response, pr_details.owner, pr_details.repo)
                print(f"Comments created from AI response: {new_comments}")
                if new_comments:
                    comments.extend(new_comments)
                    print(f"Updated comments list: {comments}")

    print(f"\nFinal comments list: {comments}")
    return comments


def create_prompt(file: PatchedFile, hunk: Hunk, pr_details: PRDetails) -> str:
    """Creates the prompt for the Gemini model."""
    return f"""Your task is reviewing pull requests. Instructions:
    - Provide the response in following JSON format:  {{"reviews": [{{"lineNumber":  <line_number>, "reviewComment": "<review comment>"}}]}}
    - Provide comments and suggestions ONLY if there is something to improve, otherwise "reviews" should be an empty array.
    - Use GitHub Markdown in comments
    - Focus on bugs, security issues, and performance problems
    - Please answer "Korean".
    - IMPORTANT: NEVER suggest adding comments to the code

Review the following code diff in the file "{file.path}" and take the pull request title and description into account when writing the response.

Pull request title: {pr_details.title}
Pull request description:

---
{pr_details.description or 'No description provided'}
---

Git diff to review:

```diff
{hunk.content}
```
"""


def get_ai_response(prompt: str) -> List[Dict[str, str]]:
    """Sends the prompt to Gemini API and retrieves the response."""
    # Use 'gemini-2.0-flash-001' as a fallback default value if the environment variable isn't set
    gemini_model = Client.GenerativeModel(os.environ.get('GEMINI_MODEL', 'gemini-2.0-flash-001'))

    generation_config = {
        "max_output_tokens": 8192,
        "temperature": 0.8,
        "top_p": 0.95,
    }

    print("===== The promt sent to Gemini is: =====")
    print(prompt)
    try:
        response = gemini_model.generate_content(prompt, generation_config=generation_config)

        response_text = response.text.strip()
        if response_text.startswith('```json'):
            response_text = response_text[7:]  # Remove ```json
        if response_text.endswith('```'):
            response_text = response_text[:-3]  # Remove ```
        response_text = response_text.strip()

        print(f"Cleaned response text: {response_text}")

        try:
            data = json.loads(response_text)
            print(f"Parsed JSON data: {data}")

            if "reviews" in data and isinstance(data["reviews"], list):
                reviews = data["reviews"]
                valid_reviews = []
                for review in reviews:
                    if "lineNumber" in review and "reviewComment" in review:
                        valid_reviews.append(review)
                    else:
                        print(f"Invalid review format: {review}")
                return valid_reviews
            else:
                print("Error: Response doesn't contain valid 'reviews' array")
                print(f"Response content: {data}")
                return []
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON response: {e}")
            print(f"Raw response: {response_text}")
            return []
    except Exception as e:
        print(f"Error during Gemini API call: {e}")
        return []

class FileInfo:
    """Simple class to hold file information."""
    def __init__(self, path: str):
        self.path = path

def create_comment(file: FileInfo, hunk: Hunk, ai_responses: List[Dict[str, str]], owner: str, repo: str) -> List[Dict[str, Any]]:
    """Creates comment objects from AI responses."""
    print("AI responses in create_comment:", ai_responses)
    print(f"Hunk details - start: {hunk.source_start}, length: {hunk.source_length}")
    print(f"Hunk content:\n{hunk.content}")

    file_lines = get_file_content(owner, repo, file.path)
    comments = []
    for ai_response in ai_responses:
        try:
            line_number = int(ai_response["lineNumber"])
            full_line_number = get_full_line_number(file_lines, hunk, line_number)

            print(f"Original AI suggested line: {line_number}")

            # Ensure the line number is within the hunk's range
            if line_number < 1 or line_number > hunk.source_length:
                print(f"Warning: Line number {line_number} is outside hunk range")
                continue
            
            comment = {
                "body": ai_response["reviewComment"],
                "path": file.path,
                "position": line_number,
                "full_line_number": full_line_number,
            }
            print(f"Created comment: {json.dumps(comment, indent=2)}")
            comments.append(comment)

        except (KeyError, TypeError, ValueError) as e:
            print(f"Error creating comment from AI response: {e}, Response: {ai_response}")
    return comments


    # file_lines = get_file_content(owner, repo, file.path)
def get_file_content(owner: str, repo: str, file_path: str, ref: str = "main") -> List[str]:
    """Fetches the file content from GitHub at a given reference (default: main branch)."""
    repo_obj = gh.get_repo(f"{owner}/{repo}")
    try:
        file_content = repo_obj.get_contents(file_path, ref=ref)
        return file_content.decoded_content.decode("utf-8").splitlines()
    except Exception as e:
        print(f"Error fetching file content: {file_path} on {ref}: {e}")
        return []

#  full_line_number = get_full_line_number(file_lines, hunk, line_number)
def get_full_line_number(file_lines: List[str], hunk: Hunk, diff_line_number: int) -> int:
    """
    Converts a diff-based line number to a full-file line number.
    
    - `diff_line_number`: diff 내부에서 AI가 반환한 상대적인 줄 번호
    - `hunk.source_start`: 원본 파일에서 변경이 시작되는 줄 번호
    - `hunk.lines`: 변경된 줄들을 포함하는 리스트
    """
    source_start = hunk.source_start  # 원본 파일에서 변경된 코드의 시작 줄
    target_start = hunk.target_start  # 변경 후 파일에서 시작하는 줄

    # 실제 파일의 줄 번호를 추적
    absolute_line_number = source_start  
    current_diff_line = 0

    for line in hunk.content.split("\n"):
        # diff 내에서 특정 줄(`diff_line_number`)을 찾으면 반환
        if current_diff_line == diff_line_number:
            return absolute_line_number
        
        # 삭제된 줄 (-)인 경우, 실제 파일 줄 번호는 증가하지 않음
        if line.startswith("-"):
            continue

        # 추가된 줄(+)이 아닌 경우에만 원본 파일의 줄 번호 증가
        if not line.startswith("+"):
            absolute_line_number += 1

        current_diff_line += 1  # diff 내부의 줄 번호 증가

    print(f"Warning: Couldn't find exact match for diff_line_number {diff_line_number}")
    return absolute_line_number  # 기본적으로 변환된 줄 번호 반환

# def create_review_comment(
#     owner: str,
#     repo: str,
#     pull_number: int,
#     comments: List[Dict[str, Any]],
#     batch_size: int = 10,
# ):
#     """Submits the review comments to the GitHub API."""
#     print(f"Attempting to create {len(comments)} review comments")
#     print(f"Comments content: {json.dumps(comments, indent=2)}")

#     repo = gh.get_repo(f"{owner}/{repo}")
#     pr = repo.get_pull(pull_number)
#     total_batches = (len(comments) + batch_size - 1) 
#     try:
#         # Create the review with only the required fields
#         for i in range(total_batches):
#             start = i * batch_size
#             batch = comments[start: start + batch_size]
#             body_msg = f"Gemini AI Code Reviewer Comments ({i+1}/{total_batches})"
#             review = pr.create_review(
#                 body=body_msg,
#                 comments=batch,
#                 event="COMMENT"
#             )
#             print(f"Review created successfully with ID: {review.id}")

#     except Exception as e:
#         print(f"Error creating review: {str(e)}")
#         print(f"Error type: {type(e)}")
#         print(f"Review payload: {comments}")

def create_review_comment_batch(owner: str, repo: str, pull_number: int, batch: List[Dict[str, Any]], batch_index: int, total_batches: int):
    """Creates a review for a batch of comments, combining them into the review body."""
    repo_obj = gh.get_repo(f"{owner}/{repo}")
    pr = repo_obj.get_pull(pull_number)
    
    # Compose a body message containing all batch comments
    body_lines = [f"Gemini AI Code Reviewer Comments Batch {batch_index+1}/{total_batches}:"]
    for comment in batch:
        body_lines.append(f"**File:** `{comment['path']}`  |  **Line:** {comment['full_line_number']}")
        body_lines.append("")
        body_lines.append(f"> {comment['body']}")
        body_lines.append("")  # 빈 줄로 구분
    body_msg = "\n".join(body_lines)
    
    review = pr.create_review(
        body=body_msg,
        event="COMMENT"
    )
    print(f"Created review batch {batch_index+1} with ID: {review.id}")
    return review

def create_review_comment(owner: str, repo: str, pull_number: int, comments: List[Dict[str, Any]], batch_size: int = 10):
    """Submits review comments in batches, combining each batch into one review body."""
    total_batches = (len(comments) + batch_size - 1) // batch_size
    print(f"Total comments: {len(comments)}; creating {total_batches} review batches (batch size: {batch_size})")
    for i in range(0, len(comments), batch_size):
        batch = comments[i:i+batch_size]
        create_review_comment_batch(owner, repo, pull_number, batch, i // batch_size, total_batches)


def parse_diff(diff_str: str) -> List[Dict[str, Any]]:
    """Parses the diff string and returns a structured format."""
    files = []
    current_file = None
    current_hunk = None

    for line in diff_str.splitlines():
        if line.startswith('diff --git'):
            if current_file:
                files.append(current_file)
            current_file = {'path': '', 'hunks': []}

        elif line.startswith('--- a/'):
            if current_file:
                current_file['path'] = line[6:]

        elif line.startswith('+++ b/'):
            if current_file:
                current_file['path'] = line[6:]

        elif line.startswith('@@'):
            if current_file:
                current_hunk = {'header': line, 'lines': []}
                current_file['hunks'].append(current_hunk)

        elif current_hunk is not None:
            current_hunk['lines'].append(line)

    if current_file:
        files.append(current_file)

    return files



def main():
    """Main function to execute the code review process."""
    pr_details = get_pr_details()
    event_data = json.load(open(os.environ["GITHUB_EVENT_PATH"], "r"))

    event_name = os.environ.get("GITHUB_EVENT_NAME")
    if event_name == "issue_comment":
        # Process comment trigger
        if not event_data.get("issue", {}).get("pull_request"):
            print("Comment was not on a pull request")
            return

        diff = get_diff(pr_details.owner, pr_details.repo, pr_details.pull_number)
        if not diff:
            print("There is no diff found")
            return

        parsed_diff = parse_diff(diff)

        # Get and clean exclude patterns, handle empty input
        exclude_patterns_raw = os.environ.get("INPUT_EXCLUDE", "")
        print(f"Raw exclude patterns: {exclude_patterns_raw}")  # Debug log
        
        # Only split if we have a non-empty string
        exclude_patterns = []
        if exclude_patterns_raw and exclude_patterns_raw.strip():
            exclude_patterns = [p.strip() for p in exclude_patterns_raw.split(",") if p.strip()]
        print(f"Exclude patterns: {exclude_patterns}")  # Debug log

        # Filter files before analysis
        filtered_diff = []
        for file in parsed_diff:
            file_path = file.get('path', '')
            should_exclude = any(fnmatch.fnmatch(file_path, pattern) for pattern in exclude_patterns)
            if should_exclude:
                print(f"Excluding file: {file_path}")  # Debug log
                continue
            filtered_diff.append(file)

        print(f"Files to analyze after filtering: {[f.get('path', '') for f in filtered_diff]}")  # Debug log
        
        comments = analyze_code(filtered_diff, pr_details)
        if comments:
            try:
                create_review_comment(
                    pr_details.owner, pr_details.repo, pr_details.pull_number, comments
                )
            except Exception as e:
                print("Error in create_review_comment:", e)
    else:
        print("Unsupported event:", os.environ.get("GITHUB_EVENT_NAME"))
        return


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print("Error:", error)
