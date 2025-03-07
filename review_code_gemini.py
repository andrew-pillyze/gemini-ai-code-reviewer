import json
import os
from typing import List, Dict, Any
import google.generativeai as Client
from github import Github
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

    pull_number = event_data["number"]
    repo_full_name = event_data["repository"]["full_name"]
    owner, repo = repo_full_name.split("/")

    repo = gh.get_repo(repo_full_name)
    pr = repo.get_pull(pull_number)

    return PRDetails(owner, repo.name, pull_number, pr.title, pr.body)


def get_file_content(owner: str, repo: str, file_path: str, ref: str = "main") -> List[str]:
    """Fetches the file content from GitHub at a given reference (default: main branch)."""
    repo_obj = gh.get_repo(f"{owner}/{repo}")
    try:
        file_content = repo_obj.get_contents(file_path, ref=ref)
        return file_content.decoded_content.decode("utf-8").splitlines()
    except Exception as e:
        print(f"Error fetching file content: {file_path} on {ref}: {e}")
        return []


def get_full_line_number(file_lines: List[str], hunk: Hunk, diff_line_number: int) -> int:
    """
    Converts a diff-based line number to a full-file line number.
    """
    source_start = hunk.source_start
    source_lines = file_lines[:source_start + diff_line_number - 1]
    return len(source_lines)


def get_diff(owner: str, repo: str, pull_number: int) -> str:
    """Fetches the diff of the pull request from GitHub API."""
    repo_name = f"{owner}/{repo}"
    api_url = f"https://api.github.com/repos/{repo_name}/pulls/{pull_number}"
    headers = {
        'Authorization': f'Bearer {GITHUB_TOKEN}',
        'Accept': 'application/vnd.github.v3.diff'
    }

    response = requests.get(f"{api_url}.diff", headers=headers)

    if response.status_code == 200:
        return response.text
    else:
        print(f"Failed to get diff. Status code: {response.status_code}")
        return ""


def get_ai_response(prompt: str) -> List[Dict[str, str]]:
    """Sends the prompt to Gemini API and retrieves the response."""
    gemini_model = Client.GenerativeModel(os.environ.get('GEMINI_MODEL', 'gemini-2.0-flash-001'))

    try:
        response = gemini_model.generate_content(prompt)
        response_text = response.text.strip()

        if response_text.startswith('```json'):
            response_text = response_text[7:]
        if response_text.endswith('```'):
            response_text = response_text[:-3]
        
        return json.loads(response_text).get("reviews", [])

    except Exception as e:
        print(f"Error during Gemini API call: {e}")
        return []


def create_comment(file: PatchedFile, hunk: Hunk, ai_responses: List[Dict[str, str]], owner: str, repo: str) -> List[Dict[str, Any]]:
    """Creates comment objects from AI responses, including full file line numbers."""
    file_lines = get_file_content(owner, repo, file.path)

    comments = []
    for ai_response in ai_responses:
        try:
            line_number = int(ai_response["lineNumber"])
            full_line_number = get_full_line_number(file_lines, hunk, line_number)

            comment = {
                "body": ai_response["reviewComment"],
                "path": file.path,
                "position": line_number,
                "codeLine": full_line_number
            }
            comments.append(comment)

        except Exception as e:
            print(f"Error creating comment: {e}")
    return comments


def create_review_comment(owner: str, repo: str, pull_number: int, comments: List[Dict[str, Any]], batch_size: int = 10):
    """Submits review comments in batches."""
    repo_obj = gh.get_repo(f"{owner}/{repo}")
    pr = repo_obj.get_pull(pull_number)

    total_batches = (len(comments) + batch_size - 1) // batch_size
    for i in range(0, len(comments), batch_size):
        batch = comments[i:i+batch_size]

        body_lines = [f"### Gemini AI Code Reviewer Comments (Batch {i // batch_size + 1}/{total_batches})"]
        for comment in batch:
            body_lines.append(f"\n<details><summary>📂 **File:** `{comment['path']}`  |  **Line:** {comment['codeLine']}</summary>\n")
            body_lines.append(f"> {comment['body']}")
            body_lines.append("</details>\n")

        body_msg = "\n".join(body_lines)

        review = pr.create_review(
            body=body_msg,
            event="COMMENT"
        )
        print(f"Created review batch {i // batch_size + 1} with ID: {review.id}")


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
    pr_details = get_pr_details()
    diff = get_diff(pr_details.owner, pr_details.repo, pr_details.pull_number)

    if not diff:
        return

    parsed_diff = parse_diff(diff)
    comments = []

    for file in parsed_diff:
        for hunk in file['hunks']:
            ai_response = get_ai_response(create_prompt(file, hunk, pr_details))  # create_prompt는 별도로 추가
            comments.extend(create_comment(file, hunk, ai_response, pr_details.owner, pr_details.repo))

    if comments:
        create_review_comment(pr_details.owner, pr_details.repo, pr_details.pull_number, comments)


if __name__ == "__main__":
    main()


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