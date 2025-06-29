from github import Github
import os

def get_pr(pr_number):

    g = Github(os.environ['GITHUB_TOKEN'])
    repo = g.get_repo("huggingface/transformers")
    pr = repo.get_pull(pr_number)

    print(pr)
    for file in pr.get_files():
        print(file)