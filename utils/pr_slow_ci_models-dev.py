import sys

from github import Github
import os

def get_pr(pr_number):

    g = Github(os.environ['GITHUB_TOKEN'])
    repo = g.get_repo("huggingface/transformers")
    pr = repo.get_pull(pr_number)

    print(pr)
    for file in pr.get_files():
        print(file)



if __name__ == '__main__':

    pr_number = "39100"
    pr_number = int(pr_number)

    get_pr(pr_number)