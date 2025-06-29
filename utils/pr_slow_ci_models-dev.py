import sys

from github import Github
import os

def get_pr(pr_number):

    g = Github()
    repo = g.get_repo("huggingface/transformers")
    pr = repo.get_pull(pr_number)

    print(pr)
    for file in pr.get_files():
        print(file)
        print(file.filename)
        print(file.status)


def get_pr2():

    import json
    fp = open("pr_files.txt")
    files = json.load(fp); fp.close()
    files = [{k: v for k, v in item.items() if k in ["filename", "status"]} for item in files];print(json.dumps(files, indent=4))
    print(files)


if __name__ == '__main__':

    pr_number = "39100"
    pr_number = int(pr_number)

    get_pr(pr_number)
    get_pr2()