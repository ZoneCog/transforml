import sys
import os

# def get_pr(pr_number):
#     from github import Github
#
#     g = Github()
#     repo = g.get_repo("huggingface/transformers")
#     pr = repo.get_pull(pr_number)
#
#     print(pr)
#     for file in pr.get_files():
#         print(file)
#         print(file.filename)
#         print(file.status)


def get_pr_files():

    import json
    fp = open("pr_files.txt")
    files = json.load(fp)
    fp.close()
    files = [{k: v for k, v in item.items() if k in ["filename", "status"]} for item in files]
    print(files)

    # TODO: get directories under `(tests/)models/xxx`, `(tests/)models/quantization` and `(tests/)xxx`
    # GOAL: get new modeling files / get list of test files to suggest to run / match a list of specified items to run

    new_files = [item["filename"] for item in files if item["status"] == "added"]
    modified_files = [item["filename"] for item in files if item["status"] == "modified"]

    print(new_files)
    print(modified_files)

if __name__ == '__main__':

    # pr_number = "39100"
    # pr_number = int(pr_number)
    # get_pr2(pr_number)

    get_pr_files()