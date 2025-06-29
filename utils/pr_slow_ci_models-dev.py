import sys
import os
import re

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

    # models or quantizers
    file_re_1 = re.compile(r"src/transformers/(models/.*)/modeling_.*\.py")

    # Unfortunately, there is no proper way to map this to quantization tests.
    file_re_2 = re.compile(r"src/transformers/(quantizers/quantizer_.*)\.py")

    # tests for models or quantizers
    file_re_3 = re.compile(r"tests/(models/.*)/test_.*\.py")
    file_re_4 = re.compile(r"tests/(quantization/.*)/test_.*\.py")

    # directories of models or quantizers
    file_re_5 = re.compile(r"src/transformers/(models/.*)/.*\.py")


    regexes = [file_re_1, file_re_2, file_re_3, file_re_4, file_re_5]

    new_files_to_run = []
    for new_file in new_files:
        for regex in regexes:
            matched = regex.findall(new_file)
            if len(matched) > 0:
                item = matched[0]
                item = item.replace("quantizers/quantizer_", "quantization/")
                new_files_to_run.append(item)
                break

    modified_files_to_run = []
    for modified_file in modified_files:
        for regex in regexes:
            matched = regex.findall(modified_file)
            if len(matched) > 0:
                item = matched[0]
                item = item.replace("quantizers/quantizer_", "quantization/")
                modified_files_to_run.append(item)
                break

    new_files_to_run = sorted(set(new_files_to_run))
    modified_files_to_run = sorted(set(modified_files_to_run))

    print(new_files_to_run)
    print(modified_files_to_run)


if __name__ == '__main__':

    # pr_number = "39100"
    # pr_number = int(pr_number)
    # get_pr2(pr_number)

    get_pr_files()