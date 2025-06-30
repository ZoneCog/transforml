import argparse
import json
import requests
import sys
import os
import re


def get_pr_files():

    with open("pr_files.txt") as fp:
        files = json.load(fp)
        files = [{k: v for k, v in item.items() if k in ["filename", "status"]} for item in files]

    # TODO: get directories under `(tests/)models/xxx`, `(tests/)models/quantization` and `(tests/)xxx`
    # GOAL: get new modeling files / get list of test files to suggest to run / match a list of specified items to run

    new_files = [item["filename"] for item in files if item["status"] == "added"]
    modified_files = [item["filename"] for item in files if item["status"] == "modified"]

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
                # TODO: what if not
                if item in content:
                    new_files_to_run.append(item)
                break

    modified_files_to_run = []
    for modified_file in modified_files:
        for regex in regexes:
            matched = regex.findall(modified_file)
            if len(matched) > 0:
                item = matched[0]
                item = item.replace("quantizers/quantizer_", "quantization/")
                # TODO: what if not
                if item in content:
                    modified_files_to_run.append(item)
                break

    new_files_to_run = sorted(set(new_files_to_run))
    modified_files_to_run = sorted(set(modified_files_to_run))

    return new_files_to_run, modified_files_to_run


def parse_message(message: str) -> str:
    """
    Parses a GitHub pull request's comment to find the models specified in it to run slow CI.

    Args:
        message (`str`): The body of a GitHub pull request's comment.

    Returns:
        `str`: The substring in `message` after `run-slow`, run_slow` or run slow`. If no such prefix is found, the
        empty string is returned.
    """
    if message is None:
        return ""

    message = message.strip().lower()

    # run-slow: model_1, model_2, quantization_1, quantization_2
    if not message.startswith(("run-slow", "run_slow", "run slow")):
        return ""
    message = message[len("run slow") :]
    # remove leading `:`
    while message.strip().startswith(":"):
        message = message.strip()[1:]

    return message


def get_models(message: str):
    models = parse_message(message)
    return models.replace(",", " ").split()


if __name__ == '__main__':

    # We can also use the following to fetch the information if we don't have them before calling this script.
    # url = f"https://api.github.com/repos/huggingface/transformers/contents/tests/models?ref={pr_sha}"

    # specific to this script and action
    content = []
    for filename in ["tests_dir.txt", "tests_models_dir.txt", "tests_quantization_dir.txt"]:
        with open(filename) as fp:
            data = json.load(fp)
            data = [item["path"][len("tests/"):] for item in data if item["type"] == "dir"]
            content.extend(data)

    parser = argparse.ArgumentParser()
    parser.add_argument("--message", type=str, default="", help="The content of a comment.")
    args = parser.parse_args()

    # Computed from the changed files.
    # These are already with the prefix `models/` or `quantization/`, so we don't need to add them.
    new_files_to_run, modified_files_to_run = get_pr_files()

    print(new_files_to_run)
    print(modified_files_to_run)

    # These don't have the prefix `models/` or `quantization/`, so we need to add them.
    # At this moment, we don't know if they are in tests/models or in tests/quantization, or if they even exist
    specified_models = []
    if args.message:
        specified_models = get_models(args.message)

    # add prefix and check the path exists
