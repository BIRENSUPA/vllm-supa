# -*- coding: utf-8 -*-
# Copyright©2020-2026 Shanghai Biren Technology Co., Ltd. All rights reserved.

import sys

import pytest

class TestCollection:
    def __init__(self):
        self.config = ""
        self.collected = []
        self.improperly_marked_cases = set()

    def pytest_collection_modifyitems(self, config, items):
        self.config = config.option.markexpr
        valid_items = []
        for item in items:
            casename = [item.path]
            if casename not in self.collected:
                self.collected.append(casename)
            # so far, no need process product
            valid_items.append(item)

        items.clear()
        items.extend(valid_items)


class MyPlugin(TestCollection):
    def pytest_sessionfinish(self):
        print("\n Finish pytest testing ")


if __name__ == "__main__":
    pytest_args = sys.argv[1:]
    ret = pytest.main(pytest_args, plugins=[MyPlugin()])
    print(" Exit vllm-supa-200 pytest with code: {} ".format(ret))
    sys.exit(ret)
