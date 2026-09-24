# Copyright (C) 2020-2026 Shanghai Biren Technology Co., Ltd.

# -*- coding: utf-8 -*-

import os
import logging
import sys
import gc
from typing import Any, Tuple

import fastcore.basics


"""
This module wraps `fastcore.basics.patch_to` to automatically log debugging information
whenever a patch is applied. The log includes:
- The current process's rank and local_rank,
- The class and method name being patched,
- The relative file path where the method is defined.

"""


def get_rank_info() -> Tuple[int, int]:
    """Get rank and local rank from environment variables."""
    try:
        rank = int(os.environ.get("RANK", "0"))
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    except (ValueError, TypeError):
        rank = 0
        local_rank = 0
    return rank, local_rank


# Configure logger
patch_logger = logging.getLogger("patch_logger")

# Avoid adding multiple handlers in case module is reloaded
if not patch_logger.hasHandlers():
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(pathname)s:%(lineno)d %(message)s")
    handler.setFormatter(formatter)
    patch_logger.addHandler(handler)
    level = os.environ.get("VLLM_LOGGING_LEVEL", "INFO").upper()
    patch_logger.setLevel(level)
    patch_logger.propagate = False


def __log_patches(cls, f: Any, _nm: str | None = None, title: str = "patch") -> bool:
    """save log for recording patches.

    Args:
        cls (object): Target class
        f (callable): target function
        title (str, optional): "patch" or "wrap", Defaults to "patch".
    Returns:
        bool: True: already patched, False: not yet.
    """
    if hasattr(f, "_patch_logged"):
        return True

    rank, local_rank = get_rank_info()
    callable_name = getattr(f, "__name__", _nm)
    name = [f"{c_.__name__}.{callable_name}" if c_ else callable_name for c_ in fastcore.basics.tuplify(cls)]
    patch_logger.debug(
        "[rank:%s/local_rank:%s] [%s] %s",
        rank,
        local_rank,
        title,
        name,
        stacklevel=3,
    )
    f._patch_logged = True
    return False


def __update_refence_list(cls: Any, name) -> list:
    """update cls object, ensure it contains all referrers. so no need to use '_propagate_patch'
    Returns:
        list: final list contains all referers.
    """
    c = fastcore.basics.tuplify(cls)

    target = getattr(c[0], name, None)
    if target:
        # following subtraction becaude:  function getrefcount() takes 1 ref, and variable 'target' tasks 1 ref.
        # rcount should be the real count for 'target'.
        rcount = sys.getrefcount(target) - 2
        if rcount > len(c):
            final_cls = []
            for ref in gc.get_referrers(target):
                if isinstance(ref, dict) and "__name__" in ref:
                    final_cls.append(sys.modules[ref["__name__"]])
            patch_logger.debug(
                "[patch] expand 'cls' to %s",
                [m.__name__ for m in final_cls],
                stacklevel=3,
            )
            return final_cls
    return c


def patch_to_with_log(cls, **kw):
    """Patch function with one-time logging of patched method name, rank, and file path.\n
    cls could be (tuple of) module, class
    kw (and default) could be: as_prop=False, cls_method=False, set_prop=False, static_method=False, nm=None, glb=None
    Note: a ptr to original function named `_orig_<func_name>` is added into cls object, if it exists.
    so that it can be called in new function, like.
    ```python
       class Base:
           def div(self, a, b):
               # calcuation..
               pass

       @wrap_to_with_log(Base)
       def div(self, a, b):
           # pre processes:
           if b == 0:
               raise Exception("can't devided by zero")

           ret = self._orig_div(a, b)  # call original function.

           # post processes
           ret = ret + 1
           return ret
    ```
    """
    if isinstance(cls, (list, tuple)):
        cls = [c for c in cls if c]
    if not cls:
        return lambda X: None

    def deco(f):
        _nm = kw.get("nm") or getattr(f, "__name__", None)
        assert _nm, "must give 'nm' becuase there is no'__name__'"
        if __log_patches(cls, f, _nm, "patch"):
            return
        updated_cls = __update_refence_list(cls, _nm)

        if not hasattr(f, "__code__"):
            # special case for callable class. just replace it with new instance obj.
            for c in updated_cls:
                if (oo := getattr(c, _nm)) is not None:
                    setattr(c, "_orig_" + _nm, oo)
                setattr(c, _nm, f)
            return f

        fastcore.basics.patch_to(updated_cls, **kw)(f)
        return f

    return deco
