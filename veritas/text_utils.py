"""Shared text tokenisation helpers."""

from __future__ import annotations

import re

_TOKEN = re.compile(r"[a-z0-9]+(?:[.'-][a-z0-9]+)*|%|\$")
_STOP = frozenset("""
a an the of to and or for with without in on at by from as is are was were be been
being this that these those it its their our your his her they we you i not no if
then than so such can will would should may might must into over under between
""".split())

_NUM = re.compile(r"[-+]?\$?\d[\d,]*(?:\.\d+)?%?")


def tokenize(text: str, keep_stop: bool = False) -> list[str]:
    toks = _TOKEN.findall(text.lower())
    if keep_stop:
        return toks
    return [t for t in toks if t not in _STOP and len(t) > 1]


def sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9$])", text.strip())
    return [p.strip() for p in parts if p.strip()]


def find_numbers(text: str) -> list[str]:
    return _NUM.findall(text)
