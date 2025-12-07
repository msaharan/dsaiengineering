# Copyright 2025 Mohit Saharan (github.com/msaharan)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

from typing import Iterable, Set


def _levenshtein(a: str, b: str) -> int:
    dp = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
    for i in range(len(a) + 1):
        dp[i][0] = i
    for j in range(len(b) + 1):
        dp[0][j] = j
    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
                dp[i - 1][j - 1] + cost,
            )
    return dp[-1][-1]


class SpellCorrector:
    """Lightweight edit-distance spell corrector with a guardrail to avoid over-correction."""

    def __init__(self, vocab: Iterable[str], max_edit_distance: int = 1):
        tokens: Set[str] = set()
        for entry in vocab:
            tokens.update(entry.lower().split())
        self.vocab = tokens
        self.max_edit_distance = max_edit_distance

    def correct(self, query: str) -> str:
        corrected_tokens = []
        for token in query.split():
            if token in self.vocab or not self.vocab:
                corrected_tokens.append(token)
                continue
            best_token = min(self.vocab, key=lambda v: _levenshtein(token, v))
            best_distance = _levenshtein(token, best_token)
            # Only correct when the best candidate is within the allowed distance.
            corrected_tokens.append(best_token if best_distance <= self.max_edit_distance else token)
        return " ".join(corrected_tokens)
