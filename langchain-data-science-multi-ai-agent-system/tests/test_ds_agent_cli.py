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

from types import SimpleNamespace

import pytest

from src.ds_agent import build_system_prompt, parse_cli_args


class _StubCatalog:
    def entries(self):
        yield SimpleNamespace(name="alpha")
        yield SimpleNamespace(name="beta")


def test_build_system_prompt_includes_mcp_summary():
    prompt = build_system_prompt(_StubCatalog(), mcp_summary="remote_fetch, s3_export")
    assert "remote_fetch" in prompt
    assert "s3_export" in prompt


def test_parse_cli_args_accepts_mcp_config():
    args = parse_cli_args(
        [
            "--mcp-config",
            '{"servers": {"demo": {"transport": "stdio", "command": "echo", "args": ["hi"]}}}',
        ]
    )
    assert args.mcp_config.startswith("{")


def test_parse_cli_args_conflicting_stream_flags():
    with pytest.raises(SystemExit):
        parse_cli_args(["--event-stream", "--no-stream"])

