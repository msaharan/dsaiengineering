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

from src.ds_agent_tools import merge_datasets


def test_merge_datasets_success(sample_context):
    output = merge_datasets.invoke(
        {
            "left_dataset": "left_dataset",
            "right_dataset": "right_dataset",
            "left_on": "id",
            "right_on": "identifier",
            "how": "left",
            "limit": 3,
        }
    )

    assert "Merged shape" in output
    assert "Join type: left" in output
    assert "left_dataset" not in output  # ensure we don't leak reprs
    assert "id" in output
    assert "sales" in output


def test_merge_datasets_unknown_join(sample_context):
    output = merge_datasets.invoke(
        {
            "left_dataset": "left_dataset",
            "right_dataset": "right_dataset",
            "left_on": "id",
            "right_on": "identifier",
            "how": "invalid",
        }
    )

    assert "Unsupported join type" in output


def test_merge_datasets_missing_column(sample_context):
    output = merge_datasets.invoke(
        {
            "left_dataset": "left_dataset",
            "right_dataset": "right_dataset",
            "left_on": "does_not_exist",
            "right_on": "identifier",
        }
    )

    assert "Columns not found in left dataset" in output

