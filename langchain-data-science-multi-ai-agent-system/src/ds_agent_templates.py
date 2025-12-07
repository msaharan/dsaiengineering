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

"""Task templates guiding multi-step reasoning for the data science agent."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Mapping


@dataclass(slots=True)
class TaskStep:
    """A single step within a task template."""

    name: str
    description: str
    rationale: str | None = None
    suggested_tool: str | None = None
    expected_output: str | None = None

    def to_dict(self) -> dict[str, str]:
        payload: dict[str, str] = {
            "name": self.name,
            "description": self.description,
        }
        if self.rationale:
            payload["rationale"] = self.rationale
        if self.suggested_tool:
            payload["suggested_tool"] = self.suggested_tool
        if self.expected_output:
            payload["expected_output"] = self.expected_output
        return payload


@dataclass(slots=True)
class TaskTemplate:
    """Reusable template describing how to accomplish a task type."""

    task_type: str
    title: str
    description: str
    steps: list[TaskStep] = field(default_factory=list)
    tips: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return {
            "task_type": self.task_type,
            "title": self.title,
            "description": self.description,
            "steps": [step.to_dict() for step in self.steps],
            "tips": list(self.tips),
        }


def _normalise_task_type(task_type: str) -> str:
    return task_type.strip().lower().replace(" ", "_")


def _build_registry() -> dict[str, TaskTemplate]:
    return {
        template.task_type: template
        for template in [
            TaskTemplate(
                task_type="eda",
                title="Exploratory Data Analysis",
                description=(
                    "Initial pass over a dataset to understand structure, summary statistics,"
                    " data quality issues, and noteworthy patterns."
                ),
                steps=[
                    TaskStep(
                        name="catalog_overview",
                        description="Confirm dataset metadata (shape, column types, description).",
                        suggested_tool="list_datasets",
                        expected_output="Short bullet list describing dataset(s) selected.",
                    ),
                    TaskStep(
                        name="sample_preview",
                        description="Inspect a representative sample of rows to gauge data contents.",
                        suggested_tool="preview_dataset",
                        expected_output="Table snippet with <= 20 rows and pertinent observations.",
                    ),
                    TaskStep(
                        name="profiling",
                        description="Run profiling to capture nulls, uniques, and summary stats per column.",
                        suggested_tool="profile_dataset_tool",
                        expected_output="JSON summary highlighting key metrics per column.",
                    ),
                    TaskStep(
                        name="insight_synthesis",
                        description="Summarise notable data quality issues and candidate next questions.",
                        rationale="Helps transition from low-level facts to actionable findings.",
                        expected_output="Bullet list of observations and recommended follow-ups.",
                    ),
                ],
                tips=[
                    "Cap row outputs to keep responses concise.",
                    "Call profiling before drawing conclusions about distributions.",
                ],
            ),
            TaskTemplate(
                task_type="trend_detection",
                title="Trend Detection",
                description=(
                    "Assess temporal or sequential trends across numeric metrics, highlighting"
                    " increases, decreases, and anomalies."
                ),
                steps=[
                    TaskStep(
                        name="identify_time_axis",
                        description="Confirm presence of a datetime or sequence column suitable for trend analysis.",
                        suggested_tool="profile_dataset_tool",
                        expected_output="Statement identifying the time key and frequency (if any).",
                    ),
                    TaskStep(
                        name="aggregate_metrics",
                        description="Aggregate key metrics by the temporal granularity to smooth noise.",
                        suggested_tool="analyze_dataset",
                        expected_output="Table of aggregated metrics with commentary on observed direction.",
                    ),
                    TaskStep(
                        name="detect_change_points",
                        description="Highlight notable shifts, spikes, or declines with supporting evidence.",
                        expected_output="Bullet list citing time ranges and magnitude of changes.",
                    ),
                    TaskStep(
                        name="summarise_implications",
                        description="Explain potential causes or next steps based on detected trends.",
                        expected_output="Short narrative linking trends to business or analytical questions.",
                    ),
                ],
                tips=[
                    "Ensure data is sorted by the time axis before aggregation.",
                    "Call out periods with missing data or irregular frequency explicitly.",
                ],
            ),
            TaskTemplate(
                task_type="forecasting",
                title="Baseline Forecasting",
                description=(
                    "Create a lightweight forecast for a numeric metric, including assumptions and"
                    " validation of input data quality."
                ),
                steps=[
                    TaskStep(
                        name="diagnose_series",
                        description="Check data sufficiency, missing values, and stationarity assumptions.",
                        suggested_tool="profile_dataset_tool",
                        expected_output="Summary of data readiness and any preprocessing requirements.",
                    ),
                    TaskStep(
                        name="prepare_features",
                        description="Engineer simple features or transformations (e.g., rolling means).",
                        suggested_tool="analyze_dataset",
                        expected_output="Description of features engineered and rationale.",
                    ),
                    TaskStep(
                        name="fit_baseline_model",
                        description="Apply a baseline forecasting approach (naive, moving average, or simple regression).",
                        expected_output="Forecast table or key figures for the desired horizon.",
                    ),
                    TaskStep(
                        name="communicate_uncertainty",
                        description="Discuss confidence, limitations, and suggested improvements.",
                        expected_output="Narrative covering forecast limitations and next analyses.",
                    ),
                ],
                tips=[
                    "Document assumptions clearly, especially when using simplified techniques.",
                    "Recommend more advanced modeling if variance is high or residuals are patterned.",
                ],
            ),
        ]
    }


_TEMPLATES: dict[str, TaskTemplate] = _build_registry()


def list_task_templates() -> list[TaskTemplate]:
    """Return all registered task templates."""

    return list(_TEMPLATES.values())


def get_task_template(task_type: str) -> TaskTemplate:
    """Fetch a task template by type, normalising common aliases."""

    key = _normalise_task_type(task_type)
    if key not in _TEMPLATES:
        raise KeyError(f"Unknown task template '{task_type}'. Available: {', '.join(sorted(_TEMPLATES))}")
    return _TEMPLATES[key]


def available_task_types() -> list[str]:
    """Return the canonical task type identifiers."""

    return sorted(_TEMPLATES)


def template_catalog() -> list[Mapping[str, object]]:
    """Return all templates as serialisable dictionaries."""

    return [template.to_dict() for template in list_task_templates()]


__all__ = [
    "TaskStep",
    "TaskTemplate",
    "available_task_types",
    "get_task_template",
    "list_task_templates",
    "template_catalog",
]


