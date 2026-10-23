#!/usr/bin/env python3
"""Convert cumulative GUI trajectories to ms-swift Gemma 4 Agent JSONL.

The converter intentionally keeps the source train/validation split and the
source rolling three-image window.  It emits two views of every source row:

* SFT: the final assistant reasoning and tool call are supervised.
* GKD: the final gold completion is removed and saved as reference_action.

Assistant reasoning is normalized to Gemma 4's native thought-channel wrapper;
ms-swift renders the complete Agent conversation at training time with
``--template gemma4 --agent_template gemma4``.
"""

from __future__ import annotations

import argparse
import calendar
import copy
import datetime as dt
import hashlib
import json
import os
import re
import shutil
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


DEFAULT_DATASET_DIR = Path("/data/lcx/foxx/train/dataset/0725")
DEFAULT_SYSTEM_PROMPT = Path("/data/lcx/foxx/train/dataset/system_prompt.md")
DEFAULT_OUTPUT_DIR = Path("/data/lcx/foxx/train/dataset/opd_data")
DEFAULT_IMAGE_ROOT = Path("/data/lcx/foxx/train/dataset")

THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL)
ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
REMINDER_RE = re.compile(r"<system-reminder>(.*?)</system-reminder>", re.DOTALL)
DATE_RE = re.compile(r"current\s+date\s*:\s*(\d{4})[.\-/](\d{1,2})[.\-/](\d{1,2})", re.I)
APP_RE = re.compile(r"current\s+app\s*:\s*([^;<\n]+)", re.I)
DO_RE = re.compile(
    r"^\s*do\s*\(\s*name\s*=\s*(['\"])(?P<name>.*?)\1\s*,\s*"
    r"arguments\s*=\s*(?P<arguments>\{.*\})\s*\)\s*$",
    re.DOTALL,
)


TOOL_REGISTRY: dict[str, dict[str, Any]] = {
    "Launch": {
        "description": "启动指定应用。",
        "parameters": {
            "type": "object",
            "properties": {"app_name": {"type": "string", "description": "应用名称"}},
            "required": ["app_name"],
            "additionalProperties": False,
        },
    },
    "Tap": {
        "description": "点击屏幕指定坐标。",
        "parameters": {
            "type": "object",
            "properties": {
                "coordinate": {
                    "type": "array",
                    "items": {"type": "integer"},
                    "minItems": 2,
                    "maxItems": 2,
                    "description": "[x, y] 坐标",
                }
            },
            "required": ["coordinate"],
            "additionalProperties": False,
        },
    },
    "Wait": {
        "description": "等待页面加载。",
        "parameters": {
            "type": "object",
            "properties": {
                "duration": {
                    "type": "number",
                    "minimum": 0,
                    "description": "等待秒数",
                }
            },
            "required": ["duration"],
            "additionalProperties": False,
        },
    },
    "Back": {
        "description": "返回上一页。",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
            "additionalProperties": False,
        },
    },
    "Swipe": {
        "description": "从起点滑动到终点。",
        "parameters": {
            "type": "object",
            "properties": {
                "start_coordinate": {
                    "type": "array",
                    "items": {"type": "integer"},
                    "minItems": 2,
                    "maxItems": 2,
                    "description": "起点 [x, y]",
                },
                "end_coordinate": {
                    "type": "array",
                    "items": {"type": "integer"},
                    "minItems": 2,
                    "maxItems": 2,
                    "description": "终点 [x, y]",
                },
            },
            "required": ["start_coordinate", "end_coordinate"],
            "additionalProperties": False,
        },
    },
    "Type": {
        "description": "在当前输入框输入文本（会清除原有内容）。",
        "parameters": {
            "type": "object",
            "properties": {"text": {"type": "string", "description": "要输入的文本"}},
            "required": ["text"],
            "additionalProperties": False,
        },
    },
    "finish": {
        "description": "任务完成后结束操作，并向用户返回最终结果。",
        "parameters": {
            "type": "object",
            "properties": {
                "response": {
                    "type": "string",
                    "description": "向用户返回的完整最终答案或简洁任务完成说明。",
                }
            },
            "required": ["response"],
            "additionalProperties": False,
        },
    },
}

APP_NORMALIZATION = {
    "星巴克": "Starbucks",
    "未识别": "Unknown",
}


class ConversionError(RuntimeError):
    pass


def compact_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ConversionError(f"{path}:{line_no}: invalid JSON: {exc}") from exc
            if not isinstance(row, dict):
                raise ConversionError(f"{path}:{line_no}: top-level JSON must be an object")
            row["__source_line__"] = line_no
            rows.append(row)
    return rows


def reminder_state(content: str, *, row_id: str) -> tuple[dt.date | None, str | None]:
    reminders = REMINDER_RE.findall(content)
    if not reminders:
        return None, None
    reminder = reminders[-1]
    date_match = DATE_RE.search(reminder)
    app_match = APP_RE.search(reminder)
    parsed_date = None
    if date_match:
        try:
            parsed_date = dt.date(*(int(part) for part in date_match.groups()))
        except ValueError as exc:
            raise ConversionError(f"{row_id}: invalid reminder date: {date_match.group(0)!r}") from exc
    app = app_match.group(1).strip() if app_match else None
    if app is not None:
        app = APP_NORMALIZATION.get(app, app)
    return parsed_date, app


def remove_reminders(content: str) -> str:
    return REMINDER_RE.sub("", content).strip()


def render_system_prompt(template: str, current_date: dt.date, *, row_id: str) -> str:
    monday = current_date - dt.timedelta(days=current_date.weekday())
    replacements = {
        "{当前日期}": current_date.isoformat(),
        "{本周周一}": monday.isoformat(),
        "{本月天数}": str(calendar.monthrange(current_date.year, current_date.month)[1]),
    }
    rendered = template
    on_count = rendered.count("{ON}")
    for placeholder, value in replacements.items():
        rendered = rendered.replace(placeholder, value)
    unresolved = [placeholder for placeholder in replacements if placeholder in rendered]
    if unresolved:
        raise ConversionError(f"{row_id}: unresolved system prompt placeholders: {unresolved}")
    if rendered.count("{ON}") != on_count:
        raise ConversionError(f"{row_id}: literal {{ON}} was modified")
    return rendered.strip()


def parse_action(answer: str, *, row_id: str, message_index: int) -> dict[str, Any]:
    match = DO_RE.match(answer)
    if not match:
        raise ConversionError(
            f"{row_id}: message {message_index}: unsupported action expression: {answer[:240]!r}"
        )
    name = match.group("name")
    try:
        arguments = json.loads(match.group("arguments"))
    except json.JSONDecodeError as exc:
        raise ConversionError(
            f"{row_id}: message {message_index}: action arguments are not strict JSON: {exc}"
        ) from exc
    if not isinstance(arguments, dict):
        raise ConversionError(f"{row_id}: message {message_index}: arguments must be an object")
    return {"name": name, "arguments": arguments}


def validate_action(action: dict[str, Any], *, row_id: str, message_index: int) -> None:
    name = action.get("name")
    args = action.get("arguments")
    if name not in TOOL_REGISTRY:
        raise ConversionError(f"{row_id}: message {message_index}: unknown tool {name!r}")
    if not isinstance(args, dict):
        raise ConversionError(f"{row_id}: message {message_index}: arguments must be an object")

    def exact_keys(required: set[str]) -> None:
        if set(args) != required:
            raise ConversionError(
                f"{row_id}: message {message_index}: {name} argument keys {sorted(args)} != {sorted(required)}"
            )

    def coordinate(key: str) -> None:
        value = args[key]
        if (
            not isinstance(value, list)
            or len(value) != 2
            or any(not isinstance(item, int) or isinstance(item, bool) for item in value)
        ):
            raise ConversionError(
                f"{row_id}: message {message_index}: {name}.{key} must be exactly two integers"
            )

    if name == "Launch":
        exact_keys({"app_name"})
        if not isinstance(args["app_name"], str):
            raise ConversionError(f"{row_id}: message {message_index}: Launch.app_name must be a string")
    elif name == "Tap":
        exact_keys({"coordinate"})
        coordinate("coordinate")
    elif name == "Swipe":
        exact_keys({"start_coordinate", "end_coordinate"})
        coordinate("start_coordinate")
        coordinate("end_coordinate")
    elif name == "Type":
        exact_keys({"text"})
        if not isinstance(args["text"], str):
            raise ConversionError(f"{row_id}: message {message_index}: Type.text must be a string")
    elif name == "Wait":
        exact_keys({"duration"})
        duration = args["duration"]
        if isinstance(duration, bool) or not isinstance(duration, (int, float)) or duration < 0:
            raise ConversionError(f"{row_id}: message {message_index}: Wait.duration must be non-negative")
    elif name == "finish":
        exact_keys({"response"})
        if not isinstance(args["response"], str) or not args["response"].strip():
            raise ConversionError(
                f"{row_id}: message {message_index}: finish.response must be a non-empty string"
            )
    else:
        exact_keys(set())


def split_assistant(
    content: str,
    *,
    row_id: str,
    message_index: int,
    expected_final_action: str | None,
    anomalies: list[dict[str, Any]],
) -> tuple[str, dict[str, Any]]:
    think_match = THINK_RE.search(content)
    answer_match = ANSWER_RE.search(content)

    if think_match:
        reasoning = think_match.group(1).strip()
    else:
        prefix = content[: answer_match.start()] if answer_match else content
        reasoning = prefix.strip()
        anomalies.append(
            {
                "type": "missing_think_wrapper",
                "id": row_id,
                "message_index": message_index,
                "content_sha256": hashlib.sha256(content.encode("utf-8")).hexdigest(),
                "resolution": "wrapped_pre_answer_text",
            }
        )
    if not reasoning:
        raise ConversionError(f"{row_id}: message {message_index}: empty reasoning")

    if think_match:
        visible_response = content[think_match.end():]
        if answer_match:
            visible_response = (
                visible_response[: max(0, answer_match.start() - think_match.end())]
                + visible_response[max(0, answer_match.end() - think_match.end()):]
            )
        visible_response = visible_response.strip()
    else:
        visible_response = ""

    if answer_match:
        action = parse_action(answer_match.group(1), row_id=row_id, message_index=message_index)
        if action["name"] == "finish" and not action["arguments"]:
            action["arguments"] = {"response": visible_response}
    elif expected_final_action == "finish":
        action = {"name": "finish", "arguments": {"response": visible_response}}
    else:
        raise ConversionError(f"{row_id}: message {message_index}: assistant message has no action")

    validate_action(action, row_id=row_id, message_index=message_index)
    if expected_final_action is not None and action["name"] != expected_final_action:
        raise ConversionError(
            f"{row_id}: final action {action['name']!r} != action_name {expected_final_action!r}"
        )
    return f"<|channel>thought\n{reasoning}\n<channel|>", action


def collect_final_date(messages: list[dict[str, Any]], *, row_id: str) -> dt.date:
    dates: list[dt.date] = []
    for message in messages:
        if message.get("role") == "user":
            current_date, _ = reminder_state(str(message.get("content", "")), row_id=row_id)
            if current_date is not None:
                dates.append(current_date)
    if not dates:
        raise ConversionError(f"{row_id}: no current date found in system reminders")
    if len(set(dates)) != 1:
        raise ConversionError(f"{row_id}: inconsistent reminder dates: {[d.isoformat() for d in dates]}")
    return dates[-1]


def validate_source_images(row: dict[str, Any], *, image_root: Path) -> None:
    row_id = str(row.get("id"))
    step_id = row.get("step_id")
    images = row.get("images")
    if not isinstance(step_id, int) or step_id < 1:
        raise ConversionError(f"{row_id}: invalid step_id {step_id!r}")
    if not isinstance(images, list) or any(not isinstance(path, str) for path in images):
        raise ConversionError(f"{row_id}: images must be a list of strings")
    expected_count = min(step_id, 3)
    if len(images) != expected_count:
        raise ConversionError(
            f"{row_id}: rolling image window has {len(images)} images, expected {expected_count}"
        )
    expected_steps = list(range(max(1, step_id - 2), step_id + 1))
    actual_steps: list[int] = []
    for relative_path in images:
        resolved = image_root / relative_path
        if not resolved.is_file():
            raise ConversionError(f"{row_id}: missing image: {resolved}")
        stem = Path(relative_path).stem
        if not stem.isdigit():
            raise ConversionError(f"{row_id}: image filename does not encode a numeric step: {relative_path}")
        actual_steps.append(int(stem))
    if actual_steps != expected_steps:
        raise ConversionError(
            f"{row_id}: image steps {actual_steps} != rolling window {expected_steps}"
        )
    marker_count = sum(str(message.get("content", "")).count("<image>") for message in row["messages"])
    if marker_count != len(images):
        raise ConversionError(
            f"{row_id}: source has {marker_count} <image> markers but {len(images)} images"
        )


def build_tools(observed_names: Iterable[str]) -> list[dict[str, Any]]:
    order = ["Launch", "Tap", "Wait", "Back", "Swipe", "Type", "finish"]
    names = set(observed_names)
    unknown = names - TOOL_REGISTRY.keys()
    if unknown:
        raise ConversionError(f"no schema registered for observed tools: {sorted(unknown)}")
    return [
        {
            "type": "function",
            "function": {
                "name": name,
                "description": TOOL_REGISTRY[name]["description"],
                "parameters": TOOL_REGISTRY[name]["parameters"],
            },
        }
        for name in order
        if name in names
    ]


def convert_row(
    source_row: dict[str, Any],
    *,
    system_template: str,
    tools_json: str,
    image_root: Path,
    anomalies: list[dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    row = {key: copy.deepcopy(value) for key, value in source_row.items() if key != "__source_line__"}
    row_id = str(row.get("id"))
    source_messages = row.get("messages")
    if not isinstance(source_messages, list) or len(source_messages) < 3:
        raise ConversionError(f"{row_id}: messages must contain at least system, user and assistant")
    if source_messages[0].get("role") != "system" or source_messages[1].get("role") != "user":
        raise ConversionError(f"{row_id}: expected system then user at the start")
    if source_messages[-1].get("role") != "assistant":
        raise ConversionError(f"{row_id}: source row must end with assistant")

    validate_source_images(row, image_root=image_root)
    current_date = collect_final_date(source_messages, row_id=row_id)
    converted: list[dict[str, Any]] = [
        {
            "role": "system",
            "content": render_system_prompt(system_template, current_date, row_id=row_id),
            "loss": False,
        }
    ]

    first_user = remove_reminders(str(source_messages[1].get("content", "")))
    if not first_user:
        raise ConversionError(f"{row_id}: first user task is empty after removing reminder")
    converted.append({"role": "user", "content": first_user, "loss": False})

    previous_action: dict[str, Any] | None = None
    final_action: dict[str, Any] | None = None
    for message_index, message in enumerate(source_messages[2:], 2):
        role = message.get("role")
        content = str(message.get("content", ""))
        is_final = message_index == len(source_messages) - 1
        if role == "assistant":
            if previous_action is not None:
                raise ConversionError(f"{row_id}: message {message_index}: consecutive assistant messages")
            expected = str(row.get("action_name")) if is_final else None
            reasoning, action = split_assistant(
                content,
                row_id=row_id,
                message_index=message_index,
                expected_final_action=expected,
                anomalies=anomalies,
            )
            loss = bool(message.get("loss", False))
            converted.append({"role": "assistant", "content": reasoning, "loss": loss})
            converted.append(
                {"role": "tool_call", "content": compact_json(action), "loss": loss}
            )
            previous_action = action
            if is_final:
                final_action = action
        elif role == "user":
            if previous_action is None:
                raise ConversionError(f"{row_id}: message {message_index}: feedback has no preceding action")
            residue = remove_reminders(content).replace("<image>", "").strip()
            if residue:
                raise ConversionError(
                    f"{row_id}: message {message_index}: unexpected feedback text: {residue[:200]!r}"
                )
            feedback_date, app = reminder_state(content, row_id=row_id)
            if feedback_date is None or app is None:
                raise ConversionError(f"{row_id}: message {message_index}: incomplete environment reminder")
            observation = "<image>" if "<image>" in content else ""
            response = {
                "status": "success",
                "current_app": app,
                "current_date": feedback_date.isoformat(),
                "observation": observation,
            }
            converted.append(
                {
                    "role": "tool_response",
                    "name": previous_action["name"],
                    "content": compact_json(response),
                    "loss": False,
                }
            )
            previous_action = None
        else:
            raise ConversionError(f"{row_id}: message {message_index}: unsupported role {role!r}")

    if final_action is None or previous_action != final_action:
        raise ConversionError(f"{row_id}: did not finish on an unconsumed final tool call")
    if [message["role"] for message in converted[-2:]] != ["assistant", "tool_call"]:
        raise ConversionError(f"{row_id}: converted SFT row does not end with assistant/tool_call")
    if not all(message["loss"] is True for message in converted[-2:]):
        raise ConversionError(f"{row_id}: final SFT assistant/tool_call must both have loss=true")

    output_marker_count = sum(str(message.get("content", "")).count("<image>") for message in converted)
    if output_marker_count != len(row["images"]):
        raise ConversionError(
            f"{row_id}: converted messages have {output_marker_count} markers but {len(row['images'])} images"
        )

    shared = copy.deepcopy(row)
    shared["tools"] = tools_json
    shared["tool_choice"] = "auto"

    sft = copy.deepcopy(shared)
    sft["messages"] = converted
    sft["format"] = "ms_swift_gemma4_agent_sft_v2"

    gkd = copy.deepcopy(shared)
    gkd["messages"] = copy.deepcopy(converted[:-2])
    gkd["reference_action"] = copy.deepcopy(final_action)
    gkd["format"] = "ms_swift_gemma4_agent_gkd_v2"
    if not gkd["messages"] or gkd["messages"][-1]["role"] not in {"user", "tool_response"}:
        raise ConversionError(f"{row_id}: GKD prompt must end with user or tool_response")

    return sft, gkd, final_action


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(compact_json(row))
            handle.write("\n")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_output_pair(
    source_rows: list[dict[str, Any]],
    sft_rows: list[dict[str, Any]],
    gkd_rows: list[dict[str, Any]],
    *,
    tools_json: str,
) -> None:
    if not (len(source_rows) == len(sft_rows) == len(gkd_rows)):
        raise ConversionError("source/SFT/GKD line counts differ")
    for source, sft, gkd in zip(source_rows, sft_rows, gkd_rows):
        row_id = str(source.get("id"))
        if sft.get("id") != row_id or gkd.get("id") != row_id:
            raise ConversionError(f"{row_id}: output order/id mismatch")
        if sft.get("images") != source.get("images") or gkd.get("images") != source.get("images"):
            raise ConversionError(f"{row_id}: source image list was changed")
        if sft.get("tools") != tools_json or gkd.get("tools") != tools_json:
            raise ConversionError(f"{row_id}: tools definition differs")
        if sft["messages"][:-2] != gkd["messages"]:
            raise ConversionError(f"{row_id}: GKD prompt differs from SFT prefix")
        final = json.loads(sft["messages"][-1]["content"])
        if final != gkd.get("reference_action"):
            raise ConversionError(f"{row_id}: reference_action differs from SFT gold tool call")
        if final.get("name") != source.get("action_name"):
            raise ConversionError(f"{row_id}: reference_action differs from source action_name")


def report_markdown(report: dict[str, Any]) -> str:
    split_rows = report["splits"]
    actions = report["actions"]["counts"]
    anomalies = report["anomalies"]
    unique_anomalies = report["anomaly_summary"]["unique_content_count"]
    lines = [
        "# Gemma 4 OPD 数据转换报告",
        "",
        f"生成时间：`{report['generated_at_utc']}`",
        "",
        "## 结果",
        "",
        "| split | 源数据/SFT/GKD 行数 | episode 数 | 图片窗口分布 |",
        "|---|---:|---:|---|",
    ]
    for split in ("train", "val"):
        item = split_rows[split]
        distribution = ", ".join(f"{key}张={value}" for key, value in item["image_window_counts"].items())
        lines.append(
            f"| {split} | {item['rows']} | {item['episodes']} | {distribution} |"
        )
    lines.extend(
        [
            "",
            f"train/val episode 交集：`{report['episode_split_check']['intersection_count']}`。",
            "",
            "三图规则已作为硬校验：step N 仅引用 `max(1, N-2) ... N` 的截图；输出中的图片列表保持不变。",
            "",
            "## 工具统计",
            "",
            ", ".join(f"`{name}`={count}" for name, count in actions.items()),
            "",
            "工具定义来自整个 train+val 的 `action_name` 扫描结果，逐行 `tools` 字段保存为 JSON 字符串。",
            "",
            "## 异常处理",
            "",
            f"发现 `{len(anomalies)}` 次缺少 `<think>` 包装（累计历史导致重复），对应 `{unique_anomalies}` 个唯一原始 assistant 消息；转换器将 `<answer>` 前的有效文本包装为 `<think>`。明细见 `conversion_report.json`。",
            "",
            "## 训练侧约束",
            "",
            "训练时使用 `--template gemma4 --agent_template gemma4`，不要在数据文件中手写 Gemma 4 特殊 token。GKD 文件是纯 prompt，最后的 gold completion 仅保存在 `reference_action`，不得拼回训练上下文。",
            "",
            "## 文件校验",
            "",
            "| 文件 | 字节数 | SHA256 |",
            "|---|---:|---|",
        ]
    )
    for relative_path, metadata in report["files"].items():
        lines.append(f"| `{relative_path}` | {metadata['bytes']} | `{metadata['sha256']}` |")
    lines.append("")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--system-prompt", type=Path, default=DEFAULT_SYSTEM_PROMPT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--image-root", type=Path, default=DEFAULT_IMAGE_ROOT)
    parser.add_argument("--overwrite", action="store_true", help="replace existing generated files")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source_paths = {
        "train": args.dataset_dir / "train_ms_swift.jsonl",
        "val": args.dataset_dir / "val_ms_swift.jsonl",
    }
    for path in [*source_paths.values(), args.system_prompt]:
        if not path.is_file():
            raise ConversionError(f"required input does not exist: {path}")
    if not args.image_root.is_dir():
        raise ConversionError(f"image root does not exist: {args.image_root}")

    sources = {split: read_jsonl(path) for split, path in source_paths.items()}
    system_template = args.system_prompt.read_text(encoding="utf-8")
    required_placeholders = {"{当前日期}", "{本周周一}", "{本月天数}"}
    missing_placeholders = required_placeholders - set(
        placeholder for placeholder in required_placeholders if placeholder in system_template
    )
    if missing_placeholders:
        raise ConversionError(f"system prompt is missing placeholders: {sorted(missing_placeholders)}")

    observed_action_counts: Counter[str] = Counter()
    for rows in sources.values():
        for row in rows:
            name = row.get("action_name")
            if not isinstance(name, str):
                raise ConversionError(f"{row.get('id')}: action_name must be a string")
            observed_action_counts[name] += 1
    tools = build_tools(observed_action_counts)
    tools_json = compact_json(tools)

    anomalies: list[dict[str, Any]] = []
    outputs: dict[str, dict[str, list[dict[str, Any]]]] = {}
    parsed_action_counts: Counter[str] = Counter()
    for split, rows in sources.items():
        sft_rows: list[dict[str, Any]] = []
        gkd_rows: list[dict[str, Any]] = []
        for row in rows:
            sft, gkd, action = convert_row(
                row,
                system_template=system_template,
                tools_json=tools_json,
                image_root=args.image_root,
                anomalies=anomalies,
            )
            sft_rows.append(sft)
            gkd_rows.append(gkd)
            parsed_action_counts[action["name"]] += 1
        validate_output_pair(rows, sft_rows, gkd_rows, tools_json=tools_json)
        outputs[split] = {"sft": sft_rows, "gkd": gkd_rows}

    if parsed_action_counts != observed_action_counts:
        raise ConversionError(
            f"parsed final action counts {dict(parsed_action_counts)} != action_name counts {dict(observed_action_counts)}"
        )

    train_episodes = {str(row["episode_id"]) for row in sources["train"]}
    val_episodes = {str(row["episode_id"]) for row in sources["val"]}
    intersection = sorted(train_episodes & val_episodes)
    if intersection:
        raise ConversionError(f"train/val episode overlap detected: {intersection[:20]}")

    final_relpaths = [
        Path("sft/train.jsonl"),
        Path("sft/val.jsonl"),
        Path("gkd/train.jsonl"),
        Path("gkd/val.jsonl"),
        Path("tools.json"),
        Path("conversion_report.json"),
        Path("conversion_report.md"),
    ]
    existing = [args.output_dir / path for path in final_relpaths if (args.output_dir / path).exists()]
    if existing and not args.overwrite:
        raise ConversionError(
            "output files already exist; pass --overwrite to replace them: "
            + ", ".join(str(path) for path in existing)
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    temp_dir = Path(tempfile.mkdtemp(prefix=".convert-gemma4-opd-", dir=args.output_dir))
    try:
        for subdir in ("sft", "gkd"):
            (temp_dir / subdir).mkdir(parents=True, exist_ok=True)
        write_jsonl(temp_dir / "sft/train.jsonl", outputs["train"]["sft"])
        write_jsonl(temp_dir / "sft/val.jsonl", outputs["val"]["sft"])
        write_jsonl(temp_dir / "gkd/train.jsonl", outputs["train"]["gkd"])
        write_jsonl(temp_dir / "gkd/val.jsonl", outputs["val"]["gkd"])
        (temp_dir / "tools.json").write_text(
            json.dumps(tools, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )

        file_metadata: dict[str, dict[str, Any]] = {}
        for relative_path in final_relpaths[:5]:
            path = temp_dir / relative_path
            file_metadata[str(relative_path)] = {
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }

        report: dict[str, Any] = {
            "generated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "converter": str(Path(__file__).resolve()),
            "inputs": {
                "train": str(source_paths["train"]),
                "val": str(source_paths["val"]),
                "system_prompt": str(args.system_prompt),
                "image_root": str(args.image_root),
            },
            "outputs": str(args.output_dir),
            "splits": {},
            "episode_split_check": {
                "train_episodes": len(train_episodes),
                "val_episodes": len(val_episodes),
                "intersection_count": 0,
                "intersection": [],
            },
            "actions": {
                "counts": dict(observed_action_counts),
                "observed_names": list(observed_action_counts),
                "tools_json_is_string_per_row": True,
            },
            "image_policy": {
                "max_context_images": 3,
                "rule": "step N contains screenshots max(1,N-2)..N",
                "source_image_lists_preserved": True,
                "all_referenced_files_exist": True,
            },
            "system_prompt": {
                "dynamic_date_per_row": True,
                "placeholders": sorted(required_placeholders),
                "literal_ON_preserved": True,
            },
            "sft": {
                "final_assistant_and_tool_call_retained": True,
                "final_completion_loss": True,
            },
            "gkd": {
                "final_gold_completion_removed": True,
                "reference_action_evaluation_only": True,
                "prompt_ends_in": ["user", "tool_response"],
            },
            "anomalies": anomalies,
            "anomaly_summary": {
                "occurrence_count": len(anomalies),
                "unique_content_count": len({item["content_sha256"] for item in anomalies}),
                "note": "source rows contain cumulative history, so one original message can recur in later rows",
            },
            "files": file_metadata,
        }
        for split, rows in sources.items():
            image_counts = Counter(len(row["images"]) for row in rows)
            report["splits"][split] = {
                "rows": len(rows),
                "episodes": len({str(row["episode_id"]) for row in rows}),
                "image_window_counts": {str(key): image_counts[key] for key in sorted(image_counts)},
            }

        (temp_dir / "conversion_report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        (temp_dir / "conversion_report.md").write_text(report_markdown(report), encoding="utf-8")

        for relative_path in final_relpaths:
            destination = args.output_dir / relative_path
            destination.parent.mkdir(parents=True, exist_ok=True)
            os.replace(temp_dir / relative_path, destination)
    finally:
        if temp_dir.exists():
            shutil.rmtree(temp_dir)

    print(f"Converted train={len(sources['train'])}, val={len(sources['val'])}")
    print(f"Observed actions: {dict(observed_action_counts)}")
    print(f"Missing-think repairs: {len(anomalies)}")
    print(f"Episode overlap: {len(intersection)}")
    print(f"Output: {args.output_dir}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ConversionError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
