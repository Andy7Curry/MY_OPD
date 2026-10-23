#!/usr/bin/env python3

"""Validate the local GUI dataset with the actual ms-swift Gemma 4 template."""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import statistics
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

from swift import get_processor, get_template
from swift.dataset import MessagesPreprocessor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=Path, default=Path(os.environ['GEMMA4_MODEL']))
    parser.add_argument('--train-data', type=Path, default=Path(os.environ['GEMMA4_TRAIN_DATA']))
    parser.add_argument('--val-data', type=Path, default=Path(os.environ['GEMMA4_VAL_DATA']))
    parser.add_argument('--image-root', type=Path, default=Path(os.environ['ROOT_IMAGE_DIR']))
    parser.add_argument(
        '--template-samples',
        type=int,
        default=0,
        help='Number of rows to template-encode per split; 0 means all rows.',
    )
    parser.add_argument('--coverage', type=float, default=1.0)
    parser.add_argument('--length-cap', type=int, default=16384)
    parser.add_argument('--report', type=Path)
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding='utf-8') as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(f'{path}:{line_number}: invalid JSON: {error}') from error
    if not rows:
        raise ValueError(f'{path}: dataset is empty')
    return rows


def structural_stats(
    split: str,
    rows: list[dict[str, Any]],
    image_root: Path,
) -> tuple[dict[str, Any], set[str], set[str], set[str]]:
    ids: list[str] = []
    episodes: list[str] = []
    images: list[str] = []
    roles: Counter[str] = Counter()
    actions: Counter[str] = Counter()
    message_counts: list[int] = []
    image_counts: list[int] = []
    errors: list[str] = []

    for row_index, row in enumerate(rows):
        messages = row.get('messages')
        row_images = row.get('images')
        if not isinstance(messages, list) or not messages:
            errors.append(f'row {row_index}: messages must be a non-empty list')
            continue
        if not isinstance(row_images, list) or not row_images:
            errors.append(f'row {row_index}: images must be a non-empty list')
            continue

        row_id = row.get('id')
        episode_id = row.get('episode_id')
        if not isinstance(row_id, str) or not row_id:
            errors.append(f'row {row_index}: missing id')
        else:
            ids.append(row_id)
        if not isinstance(episode_id, str) or not episode_id:
            errors.append(f'row {row_index}: missing episode_id')
        else:
            episodes.append(episode_id)

        true_loss_indices = []
        for message_index, message in enumerate(messages):
            role = message.get('role')
            content = message.get('content')
            roles[str(role)] += 1
            if role not in {'system', 'user', 'assistant', 'tool_call', 'tool_response', 'tool'}:
                errors.append(f'row {row_index}, message {message_index}: invalid role {role!r}')
            if not isinstance(content, str):
                errors.append(f'row {row_index}, message {message_index}: content must be a string')
            if message.get('loss') is True:
                true_loss_indices.append(message_index)

        if (len(messages) >= 2 and messages[-2].get('role') == 'assistant'
                and messages[-1].get('role') == 'tool_call'):
            expected_loss_indices = [len(messages) - 2, len(messages) - 1]
        elif messages[-1].get('role') == 'assistant':
            expected_loss_indices = [len(messages) - 1]
        else:
            expected_loss_indices = []

        if true_loss_indices != expected_loss_indices:
            errors.append(
                f'row {row_index}: expected supervised suffix {expected_loss_indices}; '
                f'found loss=true at {true_loss_indices}'
            )
        if not expected_loss_indices:
            errors.append(
                f'row {row_index}: final messages must be assistant or assistant + tool_call; '
                f'found final role {messages[-1].get("role")!r}'
            )

        for image in row_images:
            if not isinstance(image, str):
                errors.append(f'row {row_index}: image path is not a string')
                continue
            images.append(image)
            image_path = Path(image)
            resolved = image_path if image_path.is_absolute() else image_root / image_path
            if not resolved.is_file():
                errors.append(f'row {row_index}: missing image {resolved}')

        actions[str(row.get('action_name'))] += 1
        message_counts.append(len(messages))
        image_counts.append(len(row_images))

    duplicate_ids = len(ids) - len(set(ids))
    if duplicate_ids:
        errors.append(f'{duplicate_ids} duplicate ids')

    if errors:
        preview = '\n'.join(f'  - {error}' for error in errors[:30])
        suffix = '' if len(errors) <= 30 else f'\n  ... {len(errors) - 30} more errors'
        raise ValueError(f'{split}: structural validation failed:\n{preview}{suffix}')

    stats = {
        'rows': len(rows),
        'episodes': len(set(episodes)),
        'unique_images': len(set(images)),
        'total_image_references': len(images),
        'message_count': {
            'min': min(message_counts),
            'max': max(message_counts),
            'mean': statistics.fmean(message_counts),
        },
        'image_count': {
            'min': min(image_counts),
            'max': max(image_counts),
            'mean': statistics.fmean(image_counts),
        },
        'roles': dict(roles),
        'actions': dict(actions),
    }
    return stats, set(ids), set(episodes), set(images)


def percentile(sorted_values: list[int], fraction: float) -> int:
    index = max(0, math.ceil(len(sorted_values) * fraction) - 1)
    return sorted_values[index]


def template_stats(
    split: str,
    rows: list[dict[str, Any]],
    template: Any,
    sample_limit: int,
) -> dict[str, Any]:
    selected_rows = rows if sample_limit <= 0 else rows[:sample_limit]
    started_at = time.monotonic()
    lengths: list[int] = []
    supervised_tokens: list[int] = []
    multimodal_tokens: list[int] = []
    errors: list[str] = []
    preprocessor = MessagesPreprocessor()

    for row_index, row in enumerate(selected_rows):
        try:
            processed_row = preprocessor.preprocess(copy.deepcopy(row))
            preprocessor._check_messages(processed_row)
            unnamed_tool_responses = [
                message for message in processed_row['messages']
                if message['role'] in {'tool', 'tool_response'} and not message.get('name')
            ]
            if unnamed_tool_responses:
                raise ValueError(
                    f'{len(unnamed_tool_responses)} tool responses lost their name during preprocessing')
            encoded = template.encode(processed_row)
        # A preflight report should identify a bad row regardless of which
        # processor/template layer raised the encoding error.
        except Exception as error:  # noqa: BLE001
            errors.append(f'row {row_index}: {type(error).__name__}: {error}')
            continue

        input_ids = encoded.get('input_ids') or []
        labels = encoded.get('labels') or []
        mm_token_type_ids = encoded.get('mm_token_type_ids')
        supervised = sum(label != -100 for label in labels)
        mm_tokens = int(mm_token_type_ids.sum().item()) if mm_token_type_ids is not None else 0

        if not input_ids or len(input_ids) != len(labels):
            errors.append(
                f'row {row_index}: invalid input/label lengths '
                f'({len(input_ids)} vs {len(labels)})'
            )
        if supervised <= 0:
            errors.append(f'row {row_index}: no supervised tokens')
        if mm_tokens <= 0:
            errors.append(f'row {row_index}: no multimodal tokens')

        lengths.append(len(input_ids))
        supervised_tokens.append(supervised)
        multimodal_tokens.append(mm_tokens)
        completed = row_index + 1
        if completed % 100 == 0 or completed == len(selected_rows):
            elapsed = time.monotonic() - started_at
            print(
                f'{split}: template-encoded {completed}/{len(selected_rows)} rows '
                f'in {elapsed:.1f}s',
                file=sys.stderr,
                flush=True,
            )

    if errors:
        preview = '\n'.join(f'  - {error}' for error in errors[:30])
        suffix = '' if len(errors) <= 30 else f'\n  ... {len(errors) - 30} more errors'
        raise ValueError(f'{split}: template validation failed:\n{preview}{suffix}')

    lengths.sort()
    return {
        'encoded_rows': len(lengths),
        'lengths': lengths,
        'tokens': {
            'min': lengths[0],
            'p50': percentile(lengths, 0.50),
            'p95': percentile(lengths, 0.95),
            'p99': percentile(lengths, 0.99),
            'max': lengths[-1],
            'mean': statistics.fmean(lengths),
        },
        'supervised_tokens': {
            'min': min(supervised_tokens),
            'max': max(supervised_tokens),
            'mean': statistics.fmean(supervised_tokens),
        },
        'multimodal_tokens': {
            'min': min(multimodal_tokens),
            'max': max(multimodal_tokens),
            'mean': statistics.fmean(multimodal_tokens),
        },
    }


def main() -> int:
    args = parse_args()
    if not 0 < args.coverage <= 1:
        raise ValueError('--coverage must be in (0, 1]')
    if args.length_cap < 1024:
        raise ValueError('--length-cap must be at least 1024')

    os.environ['ROOT_IMAGE_DIR'] = str(args.image_root.resolve())
    train_rows = load_jsonl(args.train_data)
    val_rows = load_jsonl(args.val_data)

    train_struct, train_ids, train_episodes, train_images = structural_stats(
        'train', train_rows, args.image_root
    )
    val_struct, val_ids, val_episodes, val_images = structural_stats(
        'validation', val_rows, args.image_root
    )
    overlap = {
        'ids': len(train_ids & val_ids),
        'episodes': len(train_episodes & val_episodes),
        'images': len(train_images & val_images),
    }
    if any(overlap.values()):
        raise ValueError(f'train/validation leakage detected: {overlap}')

    processor = get_processor(str(args.model), download_model=False)
    template = get_template(
        processor,
        max_length=None,
        template_type='gemma4',
        agent_template='gemma4',
        truncation_strategy='raise',
        padding_free=False,
        loss_scale='ignore_empty_think',
        add_non_thinking_prefix=True,
        preserve_thinking=False,
    )
    template.set_mode('train')

    train_template = template_stats('train', train_rows, template, args.template_samples)
    val_template = template_stats('validation', val_rows, template, args.template_samples)
    combined_lengths = sorted(train_template.pop('lengths') + val_template.pop('lengths'))
    required_length = percentile(combined_lengths, args.coverage)
    rounded_length = max(4096, math.ceil(required_length / 1024) * 1024)
    recommended_length = min(rounded_length, args.length_cap)
    retained = sum(length <= recommended_length for length in combined_lengths)
    actual_coverage = retained / len(combined_lengths)

    report = {
        'model': str(args.model.resolve()),
        'image_root': str(args.image_root.resolve()),
        'structure': {'train': train_struct, 'validation': val_struct, 'overlap': overlap},
        'template': {'train': train_template, 'validation': val_template},
        'length_recommendation': {
            'requested_coverage': args.coverage,
            'required_unrounded': required_length,
            'recommended_max_length': recommended_length,
            'actual_coverage': actual_coverage,
            'length_cap': args.length_cap,
            'needs_context_compaction': actual_coverage < args.coverage,
        },
    }

    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    print(rendered)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(f'{rendered}\n', encoding='utf-8')

    if actual_coverage < args.coverage:
        print(
            'ERROR: token-length coverage is below the requested threshold; '
            'compact long contexts before training.',
            file=sys.stderr,
        )
        return 2
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
