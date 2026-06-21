#!/usr/bin/env python3
"""Convert meter annotations to the ms-swift multimodal SFT JSONL format.

Raw JSONL example, single meter:
{"image":"000001.jpg","meter_bbox":[120,80,520,430],"reading_bbox":[230,210,410,260],"reading":"123.45","unit":"kWh","meter_type":"digital"}

Raw JSONL example, multiple meters:
{"image":"000002.jpg","meters":[{"meter_bbox":[90,60,460,390],"reading_bbox":[180,190,360,240],"reading":"0.62","unit":"MPa","meter_type":"pointer"}]}
"""

import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from PIL import Image


DEFAULT_SYSTEM = (
    '你是一个工业表计读数识别助手。你需要定位图片中的表计和读数区域，'
    '并识别读数。只输出合法JSON，不要输出解释。'
)

DEFAULT_PROMPT = '<image>请识别图中的表计位置和读数，只输出JSON。'


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Convert meter annotations to ms-swift JSONL.')
    parser.add_argument('--input', required=True, help='Input raw annotation jsonl.')
    parser.add_argument('--image-root', default='', help='Root directory for relative image paths.')
    parser.add_argument('--output', required=True, help='Output ms-swift jsonl.')
    parser.add_argument('--val-output', default='', help='Optional validation output jsonl.')
    parser.add_argument('--val-ratio', type=float, default=0.0, help='Validation split ratio if --val-output is set.')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--system', default=DEFAULT_SYSTEM)
    parser.add_argument('--prompt', default=DEFAULT_PROMPT)
    parser.add_argument('--allow-missing-image', action='store_true', help='Skip image existence and bbox boundary checks.')
    parser.add_argument('--pretty-answer', action='store_true', help='Pretty-print assistant JSON answer.')
    return parser.parse_args()


def iter_jsonl(path: Path) -> Iterable[Tuple[int, Dict[str, Any]]]:
    with path.open('r', encoding='utf-8') as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield line_no, json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f'{path}:{line_no}: invalid JSON: {e}') from e


def resolve_image(image_root: Path, image: Any) -> Path:
    if isinstance(image, list):
        if len(image) != 1:
            raise ValueError('each sample must contain exactly one image')
        image = image[0]
    if not isinstance(image, str) or not image.strip():
        raise ValueError('image must be a non-empty string')
    path = Path(image)
    if not path.is_absolute():
        path = image_root / path
    return path.resolve()


def image_size(path: Path, allow_missing: bool) -> Optional[Tuple[int, int]]:
    if allow_missing:
        return None
    if not path.exists():
        raise FileNotFoundError(f'image not found: {path}')
    with Image.open(path) as img:
        return img.size


def normalize_bbox(name: str, value: Any, size: Optional[Tuple[int, int]], line_no: int) -> List[int]:
    if not isinstance(value, list) or len(value) != 4:
        raise ValueError(f'line {line_no}: {name} must be [x1, y1, x2, y2]')
    try:
        x1, y1, x2, y2 = [int(round(float(v))) for v in value]
    except (TypeError, ValueError) as e:
        raise ValueError(f'line {line_no}: {name} contains non-numeric value: {value}') from e
    if x1 >= x2 or y1 >= y2:
        raise ValueError(f'line {line_no}: {name} has invalid order: {value}')
    if size is not None:
        width, height = size
        if x1 < 0 or y1 < 0 or x2 > width or y2 > height:
            raise ValueError(f'line {line_no}: {name} out of image bounds {width}x{height}: {value}')
    return [x1, y1, x2, y2]


def normalize_meter(raw: Dict[str, Any], size: Optional[Tuple[int, int]], line_no: int, index: int) -> Dict[str, Any]:
    reading = str(raw.get('reading', '')).strip()
    if not reading:
        raise ValueError(f'line {line_no}: meters[{index}].reading is required')
    if 'meter_bbox' not in raw:
        raise ValueError(f'line {line_no}: meters[{index}].meter_bbox is required')

    result = {
        'meter_bbox': normalize_bbox('meter_bbox', raw['meter_bbox'], size, line_no),
        'reading': reading,
        'unit': str(raw.get('unit', '')).strip(),
        'meter_type': str(raw.get('meter_type', '')).strip(),
    }
    if raw.get('reading_bbox') is not None:
        result['reading_bbox'] = normalize_bbox('reading_bbox', raw['reading_bbox'], size, line_no)
    else:
        result['reading_bbox'] = None
    return result


def normalize_meters(row: Dict[str, Any], size: Optional[Tuple[int, int]], line_no: int) -> List[Dict[str, Any]]:
    meters = row.get('meters')
    if meters is None:
        meters = [row]
    if not isinstance(meters, list) or not meters:
        raise ValueError(f'line {line_no}: meters must be a non-empty list')
    return [normalize_meter(meter, size, line_no, i) for i, meter in enumerate(meters)]


def build_sample(image_path: Path, meters: List[Dict[str, Any]], system: str, prompt: str, pretty: bool) -> Dict[str, Any]:
    answer = {'meters': meters}
    if pretty:
        content = json.dumps(answer, ensure_ascii=False, indent=2)
    else:
        content = json.dumps(answer, ensure_ascii=False, separators=(',', ':'))
    messages = []
    if system:
        messages.append({'role': 'system', 'content': system})
    messages.extend([
        {'role': 'user', 'content': prompt},
        {'role': 'assistant', 'content': content},
    ])
    return {'messages': messages, 'images': [str(image_path)]}


def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')


def main() -> None:
    args = parse_args()
    if not 0 <= args.val_ratio < 1:
        raise ValueError('--val-ratio must be in [0, 1)')

    input_path = Path(args.input)
    image_root = Path(args.image_root or '.')
    rows = []
    for line_no, row in iter_jsonl(input_path):
        image_path = resolve_image(image_root, row.get('image', row.get('images')))
        size = image_size(image_path, args.allow_missing_image)
        meters = normalize_meters(row, size, line_no)
        rows.append(build_sample(image_path, meters, args.system, args.prompt, args.pretty_answer))

    if args.val_output and args.val_ratio > 0:
        random.Random(args.seed).shuffle(rows)
        val_count = int(round(len(rows) * args.val_ratio))
        val_rows = rows[:val_count]
        train_rows = rows[val_count:]
        if not train_rows:
            raise ValueError('no training rows left after validation split')
        write_jsonl(Path(args.output), train_rows)
        write_jsonl(Path(args.val_output), val_rows)
        print(f'wrote train rows: {len(train_rows)} -> {args.output}')
        print(f'wrote val rows: {len(val_rows)} -> {args.val_output}')
    else:
        write_jsonl(Path(args.output), rows)
        print(f'wrote rows: {len(rows)} -> {args.output}')


if __name__ == '__main__':
    main()
