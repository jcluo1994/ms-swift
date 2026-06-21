#!/usr/bin/env python3
"""A small GUI annotator for meter reading datasets.

It opens images, lets you draw meter/reading boxes, and writes ms-swift
multimodal SFT JSONL directly.
"""

import argparse
import json
import os
import sys
import tkinter as tk
from pathlib import Path, PurePath
from tkinter import filedialog, messagebox, ttk
from typing import Any, Callable, Dict, List, Optional, Tuple

from PIL import Image, ImageTk


SYSTEM_PROMPT = (
    '你是一个工业表计读数识别助手。你需要定位图片中的表计和读数区域，'
    '并识别读数。只输出合法JSON，不要输出解释。'
)
USER_PROMPT = '<image>请识别图中的表计位置和读数，只输出JSON。'
IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='GUI tool for meter annotation and ms-swift JSONL export.')
    parser.add_argument('--image-dir', default='', help='Directory containing images to annotate.')
    parser.add_argument('--output', default='', help='Output ms-swift training jsonl.')
    parser.add_argument('--system', default=SYSTEM_PROMPT)
    parser.add_argument('--prompt', default=USER_PROMPT)
    parser.add_argument('--recursive', action='store_true', help='Scan image directory recursively.')
    return parser.parse_args()


def list_images(image_dir: Path, recursive: bool) -> List[Path]:
    files = image_dir.rglob('*') if recursive else image_dir.iterdir()
    images = [p.resolve() for p in files if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
    return sorted(images)


def to_rel_key(abs_path: Any, image_dir: Path) -> Optional[str]:
    """把图片绝对路径转成「相对 image_dir 的正斜杠相对路径」，作为统一的标注 key 与 jsonl 落盘格式。

    这种形式不含盘符、不含反斜杠、不做大小写规范化，因此同一份 jsonl 在 Windows/macOS/Linux 上
    都得到完全一致的 key，真正跨平台。训练时配合 ms-swift 的 ROOT_IMAGE_DIR 环境变量
    （指向本机图片根目录=image_dir）即可还原真实路径。

    不在 image_dir 下（含 Windows 跨盘符）时返回 None。
    """
    try:
        rel = os.path.relpath(str(Path(abs_path)), str(image_dir))
    except ValueError:
        # Windows 上跨盘符（如 C: vs H:）relpath 会抛 ValueError
        return None
    if rel.startswith('..'):
        return None
    return PurePath(rel).as_posix()


def image_key(path: Any, image_dir: Path) -> Optional[str]:
    """当前会话中图片的统一 key：相对 image_dir 的正斜杠相对路径。"""
    return to_rel_key(path, image_dir)


def resolve_key(raw: str, image_dir: Path, key_to_path: Dict[str, Path],
                name_to_keys: Dict[str, List[str]]) -> Optional[str]:
    """把 jsonl 里记录的图片路径解析成当前会话的统一 key，兼容旧的绝对路径数据。

    三级回退：
    1. 新格式相对路径：直接 as_posix 后命中当前图片集。
    2. 旧绝对路径恰好落在当前 image_dir 下：转相对路径命中。
    3. basename 回退：按文件名匹配当前图片集（处理跨机器/跨平台搬运的旧数据）。
       重名（多个命中）时告警跳过，避免标错图；无命中返回 None（交由调用方当孤儿保留）。
    """
    # 1) 新格式相对路径
    cand = PurePath(raw).as_posix()
    if cand in key_to_path:
        return cand
    # 2) 旧绝对路径，且恰好在当前 image_dir 下
    rel = to_rel_key(raw, image_dir)
    if rel is not None and rel in key_to_path:
        return rel
    # 3) basename 回退（兼容 H:\... 这类异机/异平台旧路径）
    name = PurePath(raw.replace('\\', '/')).name.lower()
    keys = name_to_keys.get(name, [])
    if len(keys) == 1:
        return keys[0]
    if len(keys) > 1:
        print(f'warning: 文件名 {name!r} 在当前图片目录匹配到 {len(keys)} 张同名图，'
              f'无法确定，已跳过以免标错: {raw}', file=sys.stderr)
        return None
    return None



def assistant_content(sample: Dict[str, Any]) -> Optional[str]:
    for message in sample.get('messages', []):
        if message.get('role') == 'assistant':
            return message.get('content')
    return None


def load_existing_annotations(
    output: Path,
    image_dir: Path,
    key_to_path: Dict[str, Path],
    name_to_keys: Dict[str, List[str]],
) -> Tuple[Dict[str, List[Dict[str, Any]]], List[str]]:
    """读取已有 jsonl 标注。

    返回 (annotations, orphan_lines)：
    - annotations: rel key -> meters，key 已统一为相对 image_dir 的正斜杠路径。
    - orphan_lines: 无法匹配当前图片目录的原始 jsonl 行（异机/异平台旧数据），原样保留，
      导出时再写回，避免丢失历史标注。
    """
    annotations: Dict[str, List[Dict[str, Any]]] = {}
    orphan_lines: List[str] = []
    if not output.exists():
        return annotations, orphan_lines
    with output.open('r', encoding='utf-8') as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                sample = json.loads(line)
                image_path = sample['images'][0]
                content = assistant_content(sample)
                if not content:
                    continue
                answer = json.loads(content)
                key = resolve_key(image_path, image_dir, key_to_path, name_to_keys)
                if key is None:
                    # 当前图片目录里没有这张图，作为孤儿行原样保留
                    orphan_lines.append(line)
                    continue
                annotations[key] = answer.get('meters', [])
            except Exception as e:
                print(f'warning: skip {output}:{line_no}: {e}', file=sys.stderr)
    return annotations, orphan_lines


def make_sample(image_key: str, meters: List[Dict[str, Any]], system: str, prompt: str) -> Dict[str, Any]:
    messages = []
    if system:
        messages.append({'role': 'system', 'content': system})
    answer = json.dumps({'meters': meters}, ensure_ascii=False, separators=(',', ':'))
    messages.extend([
        {'role': 'user', 'content': prompt},
        {'role': 'assistant', 'content': answer},
    ])
    return {'messages': messages, 'images': [image_key]}


def save_annotations(
    output: Path,
    annotations: Dict[str, List[Dict[str, Any]]],
    orphan_lines: List[str],
    system: str,
    prompt: str,
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    image_keys = sorted(key for key, meters in annotations.items() if meters)
    # 原子写：先写到同目录的临时文件，flush+fsync 落盘后再 os.replace 覆盖，
    # 避免自动保存写到一半被中断时损坏已有标注文件。
    tmp = output.with_suffix(output.suffix + '.tmp')
    with tmp.open('w', encoding='utf-8') as f:
        for line in orphan_lines:
            f.write(line + '\n')
        for image_key in image_keys:
            sample = make_sample(image_key, annotations[image_key], system, prompt)
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, output)


class MeterAnnotator:
    def __init__(self, root: tk.Tk, args: argparse.Namespace):
        self.root = root
        self.args = args
        self.image_dir = Path(args.image_dir).resolve()
        self.output = Path(args.output).resolve()
        self.images = list_images(self.image_dir, args.recursive)
        if not self.images:
            raise RuntimeError(f'No images found in {self.image_dir}')
        self.key_to_path: Dict[str, Path] = {}
        self.name_to_keys: Dict[str, List[str]] = {}
        for path in self.images:
            key = image_key(path, self.image_dir)
            if key is None:
                continue
            self.key_to_path[key] = path
            self.name_to_keys.setdefault(path.name.lower(), []).append(key)
        self.annotations, self.orphan_lines = load_existing_annotations(
            self.output,
            self.image_dir,
            self.key_to_path,
            self.name_to_keys,
        )
        self.index = 0

        self.mode = tk.StringVar(value='meter_bbox')
        self.reading = tk.StringVar()
        self.unit = tk.StringVar()
        self.meter_type = tk.StringVar()
        self.status = tk.StringVar()
        self.current_meter_bbox: Optional[List[int]] = None
        self.current_reading_bbox: Optional[List[int]] = None
        self.current_image: Optional[Image.Image] = None
        self.tk_image: Optional[ImageTk.PhotoImage] = None
        self.scale = 1.0
        self.offset_x = 0
        self.offset_y = 0
        self.drag_start: Optional[Tuple[int, int]] = None
        self.drag_rect_id: Optional[int] = None
        self.box_ids: List[int] = []

        self._build_ui()
        self._load_image()

    def _build_ui(self) -> None:
        self.root.title('Meter Dataset Annotator')
        self.root.geometry('1220x780')

        main = ttk.Frame(self.root)
        main.pack(fill=tk.BOTH, expand=True)

        self.canvas = tk.Canvas(main, bg='#202020', width=900, height=720, highlightthickness=0)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.canvas.bind('<ButtonPress-1>', self._on_press)
        self.canvas.bind('<B1-Motion>', self._on_drag)
        self.canvas.bind('<ButtonRelease-1>', self._on_release)
        self.canvas.bind('<Configure>', lambda _: self._render_image())

        panel = ttk.Frame(main, width=310, padding=10)
        panel.pack(side=tk.RIGHT, fill=tk.Y)
        panel.pack_propagate(False)

        ttk.Label(panel, text='框选类型').pack(anchor=tk.W)
        ttk.Radiobutton(panel, text='表计整体框 meter_bbox', variable=self.mode, value='meter_bbox').pack(anchor=tk.W)
        ttk.Radiobutton(panel, text='读数区域框 reading_bbox', variable=self.mode, value='reading_bbox').pack(anchor=tk.W)

        ttk.Separator(panel).pack(fill=tk.X, pady=8)
        ttk.Label(panel, text='读数 reading').pack(anchor=tk.W)
        ttk.Entry(panel, textvariable=self.reading).pack(fill=tk.X)
        ttk.Label(panel, text='单位 unit').pack(anchor=tk.W, pady=(8, 0))
        ttk.Entry(panel, textvariable=self.unit).pack(fill=tk.X)
        ttk.Label(panel, text='表计类型 meter_type').pack(anchor=tk.W, pady=(8, 0))
        ttk.Entry(panel, textvariable=self.meter_type).pack(fill=tk.X)

        ttk.Button(panel, text='添加表计', command=self._add_meter).pack(fill=tk.X, pady=(10, 0))
        ttk.Button(panel, text='清空当前框', command=self._clear_current_boxes).pack(fill=tk.X, pady=(6, 0))

        ttk.Separator(panel).pack(fill=tk.X, pady=8)
        ttk.Label(panel, text='当前图片已标注表计').pack(anchor=tk.W)
        self.meter_list = tk.Listbox(panel, height=9)
        self.meter_list.pack(fill=tk.X)
        ttk.Button(panel, text='删除选中表计', command=self._delete_selected_meter).pack(fill=tk.X, pady=(6, 0))

        ttk.Separator(panel).pack(fill=tk.X, pady=8)
        nav = ttk.Frame(panel)
        nav.pack(fill=tk.X)
        ttk.Button(nav, text='上一张', command=self._prev_image).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(nav, text='下一张', command=self._next_image).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(6, 0))
        ttk.Button(panel, text='保存当前并写出JSONL', command=self._save_all).pack(fill=tk.X, pady=(8, 0))
        ttk.Button(panel, text='跳到下一张未标注', command=self._next_unlabeled).pack(fill=tk.X, pady=(6, 0))

        ttk.Separator(panel).pack(fill=tk.X, pady=8)
        ttk.Label(panel, textvariable=self.status, wraplength=290).pack(anchor=tk.W)

        self.root.bind('<KeyPress-m>', lambda event: self._run_shortcut(event, lambda: self.mode.set('meter_bbox')))
        self.root.bind('<KeyPress-r>', lambda event: self._run_shortcut(event, lambda: self.mode.set('reading_bbox')))
        self.root.bind('<KeyPress-a>', lambda event: self._run_shortcut(event, self._add_meter))
        self.root.bind('<KeyPress-s>', lambda event: self._run_shortcut(event, self._save_all))
        self.root.bind('<Left>', lambda event: self._run_shortcut(event, self._prev_image))
        self.root.bind('<Right>', lambda event: self._run_shortcut(event, self._next_image))

    def _run_shortcut(self, event: tk.Event, action: Callable[[], None]) -> Optional[str]:
        if self._is_text_input(event.widget):
            return None
        action()
        return 'break'

    def _is_text_input(self, widget: tk.Widget) -> bool:
        return widget.winfo_class() in {
            'Entry',
            'TEntry',
            'Text',
            'Spinbox',
            'TSpinbox',
            'Combobox',
            'TCombobox',
        }

    def _image_key(self) -> str:
        key = image_key(self.images[self.index], self.image_dir)
        if key is None:
            raise RuntimeError(f'Image is outside image dir: {self.images[self.index]}')
        return key

    def _load_image(self) -> None:
        path = self.images[self.index]
        self.current_image = Image.open(path).convert('RGB')
        self.current_meter_bbox = None
        self.current_reading_bbox = None
        self.reading.set('')
        self.unit.set('')
        self.meter_type.set('')
        self._render_image()
        self._refresh_meter_list()
        self._set_status()

    def _render_image(self) -> None:
        if self.current_image is None:
            return
        self.canvas.delete('all')
        self.box_ids.clear()
        cw = max(self.canvas.winfo_width(), 1)
        ch = max(self.canvas.winfo_height(), 1)
        iw, ih = self.current_image.size
        self.scale = min(cw / iw, ch / ih)
        dw = max(int(iw * self.scale), 1)
        dh = max(int(ih * self.scale), 1)
        self.offset_x = (cw - dw) // 2
        self.offset_y = (ch - dh) // 2
        resized = self.current_image.resize((dw, dh))
        self.tk_image = ImageTk.PhotoImage(resized)
        self.canvas.create_image(self.offset_x, self.offset_y, anchor=tk.NW, image=self.tk_image)
        self._draw_saved_boxes()
        self._draw_current_boxes()

    def _original_to_canvas(self, bbox: List[int]) -> Tuple[int, int, int, int]:
        x1, y1, x2, y2 = bbox
        return (
            int(x1 * self.scale + self.offset_x),
            int(y1 * self.scale + self.offset_y),
            int(x2 * self.scale + self.offset_x),
            int(y2 * self.scale + self.offset_y),
        )

    def _canvas_to_original(self, x1: int, y1: int, x2: int, y2: int) -> List[int]:
        assert self.current_image is not None
        iw, ih = self.current_image.size
        ox1 = int(round((min(x1, x2) - self.offset_x) / self.scale))
        oy1 = int(round((min(y1, y2) - self.offset_y) / self.scale))
        ox2 = int(round((max(x1, x2) - self.offset_x) / self.scale))
        oy2 = int(round((max(y1, y2) - self.offset_y) / self.scale))
        ox1 = max(0, min(iw, ox1))
        oy1 = max(0, min(ih, oy1))
        ox2 = max(0, min(iw, ox2))
        oy2 = max(0, min(ih, oy2))
        return [ox1, oy1, ox2, oy2]

    def _draw_box(self, bbox: List[int], color: str, label: str, width: int = 2) -> None:
        x1, y1, x2, y2 = self._original_to_canvas(bbox)
        self.canvas.create_rectangle(x1, y1, x2, y2, outline=color, width=width)
        self.canvas.create_text(x1 + 4, y1 + 4, text=label, anchor=tk.NW, fill=color)

    def _draw_saved_boxes(self) -> None:
        for i, meter in enumerate(self.annotations.get(self._image_key(), []), start=1):
            self._draw_box(meter['meter_bbox'], '#34d399', f'meter {i}', 2)
            if meter.get('reading_bbox'):
                self._draw_box(meter['reading_bbox'], '#60a5fa', f'reading {i}', 2)

    def _draw_current_boxes(self) -> None:
        if self.current_meter_bbox:
            self._draw_box(self.current_meter_bbox, '#fbbf24', 'current meter', 3)
        if self.current_reading_bbox:
            self._draw_box(self.current_reading_bbox, '#f472b6', 'current reading', 3)

    def _on_press(self, event: tk.Event) -> None:
        self.drag_start = (event.x, event.y)
        self.drag_rect_id = self.canvas.create_rectangle(event.x, event.y, event.x, event.y, outline='#fef08a', width=2)

    def _on_drag(self, event: tk.Event) -> None:
        if self.drag_start and self.drag_rect_id:
            x0, y0 = self.drag_start
            self.canvas.coords(self.drag_rect_id, x0, y0, event.x, event.y)

    def _on_release(self, event: tk.Event) -> None:
        if not self.drag_start:
            return
        x0, y0 = self.drag_start
        bbox = self._canvas_to_original(x0, y0, event.x, event.y)
        if self.drag_rect_id:
            self.canvas.delete(self.drag_rect_id)
        self.drag_start = None
        self.drag_rect_id = None
        if bbox[2] - bbox[0] < 4 or bbox[3] - bbox[1] < 4:
            return
        if self.mode.get() == 'meter_bbox':
            self.current_meter_bbox = bbox
        else:
            self.current_reading_bbox = bbox
        self._render_image()
        self._set_status()

    def _current_meters(self) -> List[Dict[str, Any]]:
        return self.annotations.setdefault(self._image_key(), [])

    def _add_meter(self) -> None:
        reading = self.reading.get().strip()
        if not self.current_meter_bbox:
            messagebox.showerror('缺少表计框', '请先框选表计整体位置。')
            return
        if not reading:
            messagebox.showerror('缺少读数', '请填写读数 reading。')
            return
        meter = {
            'meter_bbox': self.current_meter_bbox,
            'reading_bbox': self.current_reading_bbox,
            'reading': reading,
            'unit': self.unit.get().strip(),
            'meter_type': self.meter_type.get().strip(),
        }
        self._current_meters().append(meter)
        self._clear_current_boxes()
        self._refresh_meter_list()
        self._save_all(silent=True)

    def _clear_current_boxes(self) -> None:
        self.current_meter_bbox = None
        self.current_reading_bbox = None
        self.reading.set('')
        self.unit.set('')
        self.meter_type.set('')
        self._render_image()
        self._set_status()

    def _delete_selected_meter(self) -> None:
        selection = self.meter_list.curselection()
        if not selection:
            return
        del self._current_meters()[selection[0]]
        self._refresh_meter_list()
        self._render_image()
        self._save_all(silent=True)

    def _refresh_meter_list(self) -> None:
        self.meter_list.delete(0, tk.END)
        for i, meter in enumerate(self.annotations.get(self._image_key(), []), start=1):
            unit = f' {meter.get("unit")}' if meter.get('unit') else ''
            meter_type = f' [{meter.get("meter_type")}]' if meter.get('meter_type') else ''
            self.meter_list.insert(tk.END, f'{i}. {meter.get("reading", "")}{unit}{meter_type}')

    def _save_all(self, silent: bool = False) -> None:
        save_annotations(self.output, self.annotations, self.orphan_lines, self.args.system, self.args.prompt)
        if not silent:
            messagebox.showinfo('已保存', f'已写出：{self.output}')
        self._set_status()

    def _prev_image(self) -> None:
        if self.index > 0:
            self.index -= 1
            self._load_image()

    def _next_image(self) -> None:
        if self.index < len(self.images) - 1:
            self.index += 1
            self._load_image()

    def _next_unlabeled(self) -> None:
        for i in range(self.index + 1, len(self.images)):
            key = image_key(self.images[i], self.image_dir)
            if key is not None and not self.annotations.get(key):
                self.index = i
                self._load_image()
                return
        messagebox.showinfo('没有更多', '后面没有未标注图片了。')

    def _set_status(self) -> None:
        labeled = sum(1 for meters in self.annotations.values() if meters)
        current = self.images[self.index]
        self.status.set(
            f'图片 {self.index + 1}/{len(self.images)}\n'
            f'已标注图片：{labeled}\n'
            f'未匹配历史行：{len(self.orphan_lines)}\n'
            f'当前：{current.name}\n'
            f'快捷键：m表计框，r读数框，a添加，s保存，左右切图'
        )


class PathSelector:
    def __init__(self, root: tk.Tk, args: argparse.Namespace):
        self.root = root
        self.args = args
        self.image_dir = tk.StringVar(value=args.image_dir)
        self.output = tk.StringVar(value=args.output)
        self.recursive = tk.BooleanVar(value=args.recursive)
        self._build_ui()

    def _build_ui(self) -> None:
        self.root.title('选择表计数据集路径')
        self.root.geometry('720x220')
        frame = ttk.Frame(self.root, padding=16)
        frame.pack(fill=tk.BOTH, expand=True)

        ttk.Label(frame, text='图片目录').grid(row=0, column=0, sticky=tk.W)
        ttk.Entry(frame, textvariable=self.image_dir).grid(row=1, column=0, sticky=tk.EW, pady=(4, 10))
        ttk.Button(frame, text='选择目录', command=self._choose_image_dir).grid(row=1, column=1, padx=(8, 0), pady=(4, 10))

        ttk.Label(frame, text='输出 JSONL 文件').grid(row=2, column=0, sticky=tk.W)
        ttk.Entry(frame, textvariable=self.output).grid(row=3, column=0, sticky=tk.EW, pady=(4, 10))
        ttk.Button(frame, text='选择文件', command=self._choose_output).grid(row=3, column=1, padx=(8, 0), pady=(4, 10))

        ttk.Checkbutton(frame, text='递归扫描子目录图片', variable=self.recursive).grid(row=4, column=0, sticky=tk.W)
        ttk.Button(frame, text='开始标注', command=self._start).grid(row=5, column=0, columnspan=2, sticky=tk.EW, pady=(14, 0))

        frame.columnconfigure(0, weight=1)

    def _choose_image_dir(self) -> None:
        path = filedialog.askdirectory(title='选择图片目录')
        if path:
            self.image_dir.set(path)

    def _choose_output(self) -> None:
        path = filedialog.asksaveasfilename(
            title='选择输出 JSONL 文件',
            defaultextension='.jsonl',
            filetypes=[('JSON Lines', '*.jsonl'), ('All files', '*.*')],
        )
        if path:
            self.output.set(path)

    def _start(self) -> None:
        image_dir = self.image_dir.get().strip()
        output = self.output.get().strip()
        if not image_dir:
            messagebox.showerror('缺少图片目录', '请选择图片目录。')
            return
        if not output:
            messagebox.showerror('缺少输出文件', '请选择输出 JSONL 文件。')
            return
        if not Path(image_dir).exists():
            messagebox.showerror('图片目录不存在', image_dir)
            return

        self.args.image_dir = image_dir
        self.args.output = output
        self.args.recursive = bool(self.recursive.get())
        for child in self.root.winfo_children():
            child.destroy()
        try:
            MeterAnnotator(self.root, self.args)
        except Exception as e:
            messagebox.showerror('启动失败', str(e))
            raise


def main() -> None:
    args = parse_args()
    root = tk.Tk()
    if args.image_dir and args.output:
        try:
            MeterAnnotator(root, args)
        except Exception as e:
            messagebox.showerror('启动失败', str(e))
            raise
    else:
        PathSelector(root, args)
    root.mainloop()


if __name__ == '__main__':
    main()
