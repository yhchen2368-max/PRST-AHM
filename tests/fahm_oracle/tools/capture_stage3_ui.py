"""Capture the Stage 3 PRST FAHM client area at FAHM's frozen size."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import tkinter as tk

from PIL import ImageGrab

from PRSTCore.hm.APP.fahm_app import APP_SIZE, APP_TITLE, FahmApp


DEFAULT_OUTPUT = (Path(__file__).resolve().parents[2] / 'fixtures' /
                  'fahm_oracle' / 'v1' / 'stage3')


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def capture(output_dir: Path) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    image_path = output_dir / 'prst_initial_915x736.png'
    manifest_path = output_dir / 'screenshot.json'

    root = tk.Tk()
    try:
        app = FahmApp(root, startup_delay_ms=None,
                      render_plot_canvases=False)
        app._finish_startup()
        root.geometry('%dx%d+80+80' % APP_SIZE)
        root.deiconify()
        root.lift()
        root.update_idletasks()
        root.update()

        x = root.winfo_rootx()
        y = root.winfo_rooty()
        width = root.winfo_width()
        height = root.winfo_height()
        if (width, height) != APP_SIZE:
            raise RuntimeError(
                f'expected client size {APP_SIZE}, got {(width, height)}')
        image = ImageGrab.grab(bbox=(x, y, x + width, y + height),
                               all_screens=True)
        image.save(image_path, format='PNG', optimize=False)
    finally:
        root.destroy()

    manifest = {
        'classification': 'PARITY',
        'height': APP_SIZE[1],
        'image': image_path.name,
        'parity_scope': ('FAHM client-area dimensions, tab/control layout and '
                         'initial post-splash state; native toolkit raster '
                         'pixels are platform-specific'),
        'sha256': _sha256(image_path),
        'source': 'MRST/dev/APP/FAHM.m:createComponents/startupFcn/*SizeChanged',
        'title': APP_TITLE,
        'width': APP_SIZE[0],
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + '\n',
        encoding='utf-8', newline='\n')
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    print(json.dumps(capture(args.output), ensure_ascii=False, indent=2,
                     sort_keys=True))


if __name__ == '__main__':
    main()
