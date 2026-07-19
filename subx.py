#!/usr/bin/env python3
"""subx - extract subtitles from video.

  python subx.py list      video.mkv             # list embedded subtitle streams
  python subx.py soft      video.mkv [-s N] [-o out.srt]
  python subx.py hard      video.mkv [-o out.srt] [--fps 2] [--crop 0.35]
  python subx.py translate subs.srt --to id
  python subx.py gui
  python subx.py selftest
"""
import argparse
import difflib
import json
import re
import subprocess
import sys
from pathlib import Path

TEXT_CODECS = {"subrip", "srt", "ass", "ssa", "mov_text", "webvtt", "text"}
BITMAP_EXT = {"hdmv_pgs_subtitle": ".sup", "dvd_subtitle": ".sub", "dvb_subtitle": ".sub"}


def ffprobe(video, select=None):
    cmd = ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_streams"]
    if select:
        cmd += ["-select_streams", select]
    cmd.append(str(video))
    out = subprocess.run(cmd, capture_output=True, text=True, check=True).stdout
    return json.loads(out).get("streams", [])


def cmd_list(args):
    streams = ffprobe(args.video, "s")
    if not streams:
        print("no embedded subtitle streams (video is hardsub-only or has none). try: subx.py hard")
        return
    for s in streams:
        tags = s.get("tags", {})
        kind = "text" if s["codec_name"] in TEXT_CODECS else "bitmap"
        print(f'#{s["index"]}  codec={s["codec_name"]} ({kind})  '
              f'lang={tags.get("language", "?")}  title={tags.get("title", "")}')


def cmd_soft(args):
    streams = ffprobe(args.video, "s")
    if not streams:
        sys.exit("no embedded subtitle streams. use: subx.py hard")
    if args.stream is not None:
        streams = [s for s in streams if s["index"] == args.stream]
        if not streams:
            sys.exit(f"no subtitle stream with index {args.stream} (see: subx.py list)")
    s = streams[0]
    codec = s["codec_name"]
    base = Path(args.video).with_suffix("")
    if codec in TEXT_CODECS:
        out = args.output or f"{base}.srt"
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(args.video),
                        "-map", f'0:{s["index"]}', str(out)], check=True)
    else:
        # bitmap sub: no text to convert, copy raw stream out
        out = args.output or f'{base}{BITMAP_EXT.get(codec, ".mks")}'
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(args.video),
                        "-map", f'0:{s["index"]}', "-c", "copy", str(out)], check=True)
        print(f"stream is bitmap ({codec}), copied raw to {out}. "
              f"for text, OCR it (e.g. SubtitleEdit) or use: subx.py hard")
        return
    print(f"wrote {out}")


# ---------- hardsub ----------

def video_dims(video):
    v = ffprobe(video, "v:0")
    if not v:
        sys.exit("no video stream found")
    return int(v[0]["width"]), int(v[0]["height"])


def iter_frames(video, fps, crop_ratio):
    """Yield (t_seconds, gray_ndarray) of the bottom crop_ratio of each sampled frame."""
    import numpy as np
    w, h = video_dims(video)
    ch = int(h * crop_ratio)
    vf = f"fps={fps},crop={w}:{ch}:0:{h - ch},format=gray"
    proc = subprocess.Popen(
        ["ffmpeg", "-v", "error", "-i", str(video), "-vf", vf,
         "-f", "rawvideo", "-pix_fmt", "gray", "-"],
        stdout=subprocess.PIPE)
    size = w * ch
    i = 0
    while True:
        buf = proc.stdout.read(size)
        if len(buf) < size:
            break
        yield i / fps, np.frombuffer(buf, np.uint8).reshape(ch, w)
        i += 1
    proc.wait()


def similar(a, b):
    return difflib.SequenceMatcher(None, a, b).ratio()


def build_cues(samples, min_gap=0.0, sim_thresh=0.8):
    """samples: iterable of (t, text) for every sampled frame ('' = no text).
    Returns [(start, end, text)]. Fuzzy-merges consecutive similar texts."""
    cues = []
    cur, start, last_t = None, 0.0, 0.0
    for t, text in samples:
        text = text.strip()
        if cur is not None and text and similar(text, cur) >= sim_thresh:
            last_t = t
            continue
        if cur is not None:
            cues.append((start, t, cur))
            cur = None
        if text:
            cur, start = text, t
        last_t = t
    if cur is not None:
        cues.append((start, last_t, cur))
    return [c for c in cues if c[1] - c[0] > min_gap]


def srt_ts(t):
    ms = int(round(t * 1000))
    h, ms = divmod(ms, 3600000)
    m, ms = divmod(ms, 60000)
    s, ms = divmod(ms, 1000)
    return f"{h:02}:{m:02}:{s:02},{ms:03}"


def write_srt(cues, path):
    with open(path, "w", encoding="utf-8") as f:
        for i, (a, b, text) in enumerate(cues, 1):
            f.write(f"{i}\n{srt_ts(a)} --> {srt_ts(b)}\n{text}\n\n")


def cmd_hard(args):
    import numpy as np
    from rapidocr_onnxruntime import RapidOCR
    ocr = RapidOCR()

    def ocr_text(frame):
        result, _ = ocr(np.stack([frame] * 3, axis=-1))
        if not result:
            return ""
        lines = [(box[0][1], txt) for box, txt, score in result if float(score) >= args.min_score]
        return "\n".join(t for _, t in sorted(lines))

    def samples():
        prev, prev_text = None, ""
        n = 0
        for t, frame in iter_frames(args.video, args.fps, args.crop):
            # ponytail: changed-pixel-fraction gate skips OCR on static frames; lower --diff-thresh if subs get swallowed
            if prev is not None and float(
                    np.mean(np.abs(frame.astype(np.int16) - prev) > 25)) < args.diff_thresh:
                yield t, prev_text
            else:
                prev_text = ocr_text(frame)
                yield t, prev_text
            prev = frame
            n += 1
            if n % (args.fps * 60) == 0:
                print(f"  {int(t // 60)} min processed...", file=sys.stderr)

    cues = build_cues(samples(), min_gap=0.2)
    out = args.output or str(Path(args.video).with_suffix("")) + ".srt"
    write_srt(cues, out)
    print(f"wrote {len(cues)} cues to {out}")


# ---------- translate ----------

def parse_srt(path):
    """Returns list of (index_line, ts_line, text) blocks."""
    content = Path(path).read_text(encoding="utf-8-sig")
    blocks = []
    for b in re.split(r"\n\s*\n", content.strip()):
        lines = b.splitlines()
        if len(lines) >= 3 and "-->" in lines[1]:
            blocks.append((lines[0], lines[1], "\n".join(lines[2:])))
    return blocks


def cmd_translate(args):
    from deep_translator import GoogleTranslator
    blocks = parse_srt(args.srt)
    if not blocks:
        sys.exit(f"no cues parsed from {args.srt}")
    tr = GoogleTranslator(source=args.source, target=args.to)
    texts = tr.translate_batch([t.replace("\n", " ") for _, _, t in blocks])
    out = args.output or str(Path(args.srt).with_suffix("")) + f".{args.to}.srt"
    with open(out, "w", encoding="utf-8") as f:
        for (idx, ts, _), text in zip(blocks, texts):
            f.write(f"{idx}\n{ts}\n{text or ''}\n\n")
    print(f"wrote {out}")


# ---------- GUI ----------

def cmd_gui(args):
    import queue
    import threading
    import tkinter as tk
    from tkinter import filedialog, ttk

    root = tk.Tk()
    root.title("subx - subtitle extractor")
    root.geometry("640x480")
    frm = ttk.Frame(root, padding=8)
    frm.pack(fill="both", expand=True)

    video_var = tk.StringVar()
    lang_var = tk.StringVar(value="id")
    translate_var = tk.BooleanVar(value=False)
    q = queue.Queue()

    row = ttk.Frame(frm)
    row.pack(fill="x")
    ttk.Label(row, text="Video:").pack(side="left")
    ttk.Entry(row, textvariable=video_var).pack(side="left", fill="x", expand=True, padx=4)
    ttk.Button(row, text="Browse...", command=lambda: video_var.set(
        filedialog.askopenfilename(filetypes=[("Video", "*.mp4 *.mkv *.avi *.ts *.webm"), ("All", "*.*")])
        or video_var.get())).pack(side="left")

    row2 = ttk.Frame(frm)
    row2.pack(fill="x", pady=6)
    ttk.Checkbutton(row2, text="Translate hasil ke:", variable=translate_var).pack(side="left")
    ttk.Entry(row2, textvariable=lang_var, width=6).pack(side="left", padx=4)
    ttk.Label(row2, text="(kode bahasa: id, en, ja, ...)").pack(side="left")

    log = tk.Text(frm, state="disabled", wrap="word")
    log.pack(fill="both", expand=True, pady=6)

    buttons = []

    def logln(msg):
        log.configure(state="normal")
        log.insert("end", msg + "\n")
        log.see("end")
        log.configure(state="disabled")

    def run(cli_args, then_translate=False):
        video = video_var.get()
        if not video:
            logln("!! pilih file video dulu")
            return
        srt_out = str(Path(video).with_suffix("")) + ".srt"

        def worker():
            try:
                jobs = [cli_args + [video]]
                if then_translate and translate_var.get():
                    jobs.append(["translate", srt_out, "--to", lang_var.get()])
                for job in jobs:
                    q.put("$ subx " + " ".join(job))
                    p = subprocess.Popen([sys.executable, __file__] + job,
                                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                         encoding="utf-8", errors="replace")
                    for line in p.stdout:
                        q.put(line.rstrip())
                    if p.wait() != 0:
                        q.put(f"!! exit code {p.returncode}")
                        break
            finally:
                q.put(("__done__",))

        for b in buttons:
            b.configure(state="disabled")
        threading.Thread(target=worker, daemon=True).start()

    def poll():
        try:
            while True:
                item = q.get_nowait()
                if item == ("__done__",):
                    for b in buttons:
                        b.configure(state="normal")
                else:
                    logln(item)
        except queue.Empty:
            pass
        root.after(100, poll)

    row3 = ttk.Frame(frm)
    row3.pack(fill="x")
    for label, cli, tr in [("List Streams", ["list"], False),
                           ("Extract Softsub", ["soft"], True),
                           ("OCR Hardsub", ["hard"], True)]:
        b = ttk.Button(row3, text=label, command=lambda c=cli, t=tr: run(c, t))
        b.pack(side="left", padx=2)
        buttons.append(b)
    bt = ttk.Button(row3, text="Translate .srt...", command=lambda: (
        lambda f: f and run_translate(f))(filedialog.askopenfilename(filetypes=[("SRT", "*.srt")])))
    bt.pack(side="left", padx=2)
    buttons.append(bt)

    def run_translate(srt):
        def worker():
            try:
                q.put(f"$ subx translate {srt} --to {lang_var.get()}")
                p = subprocess.Popen([sys.executable, __file__, "translate", srt, "--to", lang_var.get()],
                                     stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                     encoding="utf-8", errors="replace")
                for line in p.stdout:
                    q.put(line.rstrip())
                p.wait()
            finally:
                q.put(("__done__",))
        for b in buttons:
            b.configure(state="disabled")
        threading.Thread(target=worker, daemon=True).start()

    poll()
    root.mainloop()


def cmd_selftest(args):
    assert srt_ts(0) == "00:00:00,000"
    assert srt_ts(3661.5) == "01:01:01,500"
    cues = build_cues([
        (0.0, ""), (0.5, "Hello world"), (1.0, "Hello world"), (1.5, "Helo world"),
        (2.0, ""), (2.5, "Bye"), (3.0, "Bye"),
    ])
    assert cues == [(0.5, 2.0, "Hello world"), (2.5, 3.0, "Bye")], cues
    import tempfile, os
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "t.srt")
        write_srt(cues, p)
        blocks = parse_srt(p)
        assert [b[2] for b in blocks] == ["Hello world", "Bye"], blocks
        assert blocks[0][1] == "00:00:00,500 --> 00:00:02,000", blocks[0]
    print("selftest OK")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    lp = sub.add_parser("list", help="list embedded subtitle streams")
    lp.add_argument("video")
    lp.set_defaults(fn=cmd_list)

    spp = sub.add_parser("soft", help="extract embedded (softsub) subtitle")
    spp.add_argument("video")
    spp.add_argument("-s", "--stream", type=int, help="stream index from 'list' (default: first)")
    spp.add_argument("-o", "--output")
    spp.set_defaults(fn=cmd_soft)

    hp = sub.add_parser("hard", help="OCR burned-in (hardsub) subtitle to .srt")
    hp.add_argument("video")
    hp.add_argument("-o", "--output")
    hp.add_argument("--fps", type=float, default=2, help="sample rate (default 2)")
    hp.add_argument("--crop", type=float, default=0.35, help="bottom fraction to scan (default 0.35)")
    hp.add_argument("--diff-thresh", type=float, default=0.003,
                    help="fraction of pixels that must change to re-OCR (default 0.003)")
    hp.add_argument("--min-score", type=float, default=0.5, help="OCR confidence cutoff (default 0.5)")
    hp.set_defaults(fn=cmd_hard)

    tp = sub.add_parser("translate", help="translate an .srt file")
    tp.add_argument("srt")
    tp.add_argument("--to", default="id", help="target language code (default id)")
    tp.add_argument("--source", default="auto")
    tp.add_argument("-o", "--output")
    tp.set_defaults(fn=cmd_translate)

    gp = sub.add_parser("gui", help="open the GUI")
    gp.set_defaults(fn=cmd_gui)

    st = sub.add_parser("selftest", help="run built-in checks")
    st.set_defaults(fn=cmd_selftest)

    args = p.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
