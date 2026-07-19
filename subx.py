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


_ISO3_TO_ISO2 = {
    "eng": "en", "ind": "id", "may": "ms", "msa": "ms", "jpn": "ja", "kor": "ko",
    "chi": "zh", "zho": "zh", "spa": "es", "fre": "fr", "fra": "fr", "ger": "de",
    "deu": "de", "por": "pt", "rus": "ru", "ara": "ar", "hin": "hi", "tha": "th",
    "vie": "vi", "ita": "it", "dut": "nl", "nld": "nl", "tur": "tr", "tgl": "tl",
    "fil": "tl",
}

def sanitize_lang(lang, fallback="orig"):
    lang = (lang or "").strip().lower()
    lang = _ISO3_TO_ISO2.get(lang, lang)
    if re.fullmatch(r"[a-z]{2,8}", lang) and lang != "und":
        return lang
    return fallback


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


def _extract_one(video, s, out):
    """Extract one subtitle stream. Text → out (converted); bitmap → raw copy. Returns out or None."""
    codec = s["codec_name"]
    if codec in TEXT_CODECS:
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(video),
                        "-map", f'0:{s["index"]}', str(out)], check=True)
        try:
            text = Path(out).read_text(encoding="utf-8", errors="replace")
            if "-->" not in text:
                Path(out).unlink(missing_ok=True)
                print(f'stream #{s["index"]} holds no cues, skipped.')
                return None
        except Exception:
            pass
        return out
    out = str(Path(out).with_suffix(BITMAP_EXT.get(codec, ".mks")))
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(video),
                    "-map", f'0:{s["index"]}', "-c", "copy", str(out)], check=True)
    print(f'stream #{s["index"]} is bitmap ({codec}), copied raw to {out}. '
          f"for text, OCR it (e.g. SubtitleEdit) or use: subx.py hard")
    return None


def cmd_soft(args):
    streams = ffprobe(args.video, "s")
    if not streams:
        sys.exit("no embedded subtitle streams. use: subx.py hard")
    base = Path(args.video).with_suffix("")
    if args.all:
        wrote = 0
        for s in streams:
            raw_lang = s.get("tags", {}).get("language")
            lang = sanitize_lang(raw_lang, fallback=f's{s["index"]}')
            if _extract_one(args.video, s, f"{base}.{lang}.srt"):
                print(f"wrote {base}.{lang}.srt")
                wrote += 1
        if not wrote:
            sys.exit("no text subtitle streams extracted (bitmap-only or empty?)")
        return
    if args.stream is not None:
        streams = [s for s in streams if s["index"] == args.stream]
        if not streams:
            sys.exit(f"no subtitle stream with index {args.stream} (see: subx.py list)")
    out = _extract_one(args.video, streams[0], args.output or f"{base}.srt")
    if out:
        print(f"wrote {out}")


# ---------- hardsub ----------

def video_dims(video):
    v = ffprobe(video, "v:0")
    if not v:
        sys.exit("no video stream found")
    return int(v[0]["width"]), int(v[0]["height"])

def video_duration(video):
    cmd = ["ffprobe", "-v", "error", "-show_entries", "format=duration",
           "-of", "default=noprint_wrappers=1:nokey=1", str(video)]
    try:
        return float(subprocess.run(cmd, capture_output=True, text=True, check=True).stdout.strip())
    except (ValueError, subprocess.CalledProcessError):
        return 0.0


def iter_frames(video, fps, crop_ratio, start=None, duration=None):
    """Yield (t_seconds, gray_ndarray) of the bottom crop_ratio of each sampled frame."""
    import numpy as np
    w, h = video_dims(video)
    ch = int(h * crop_ratio)
    vf = f"fps={fps},crop={w}:{ch}:0:{h - ch}"
    if w > 1280:  # ponytail: OCR det resizes to ~736px internally anyway; smaller pipe = less CPU
        ch = int(ch * 1280 / w) & ~1
        w = 1280
        vf += f",scale={w}:{ch}"
    vf += ",format=gray"
    
    cmd = ["ffmpeg", "-nostdin", "-v", "error", "-hwaccel", "auto"]
    if start is not None:
        cmd.extend(["-ss", str(start)])
    if duration is not None:
        cmd.extend(["-t", str(duration)])
    cmd.extend(["-i", str(video), "-vf", vf, "-f", "rawvideo", "-pix_fmt", "gray", "-"])
    
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stdin=subprocess.DEVNULL)
    size = w * ch
    i = 0
    base_t = float(start) if start is not None else 0.0
    while True:
        buf = proc.stdout.read(size)
        if len(buf) < size:
            break
        yield base_t + (i / fps), np.frombuffer(buf, np.uint8).reshape(ch, w)
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


_worker_ocr = None
_use_gpu = None  # tri-state: None = not checked, True/False = cached result


def _has_gpu():
    """Check if CUDA GPU is available via onnxruntime."""
    global _use_gpu
    if _use_gpu is None:
        try:
            import onnxruntime as ort
            _use_gpu = "CUDAExecutionProvider" in ort.get_available_providers()
        except Exception:
            _use_gpu = False
        if _use_gpu:
            print("GPU (CUDA) detected — using parallel multiprocessing GPU chunks", file=sys.stderr)
        else:
            print("No GPU detected — using CPU multiprocessing", file=sys.stderr)
    return _use_gpu


def _init_ocr():
    """Initialise the OCR engine in a worker (or main) process."""
    global _worker_ocr
    from rapidocr_onnxruntime import RapidOCR
    if _has_gpu():
        _worker_ocr = RapidOCR(
            det_use_cuda=True, cls_use_cuda=True, rec_use_cuda=True,
        )
        try:
            provs = _worker_ocr.text_det.infer.session.get_providers()
            print(f"OCR session providers: {provs}", file=sys.stderr)
            if "CUDAExecutionProvider" not in provs:
                print("WARNING: session fell back to CPU — check onnxruntime-gpu/cuDNN install",
                      file=sys.stderr)
        except AttributeError:
            pass
    else:
        _worker_ocr = RapidOCR()


def _do_ocr(args):
    frame, min_score = args
    return _do_ocr_local(frame, min_score)


def _do_ocr_local(frame, min_score):
    import numpy as np
    # use_cls=False: subtitles are never rotated, skip the angle-classifier pass
    result, _ = _worker_ocr(np.stack([frame] * 3, axis=-1), use_cls=False)
    if not result:
        return ""
    lines = [(box[0][1], txt) for box, txt, score in result if float(score) >= min_score]
    return "\n".join(t for _, t in sorted(lines))


def cmd_hard(args):
    import numpy as np
    import collections
    import multiprocessing
    import time
    from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

    gpu = _has_gpu()
    t0 = time.time()
    base = args.start or 0.0

    def resolve_samples_gpu():
        """GPU path: one shared ONNX session, 2 OCR threads (session.run is
        thread-safe) so CPU pre/post-processing overlaps GPU inference.
        Single process — multiprocessing here caused CUDA OOM on Colab."""
        _init_ocr()
        pending = collections.deque()
        prev, prev_text = None, ""
        n = 0
        # ponytail: 2 workers matches Colab's 2 CPU cores; bump if host has more
        with ThreadPoolExecutor(max_workers=2) as pool:
            for t, frame in iter_frames(args.video, args.fps, args.crop, args.start, args.duration):
                if prev is not None and float(
                        np.mean(np.abs(frame.astype(np.int16) - prev) > 25)) < args.diff_thresh:
                    fut = None
                else:
                    fut = pool.submit(_do_ocr_local, frame, args.min_score)

                pending.append((t, fut))
                prev = frame
                n += 1
                if n % (args.fps * 60) == 0:
                    print(f"  {int(t // 60)} min processed "
                          f"({(t - base) / max(time.time() - t0, 1e-9):.1f}x realtime)...",
                          file=sys.stderr)

                while len(pending) > args.fps * 30:
                    pt, pfut = pending.popleft()
                    if pfut is not None:
                        prev_text = pfut.result()
                    yield pt, prev_text

            for pt, pfut in pending:
                if pfut is not None:
                    prev_text = pfut.result()
                yield pt, prev_text

    def resolve_samples_cpu():
        """Multi-process path: fan out OCR across CPU cores."""
        pending = collections.deque()
        prev, prev_text = None, ""
        n = 0

        n_workers = args.workers or max(1, multiprocessing.cpu_count())
        with ProcessPoolExecutor(max_workers=n_workers, initializer=_init_ocr) as pool:
            for t, frame in iter_frames(args.video, args.fps, args.crop, args.start, args.duration):
                if prev is not None and float(
                        np.mean(np.abs(frame.astype(np.int16) - prev) > 25)) < args.diff_thresh:
                    fut = None
                else:
                    fut = pool.submit(_do_ocr, (frame, args.min_score))

                pending.append((t, fut))
                prev = frame
                n += 1

                if n % (args.fps * 60) == 0:
                    print(f"  {int(t // 60)} min processed "
                          f"({(t - base) / max(time.time() - t0, 1e-9):.1f}x realtime)...",
                          file=sys.stderr)

                while len(pending) > args.fps * 60:
                    pt, pfut = pending.popleft()
                    if pfut is not None:
                        prev_text = pfut.result()
                    yield pt, prev_text

            for pt, pfut in pending:
                if pfut is not None:
                    prev_text = pfut.result()
                yield pt, prev_text

    samples = resolve_samples_gpu() if gpu else resolve_samples_cpu()
    cues = build_cues(samples, min_gap=0.2)
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

        def run_job(job):
            q.put("$ subx " + " ".join(job))
            p = subprocess.Popen([sys.executable, __file__] + job,
                                 stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                 encoding="utf-8", errors="replace")
            for line in p.stdout:
                q.put(line.rstrip())
            if p.wait() != 0:
                q.put(f"!! exit code {p.returncode}")
                return False
            return True

        def worker():
            try:
                ok = run_job(cli_args + [video])
                # Translate only if the extraction actually produced the .srt —
                # a bitmap softsub writes .sup/.sub instead and can't be translated.
                if ok and then_translate and translate_var.get():
                    if Path(srt_out).exists():
                        run_job(["translate", srt_out, "--to", lang_var.get()])
                    else:
                        q.put(f"!! {srt_out} not found — nothing to translate (bitmap subtitle?)")
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
    spp.add_argument("-a", "--all", action="store_true",
                     help="extract EVERY text stream to <video>.<lang>.srt")
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
    hp.add_argument("--workers", type=int, default=0,
                    help="OCR worker processes for CPU path (default: all cores; lower if low RAM)")
    hp.add_argument("--start", type=float, help="start offset in seconds (split work across sessions)")
    hp.add_argument("--duration", type=float, help="seconds to process from --start")
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
