# subx

Tool CLI + GUI untuk mengambil subtitle dari video — baik **softsub** (embedded stream) maupun **hardsub** (teks yang terbakar di gambar, via OCR) — plus fitur translate.

## Kebutuhan

- Python 3.10+
- [FFmpeg](https://ffmpeg.org/) (`ffmpeg` dan `ffprobe` harus ada di PATH)

```bash
pip install -r requirements.txt
```

## Penggunaan

### Lihat stream subtitle embedded

```bash
python subx.py list video.mkv
```

### Ekstrak softsub

```bash
python subx.py soft video.mkv                # stream pertama → video.srt
python subx.py soft video.mkv -s 2 -o out.srt
```

Stream teks (SRT/ASS/mov_text/WebVTT) dikonversi ke `.srt`. Stream bitmap (PGS/DVD sub) disalin mentah ke `.sup`/`.sub` — OCR dulu (mis. Subtitle Edit) kalau butuh teks.

### OCR hardsub

```bash
python subx.py hard video.mp4                # → video.srt
```

Cara kerja: sampling frame via ffmpeg, crop bagian bawah, deteksi perubahan antar-frame, OCR (RapidOCR) hanya saat gambar berubah, lalu gabung jadi cue SRT.

Opsi tuning:

| Flag | Default | Fungsi |
|---|---|---|
| `--fps` | `2` | laju sampling; naikkan untuk timing lebih presisi (lebih lambat) |
| `--crop` | `0.35` | fraksi bawah frame yang dipindai; sesuaikan jika posisi subtitle beda |
| `--diff-thresh` | `0.003` | fraksi piksel berubah untuk memicu OCR ulang; turunkan jika ada subtitle tertelan |
| `--min-score` | `0.5` | ambang confidence OCR |

### Translate SRT

```bash
python subx.py translate subs.srt --to id    # → subs.id.srt
python subx.py translate subs.srt --to en --source id
```

Pakai Google Translate (gratis, via deep-translator). Kode bahasa: `id`, `en`, `ja`, dst.

### GUI

```bash
python subx.py gui
```

Pilih video → tombol **List Streams** / **Extract Softsub** / **OCR Hardsub**. Centang *Translate hasil ke:* untuk auto-translate hasil ekstraksi. Tombol **Translate .srt...** untuk translate file SRT yang sudah ada.

### Selftest

```bash
python subx.py selftest
```

## Batasan

- OCR kadang menggabungkan kata (`Selamatmalam`) — batasan model; hasil translate biasanya tetap benar.
- Default `--fps 2` berarti granularitas timing ±0.5 detik.
- Hardsub dengan posisi di luar 35% bawah frame butuh `--crop` lebih besar.
- Translate memakai endpoint gratis Google — file sangat besar bisa kena rate limit.
