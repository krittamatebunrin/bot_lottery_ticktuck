from __future__ import annotations

import argparse
import json
import os
import random
import time
from collections import deque
from contextlib import ExitStack
from datetime import datetime, timedelta, timezone
from functools import lru_cache
from pathlib import Path

import requests
from PIL import Image, ImageDraw, ImageFont, ImageOps, features

ICT = timezone(timedelta(hours=7))
BASE_DIR = Path(__file__).resolve().parent
FONT_CACHE: dict[tuple[str, int], ImageFont.FreeTypeFont] = {}
CAT_CACHE: dict[str, Image.Image] = {}
MAX_OUTPUT_SIDE = 1600
SENT_STATE_FILE = "sent_state.json"


# --------------------------------------------------------------------------- #
# File / font helpers
# --------------------------------------------------------------------------- #
def resolve_file(directory: Path, filename: str) -> Path:
    """Find `filename` in `directory`, tolerating case and extension differences."""
    direct = directory / filename
    if direct.exists():
        return direct
    items = sorted(p for p in directory.iterdir() if p.is_file())
    target_lower = filename.lower()
    target_stem = Path(filename).stem.lower()
    for item in items:
        if item.name.lower() == target_lower:
            return item
    for item in items:
        if item.stem.lower() == target_stem:
            return item
    raise FileNotFoundError(f"Missing file '{filename}' in {directory}")


def get_font(font_filename: str, size_px: int) -> ImageFont.FreeTypeFont:
    key = (font_filename, size_px)
    if key not in FONT_CACHE:
        font_path = str(resolve_file(BASE_DIR / "fonts", font_filename))
        kwargs = {"layout_engine": ImageFont.Layout.RAQM} if features.check("raqm") else {}
        FONT_CACHE[key] = ImageFont.truetype(font_path, size=size_px, **kwargs)
    return FONT_CACHE[key]


def open_rgba(path: Path) -> Image.Image:
    with Image.open(path) as src:
        return src.convert("RGBA")


# --------------------------------------------------------------------------- #
# Cat cutout
# --------------------------------------------------------------------------- #
def crop_to_alpha(img: Image.Image) -> Image.Image:
    """Crop to the non-transparent area (alpha channel only, any Pillow version)."""
    bbox = img.getchannel("A").getbbox()
    return img.crop(bbox) if bbox else img


def _clear_border_background(px, w: int, h: int) -> None:
    """Flood-fill from the image border, making near-white / transparent pixels transparent."""

    def is_bg(x: int, y: int) -> bool:
        r, g, b, a = px[x, y]
        return a < 10 or (r >= 225 and g >= 225 and b >= 225)

    visited = bytearray(w * h)
    q: deque = deque()
    seeds = [(x, y) for x in range(w) for y in (0, h - 1)]
    seeds += [(x, y) for y in range(h) for x in (0, w - 1)]
    for x, y in seeds:
        idx = y * w + x
        if not visited[idx] and is_bg(x, y):
            visited[idx] = 1
            q.append((x, y))

    while q:
        cx, cy = q.popleft()
        r, g, b, _ = px[cx, cy]
        px[cx, cy] = (r, g, b, 0)
        for nx, ny in ((cx - 1, cy), (cx + 1, cy), (cx, cy - 1), (cx, cy + 1)):
            if 0 <= nx < w and 0 <= ny < h:
                nidx = ny * w + nx
                if not visited[nidx] and is_bg(nx, ny):
                    visited[nidx] = 1
                    q.append((nx, ny))


def _remove_small_specks(px, w: int, h: int) -> None:
    """Drop tiny disconnected blobs that are not near the main subject."""
    comp_seen = bytearray(w * h)
    components: list[tuple[list[tuple[int, int]], tuple[int, int, int, int]]] = []
    for y in range(h):
        for x in range(w):
            idx = y * w + x
            if px[x, y][3] > 0 and not comp_seen[idx]:
                comp_seen[idx] = 1
                cq = deque([(x, y)])
                pts: list[tuple[int, int]] = []
                x0 = x1 = x
                y0 = y1 = y
                while cq:
                    cx, cy = cq.popleft()
                    pts.append((cx, cy))
                    x0, x1 = min(x0, cx), max(x1, cx)
                    y0, y1 = min(y0, cy), max(y1, cy)
                    for dx in (-1, 0, 1):
                        for dy in (-1, 0, 1):
                            if dx == 0 and dy == 0:
                                continue
                            nx, ny = cx + dx, cy + dy
                            if 0 <= nx < w and 0 <= ny < h:
                                nidx = ny * w + nx
                                if px[nx, ny][3] > 0 and not comp_seen[nidx]:
                                    comp_seen[nidx] = 1
                                    cq.append((nx, ny))
                components.append((pts, (x0, y0, x1, y1)))

    if not components:
        return
    main_pts, (mx0, my0, mx1, my1) = max(components, key=lambda c: len(c[0]))
    min_keep_area = max(10, int(len(main_pts) * 0.03))
    pad = 20
    for pts, (bx0, by0, bx1, by1) in components:
        near_main = not (
            bx1 < mx0 - pad or bx0 > mx1 + pad or by1 < my0 - pad or by0 > my1 + pad
        )
        if len(pts) < min_keep_area and not near_main:
            for cx, cy in pts:
                r, g, b, _ = px[cx, cy]
                px[cx, cy] = (r, g, b, 0)


def _feather_light_edges(px, w: int, h: int) -> None:
    """Soften light fringe pixels that touch the transparent background."""
    updates: list[tuple[int, int, int]] = []
    for y in range(h):
        for x in range(w):
            r, g, b, a = px[x, y]
            if a == 0:
                continue
            touches_bg = any(
                not (0 <= nx < w and 0 <= ny < h) or px[nx, ny][3] == 0
                for nx, ny in ((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1))
            )
            if touches_bg:
                m = min(r, g, b)
                if m > 150:
                    new_a = max(0, min(255, round((230 - m) / 80.0 * 255)))
                    updates.append((x, y, min(a, new_a)))
    for x, y, new_a in updates:
        r, g, b, _ = px[x, y]
        px[x, y] = (r, g, b, new_a)


def load_cutout_cat(filename: str) -> Image.Image:
    if filename in CAT_CACHE:
        return CAT_CACHE[filename]

    img = open_rgba(resolve_file(BASE_DIR / "assets", filename))
    img = ImageOps.contain(img, (480, 480), method=Image.Resampling.LANCZOS)
    w, h = img.size
    px = img.load()

    corner_alphas = [px[0, 0][3], px[w - 1, 0][3], px[0, h - 1][3], px[w - 1, h - 1][3]]
    if not all(a < 10 for a in corner_alphas):
        _clear_border_background(px, w, h)
        _remove_small_specks(px, w, h)
        _feather_light_edges(px, w, h)

    result = crop_to_alpha(img)
    CAT_CACHE[filename] = result
    return result


# --------------------------------------------------------------------------- #
# Drawing helpers
# --------------------------------------------------------------------------- #
def hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
    h = hex_color.lstrip("#")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


@lru_cache(maxsize=128)
def _cached_vertical_gradient(width: int, height: int, stops: tuple[str, ...]) -> Image.Image:
    rgb_stops = [hex_to_rgb(c) for c in stops]
    n = len(rgb_stops) - 1
    colors: list[tuple[int, int, int, int]] = []
    for y in range(height):
        t = y / max(height - 1, 1)
        seg = min(int(t * n), n - 1) if n > 0 else 0
        local_t = (t * n) - seg if n > 0 else 0.0
        c1, c2 = rgb_stops[seg], rgb_stops[min(seg + 1, n)]
        colors.append(
            (
                int(c1[0] + (c2[0] - c1[0]) * local_t),
                int(c1[1] + (c2[1] - c1[1]) * local_t),
                int(c1[2] + (c2[2] - c1[2]) * local_t),
                255,
            )
        )
    column = Image.new("RGBA", (1, height))
    column.putdata(colors)
    return column.resize((width, height), Image.Resampling.NEAREST)


def build_vertical_gradient(width: int, height: int, stops: list[str]) -> Image.Image:
    return _cached_vertical_gradient(width, height, tuple(stops))


def draw_styled_text(
    canvas: Image.Image,
    text: str,
    x_norm: float,
    y_norm: float,
    font_name: str,
    size_norm: float,
    fill: str,
    anchor: str = "mm",
    stroke_color: str | None = None,
    stroke_width_norm: float = 0.0,
    gradient: list[str] | None = None,
) -> None:
    w, h = canvas.size
    font = get_font(font_name, max(12, round(size_norm * h)))
    sw = round(stroke_width_norm * h) if stroke_color and stroke_width_norm > 0 else 0
    x, y = round(x_norm * w), round(y_norm * h)
    draw = ImageDraw.Draw(canvas)

    if not gradient:
        draw.text(
            (x, y),
            text,
            font=font,
            fill=fill,
            anchor=anchor,
            stroke_width=sw,
            stroke_fill=stroke_color if sw > 0 else None,
        )
        return

    if sw > 0 and stroke_color:
        draw.text(
            (x, y),
            text,
            font=font,
            fill=stroke_color,
            anchor=anchor,
            stroke_width=sw,
            stroke_fill=stroke_color,
        )

    bx0, by0, bx1, by1 = draw.textbbox((x, y), text, font=font, anchor=anchor, stroke_width=0)
    pad = 2
    bx0, by0 = max(0, bx0 - pad), max(0, by0 - pad)
    bx1, by1 = min(w, bx1 + pad), min(h, by1 + pad)
    bw, bh = max(1, bx1 - bx0), max(1, by1 - by0)

    mask = Image.new("L", (bw, bh), 0)
    ImageDraw.Draw(mask).text((x - bx0, y - by0), text, font=font, fill=255, anchor=anchor)
    canvas.paste(build_vertical_gradient(bw, bh, gradient), (bx0, by0), mask)


def draw_cfg_text(
    canvas: Image.Image,
    text: str,
    cfg: dict,
    x: float | None = None,
    y: float | None = None,
) -> None:
    """Draw text styled by a layout config block; x/y override the block's own position."""
    draw_styled_text(
        canvas,
        text=text,
        x_norm=cfg["x"] if x is None else x,
        y_norm=cfg["y"] if y is None else y,
        font_name=cfg["font"],
        size_norm=cfg["size"],
        fill=cfg["fill"],
        anchor=cfg["anchor"],
        stroke_color=cfg.get("stroke_color"),
        stroke_width_norm=cfg.get("stroke_width", 0.0),
        gradient=cfg.get("gradient"),
    )


def paste_flag_contain(
    canvas: Image.Image, flag_filename: str, box_cfg: dict
) -> tuple[int, int, int, int]:
    w, h = canvas.size
    flag_img = open_rgba(resolve_file(BASE_DIR / "assets" / "flags", flag_filename))
    box_x, box_y = round(box_cfg["x"] * w), round(box_cfg["y"] * h)
    box_w, box_h = round(box_cfg["w"] * w), round(box_cfg["h"] * h)
    fitted = ImageOps.contain(flag_img, (box_w, box_h), method=Image.Resampling.LANCZOS)
    canvas.paste(fitted, (box_x, box_y), fitted)
    return (box_x, box_y, fitted.size[0], fitted.size[1])


def paste_cat_below_flag(
    canvas: Image.Image, flag_rect: tuple[int, int, int, int], cat_cfg: dict
) -> None:
    fx, fy, fw, fh = flag_rect
    w, h = canvas.size
    cat_img = load_cutout_cat(cat_cfg["image"])
    max_w, max_h = round(cat_cfg["max_w"] * w), round(cat_cfg["max_h"] * h)
    fitted = ImageOps.contain(cat_img, (max_w, max_h), method=Image.Resampling.LANCZOS)
    cw, _ = fitted.size
    overlap_y = round(cat_cfg.get("overlap_y", 0.005) * h)
    cat_x = fx + (fw - cw) // 2 + round(cat_cfg.get("offset_x", 0.0) * w)
    cat_y = fy + fh - overlap_y
    canvas.paste(fitted, (cat_x, cat_y), fitted)


def open_template(path: Path) -> Image.Image:
    """Open a card template as RGBA, flattening real alpha over white if present."""
    img = open_rgba(path)
    if img.getchannel("A").getextrema()[0] < 255:
        background = Image.new("RGBA", img.size, (255, 255, 255, 255))
        img = Image.alpha_composite(background, img)
    return img


# --------------------------------------------------------------------------- #
# Content generation
# --------------------------------------------------------------------------- #
def generate_lottery_numbers(
    used_pair_sets: dict[str, set[frozenset[int]]],
    pair_group: str | None,
    double_header: bool,
) -> dict:
    if pair_group:
        forbidden = used_pair_sets.setdefault(pair_group, set())
        if len(forbidden) >= 45:
            forbidden.clear()
        for _ in range(500):
            a, b = random.sample(range(10), 2)
            pair_set = frozenset((a, b))
            if pair_set not in forbidden:
                forbidden.add(pair_set)
                break
        else:
            a, b = random.sample(range(10), 2)
    else:
        a, b = random.sample(range(10), 2)

    units_a = sorted(random.sample([d for d in range(10) if d != a], 4))
    units_b = sorted(random.sample([d for d in range(10) if d != b], 4))
    row_a = [f"{a}{u}" for u in units_a]
    row_b = [f"{b}{u}" for u in units_b]

    tail_pool = row_a + row_b + [f"{a}{a}", f"{b}{b}"]
    chosen_tails = random.sample(tail_pool, 3)
    three_digits = []
    for tail in chosen_tails:
        valid_hundreds = [str(d) for d in range(1, 10) if not (str(d) == tail[0] == tail[1])]
        three_digits.append(random.choice(valid_hundreds) + tail)

    header = f"{a}{a}-{b}{b}" if double_header else f"{a}-{b}"
    return {"header": header, "row_a": row_a, "row_b": row_b, "three_digits": three_digits}


def format_thai_date(now_ict: datetime, fmt: str) -> str:
    be_year = (now_ict.year + 543) % 100
    if fmt == "spaced":
        return f"{now_ict.day:02d} / {now_ict.month:02d} / {be_year:02d}"
    return f"{now_ict.day:02d}/{now_ict.month:02d}/{be_year:02d}"


# --------------------------------------------------------------------------- #
# Card rendering
# --------------------------------------------------------------------------- #
def render_card(tpl_cfg: dict, market: dict, numbers: dict, date_str: str, out_path: Path) -> None:
    tpl_path = resolve_file(BASE_DIR / "assets" / "templates", tpl_cfg["base_image"])
    canvas = open_template(tpl_path)
    layout = tpl_cfg["layout"]

    flag_rect = None
    if market.get("flag") and "flag_box" in layout:
        flag_rect = paste_flag_contain(canvas, market["flag"], layout["flag_box"])
    if flag_rect is not None and "flag_cat" in layout:
        paste_cat_below_flag(canvas, flag_rect, layout["flag_cat"])

    title_base = layout["title"]
    if "title_lines" in market:
        for line in market["title_lines"]:
            draw_cfg_text(canvas, line["text"], {**title_base, **line})
    else:
        overrides = {
            k: market[f"title_{k}"]
            for k in ("x", "y", "size", "anchor")
            if f"title_{k}" in market
        }
        draw_cfg_text(canvas, market["title"], {**title_base, **overrides})

    draw_cfg_text(canvas, date_str, layout["date"])

    if "static_label" in layout:
        lbl = layout["static_label"]
        draw_cfg_text(canvas, lbl["text"], lbl)

    draw_cfg_text(canvas, numbers["header"], layout["main_pair"])

    g_cfg = layout["grid"]
    if g_cfg["mode"] == "columns":
        for col_x, row_vals in (
            (g_cfg["col_a_x"], numbers["row_a"]),
            (g_cfg["col_b_x"], numbers["row_b"]),
        ):
            for row_idx, val in enumerate(row_vals):
                draw_cfg_text(canvas, val, g_cfg, x=col_x, y=g_cfg["row_ys"][row_idx])
    else:
        for row_idx, pair_list in enumerate((numbers["row_a"], numbers["row_b"])):
            draw_cfg_text(canvas, " ".join(pair_list), g_cfg, y=g_cfg["row_ys"][row_idx])

    td_cfg = layout["three_digits"]
    if td_cfg["mode"] == "stack":
        for idx, val in enumerate(numbers["three_digits"]):
            draw_cfg_text(canvas, val, td_cfg, y=td_cfg["ys"][idx])
    else:
        draw_cfg_text(canvas, " ".join(numbers["three_digits"]), td_cfg)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    rgb_out = canvas.convert("RGB")
    if max(rgb_out.size) > MAX_OUTPUT_SIDE:
        rgb_out = ImageOps.contain(
            rgb_out, (MAX_OUTPUT_SIDE, MAX_OUTPUT_SIDE), method=Image.Resampling.LANCZOS
        )
    rgb_out.save(out_path, format="JPEG", quality=88, optimize=True)


# --------------------------------------------------------------------------- #
# Telegram
# --------------------------------------------------------------------------- #
def _endpoint(url: str) -> str:
    """Last path segment, e.g. 'sendPhoto' (never contains the bot token)."""
    return url.rsplit("/", 1)[-1]


def _redact(text: str, url: str) -> str:
    """Remove the bot token embedded in `url` from `text`."""
    if "/bot" in url:
        token = url.split("/bot", 1)[1].split("/", 1)[0]
        if token:
            text = text.replace(token, "<redacted>")
    return text


def telegram_request(url: str, data: dict, files: dict | None = None) -> str:
    endpoint = _endpoint(url)
    for attempt in range(5):
        if files:
            for item in files.values():
                f_obj = item[1] if isinstance(item, tuple) else item
                f_obj.seek(0)
        try:
            resp = requests.post(url, data=data, files=files, timeout=(10, 60))
        except requests.exceptions.RequestException as exc:
            reason = _redact(f"{type(exc).__name__}: {exc}", url)
            if attempt == 4:
                raise RuntimeError(
                    f"Telegram network error on {endpoint} after retries: {reason}"
                ) from None
            wait_s = 2 ** (attempt + 1)
            print(f"Network error on {endpoint} ({reason}), retrying in {wait_s}s...")
            time.sleep(wait_s)
            continue

        if resp.status_code == 429:
            try:
                retry_after = resp.json().get("parameters", {}).get("retry_after", 5)
            except ValueError:
                retry_after = 5
            time.sleep(retry_after + 1)
            continue
        if resp.status_code == 400:
            try:
                err = resp.json()
            except ValueError:
                err = {}
            new_chat_id = err.get("parameters", {}).get("migrate_to_chat_id")
            if new_chat_id and "chat_id" in data:
                print(
                    f"Group upgraded to supergroup. Auto-switching chat_id to {new_chat_id} "
                    "(please update TELEGRAM_CHAT_ID in GitHub Secrets)."
                )
                data["chat_id"] = str(new_chat_id)
                continue
        if resp.status_code >= 500:
            if attempt == 4:
                raise RuntimeError(f"Telegram HTTP {resp.status_code} on {endpoint}: {resp.text}")
            wait_s = 2 ** (attempt + 1)
            print(f"Telegram server HTTP {resp.status_code}, retrying in {wait_s}s...")
            time.sleep(wait_s)
            continue
        if not resp.ok:
            raise RuntimeError(f"Telegram HTTP {resp.status_code} on {endpoint}: {resp.text}")
        try:
            payload = resp.json()
        except ValueError:
            raise RuntimeError(
                f"Invalid JSON from Telegram on {endpoint}: {resp.text[:200]}"
            ) from None
        if not payload.get("ok"):
            raise RuntimeError(f"Telegram API error on {endpoint}: {payload}")
        return str(data.get("chat_id", ""))
    raise RuntimeError(f"Telegram API failed after retries: {endpoint}")


def send_template_group(token: str, chat_id: str, header_text: str, image_paths: list[Path]) -> str:
    base_url = f"https://api.telegram.org/bot{token}"
    chat_id = telegram_request(
        f"{base_url}/sendMessage",
        data={"chat_id": chat_id, "text": header_text},
    )
    time.sleep(1.0)

    chunk_size = 10
    for start in range(0, len(image_paths), chunk_size):
        chunk = image_paths[start : start + chunk_size]
        if len(chunk) == 1:
            with open(chunk[0], "rb") as f:
                chat_id = telegram_request(
                    f"{base_url}/sendPhoto",
                    data={"chat_id": chat_id},
                    files={"photo": f},
                )
        else:
            media = []
            with ExitStack() as stack:
                files = {}
                for idx, p in enumerate(chunk):
                    field = f"file_{idx}"
                    files[field] = stack.enter_context(open(p, "rb"))
                    media.append({"type": "photo", "media": f"attach://{field}"})
                chat_id = telegram_request(
                    f"{base_url}/sendMediaGroup",
                    data={"chat_id": chat_id, "media": json.dumps(media)},
                    files=files,
                )
        time.sleep(3.5)
    return chat_id


# --------------------------------------------------------------------------- #
# State / env helpers
# --------------------------------------------------------------------------- #
def require_env(name: str) -> str:
    value = os.environ.get(name, "").strip().strip('"').strip("'")
    if not value:
        raise SystemExit(f"Missing required environment variable: {name}")
    return value


def load_sent_state(path: Path) -> set[str]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return {str(k) for k in data} if isinstance(data, list) else set()
    except (OSError, ValueError):
        return set()


def save_sent_state(path: Path, keys: set[str], day_key: str) -> None:
    """Persist only today's keys so the file doesn't grow forever."""
    todays = sorted(k for k in keys if k.startswith(f"{day_key}/"))
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(todays), encoding="utf-8")
    tmp.replace(path)


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=["dry_run_sample", "dry_run_all", "live_send"],
        default="dry_run_sample",
    )
    parser.add_argument(
        "--skip-sent",
        action="store_true",
        help="live_send only: skip templates already sent today (tracked in output/sent_state.json).",
    )
    parser.add_argument(
        "--allow-no-raqm",
        action="store_true",
        help="live_send only: send even if libraqm is missing (Thai text may render incorrectly).",
    )
    args = parser.parse_args()
    live = args.mode == "live_send"

    has_raqm = features.check("raqm")
    print(f"libraqm active: {has_raqm}")

    token = chat_id = ""
    if live:
        if not has_raqm and not args.allow_no_raqm:
            raise SystemExit(
                "libraqm is not available, so Thai text would render incorrectly. "
                "Install libraqm/libfribidi (or pass --allow-no-raqm to send anyway)."
            )
        token = require_env("TELEGRAM_BOT_TOKEN")
        chat_id = require_env("TELEGRAM_CHAT_ID")

    with open(BASE_DIR / "config.json", "r", encoding="utf-8") as f:
        cfg = json.load(f)

    now_ict = datetime.now(ICT)
    out_root = BASE_DIR / "output"
    state_path = out_root / SENT_STATE_FILE
    day_key = now_ict.strftime("%Y-%m-%d")
    existing_state = load_sent_state(state_path) if live else set()
    skip_set = existing_state if args.skip_sent else set()
    generated_by_template: list[tuple[str, str, list[Path]]] = []

    for tpl in cfg["templates"]:
        tpl_id = tpl["id"]
        if f"{day_key}/{tpl_id}" in skip_set:
            print(f"[{tpl_id}] Already sent today, skipping.")
            continue

        used_pair_sets: dict[str, set[frozenset[int]]] = {}
        date_str = format_thai_date(now_ict, tpl["date_format"])
        markets = tpl["markets"]

        if args.mode == "dry_run_sample":
            sample_indices = {0, 4, 9, 10, 11, 14, 16, len(markets) - 1}
            selected_markets = [
                (i, markets[i]) for i in sorted(sample_indices) if 0 <= i < len(markets)
            ]
        else:
            selected_markets = list(enumerate(markets))

        tpl_paths: list[Path] = []
        for idx, market in selected_markets:
            nums = generate_lottery_numbers(
                used_pair_sets=used_pair_sets,
                pair_group=market.get("pair_group"),
                double_header=tpl["double_header"],
            )
            out_path = out_root / tpl_id / f"{idx + 1:02d}.jpg"
            render_card(tpl, market, nums, date_str, out_path)
            tpl_paths.append(out_path)

        header_text = f"📌 {tpl['label']} ({len(markets)} หวย) ประจำวันที่ {date_str}"
        generated_by_template.append((tpl_id, header_text, tpl_paths))
        print(f"[{tpl_id}] Generated {len(tpl_paths)} images (date={date_str})")

    if live:
        if not generated_by_template:
            today_th = format_thai_date(now_ict, "compact")
            msg = f"✅ ใบหวยประจำวันที่ {today_th} ส่งครบทุกชุดแล้ว (ข้ามการส่งซ้ำ)"
            print(msg)
            telegram_request(
                f"https://api.telegram.org/bot{token}/sendMessage",
                data={"chat_id": chat_id, "text": msg},
            )
            return

        total_sent = 0
        for tpl_id, header_text, paths in generated_by_template:
            print(f"Sending {tpl_id} ({len(paths)} images) to Telegram...")
            chat_id = send_template_group(token, chat_id, header_text, paths)
            total_sent += len(paths)
            existing_state.add(f"{day_key}/{tpl_id}")
            save_sent_state(state_path, existing_state, day_key)
        print(f"All {total_sent} images delivered.")


if __name__ == "__main__":
    main()
