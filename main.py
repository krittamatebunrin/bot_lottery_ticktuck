import argparse
import json
import os
import random
import time
from collections import deque
from contextlib import ExitStack
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests
from PIL import Image, ImageDraw, ImageFont, ImageOps, features

ICT = timezone(timedelta(hours=7))
BASE_DIR = Path(__file__).resolve().parent
FONT_CACHE: dict[tuple[str, int], ImageFont.FreeTypeFont] = {}
CAT_CACHE: dict[str, Image.Image] = {}


def resolve_file(directory: Path, filename: str) -> Path:
    direct = directory / filename
    if direct.exists():
        return direct
    target_lower = filename.lower()
    target_stem = Path(filename).stem.lower()
    for item in directory.iterdir():
        if item.is_file() and item.name.lower() == target_lower:
            return item
    for item in directory.iterdir():
        if item.is_file() and item.stem.lower() == target_stem:
            return item
    raise FileNotFoundError(f"Missing file '{filename}' in {directory}")


def get_font(font_filename: str, size_px: int) -> ImageFont.FreeTypeFont:
    key = (font_filename, size_px)
    if key not in FONT_CACHE:
        font_path = resolve_file(BASE_DIR / "fonts", font_filename)
        if features.check("raqm"):
            FONT_CACHE[key] = ImageFont.truetype(
                str(font_path), size=size_px, layout_engine=ImageFont.Layout.RAQM
            )
        else:
            FONT_CACHE[key] = ImageFont.truetype(str(font_path), size=size_px)
    return FONT_CACHE[key]


def load_cutout_cat(filename: str) -> Image.Image:
    if filename in CAT_CACHE:
        return CAT_CACHE[filename]

    cat_path = resolve_file(BASE_DIR / "assets", filename)
    img = Image.open(cat_path).convert("RGBA")
    img = ImageOps.contain(img, (480, 480), method=Image.Resampling.LANCZOS)
    w, h = img.size
    px = img.load()

    corner_alphas = [px[0, 0][3], px[w - 1, 0][3], px[0, h - 1][3], px[w - 1, h - 1][3]]
    if all(a < 10 for a in corner_alphas):
        bbox = img.getbbox()
        result = img.crop(bbox) if bbox else img
        CAT_CACHE[filename] = result
        return result

    visited = bytearray(w * h)
    q = deque()

    def is_bg(x: int, y: int) -> bool:
        r, g, b, a = px[x, y]
        return a < 10 or (r >= 225 and g >= 225 and b >= 225)

    for x in range(w):
        for y in (0, h - 1):
            idx = y * w + x
            if not visited[idx] and is_bg(x, y):
                visited[idx] = 1
                q.append((x, y))
    for y in range(h):
        for x in (0, w - 1):
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

    comp_seen = bytearray(w * h)
    components: list[tuple[list[tuple[int, int]], tuple[int, int, int, int]]] = []
    for y in range(h):
        for x in range(w):
            idx = y * w + x
            if px[x, y][3] > 0 and not comp_seen[idx]:
                comp_seen[idx] = 1
                cq = deque([(x, y)])
                pts = []
                x0 = x1 = x
                y0 = y1 = y
                while cq:
                    px_x, px_y = cq.popleft()
                    pts.append((px_x, px_y))
                    if px_x < x0:
                        x0 = px_x
                    if px_x > x1:
                        x1 = px_x
                    if px_y < y0:
                        y0 = px_y
                    if px_y > y1:
                        y1 = px_y
                    for dx in (-1, 0, 1):
                        for dy in (-1, 0, 1):
                            if dx == 0 and dy == 0:
                                continue
                            nx, ny = px_x + dx, px_y + dy
                            if 0 <= nx < w and 0 <= ny < h:
                                nidx = ny * w + nx
                                if px[nx, ny][3] > 0 and not comp_seen[nidx]:
                                    comp_seen[nidx] = 1
                                    cq.append((nx, ny))
                components.append((pts, (x0, y0, x1, y1)))

    if components:
        main_pts, (mx0, my0, mx1, my1) = max(components, key=lambda c: len(c[0]))
        min_keep_area = max(10, int(len(main_pts) * 0.03))
        pad = 20
        for pts, (bx0, by0, bx1, by1) in components:
            near_main = not (
                bx1 < mx0 - pad or bx0 > mx1 + pad or by1 < my0 - pad or by0 > my1 + pad
            )
            if len(pts) < min_keep_area and not near_main:
                for px_x, px_y in pts:
                    r, g, b, _ = px[px_x, px_y]
                    px[px_x, px_y] = (r, g, b, 0)

    edge_updates: list[tuple[int, int, int]] = []
    for y in range(h):
        for x in range(w):
            r, g, b, a = px[x, y]
            if a == 0:
                continue
            touches_bg = False
            for nx, ny in ((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)):
                if not (0 <= nx < w and 0 <= ny < h) or px[nx, ny][3] == 0:
                    touches_bg = True
                    break
            if touches_bg:
                m = min(r, g, b)
                if m > 150:
                    new_a = max(0, min(255, round((230 - m) / 80.0 * 255)))
                    edge_updates.append((x, y, new_a))
    for x, y, new_a in edge_updates:
        r, g, b, _ = px[x, y]
        px[x, y] = (r, g, b, new_a)

    bbox = img.getbbox()
    result = img.crop(bbox) if bbox else img
    CAT_CACHE[filename] = result
    return result


def hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
    h = hex_color.lstrip("#")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


def build_vertical_gradient(width: int, height: int, stops: list[str]) -> Image.Image:
    rgb_stops = [hex_to_rgb(c) for c in stops]
    grad = Image.new("RGBA", (width, height))
    px = grad.load()
    n = len(rgb_stops) - 1
    for y in range(height):
        t = y / max(height - 1, 1)
        seg = min(int(t * n), n - 1) if n > 0 else 0
        local_t = (t * n) - seg if n > 0 else 0.0
        c1, c2 = rgb_stops[seg], rgb_stops[min(seg + 1, n)]
        r = int(c1[0] + (c2[0] - c1[0]) * local_t)
        g = int(c1[1] + (c2[1] - c1[1]) * local_t)
        b = int(c1[2] + (c2[2] - c1[2]) * local_t)
        for x in range(width):
            px[x, y] = (r, g, b, 255)
    return grad


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
    x = round(x_norm * w)
    y = round(y_norm * h)

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
    mask_draw = ImageDraw.Draw(mask)
    mask_draw.text((x - bx0, y - by0), text, font=font, fill=255, anchor=anchor)

    grad_img = build_vertical_gradient(bw, bh, gradient)
    canvas.paste(grad_img, (bx0, by0), mask)


def paste_flag_contain(
    canvas: Image.Image, flag_filename: str, box_cfg: dict
) -> tuple[int, int, int, int]:
    w, h = canvas.size
    flag_path = resolve_file(BASE_DIR / "assets" / "flags", flag_filename)
    flag_img = Image.open(flag_path).convert("RGBA")

    box_x = round(box_cfg["x"] * w)
    box_y = round(box_cfg["y"] * h)
    box_w = round(box_cfg["w"] * w)
    box_h = round(box_cfg["h"] * h)

    fitted = ImageOps.contain(flag_img, (box_w, box_h), method=Image.Resampling.LANCZOS)
    canvas.paste(fitted, (box_x, box_y), fitted)
    return (box_x, box_y, fitted.size[0], fitted.size[1])


def paste_cat_below_flag(
    canvas: Image.Image, flag_rect: tuple[int, int, int, int], cat_cfg: dict
) -> None:
    fx, fy, fw, fh = flag_rect
    w, h = canvas.size
    cat_img = load_cutout_cat(cat_cfg["image"])

    max_w = round(cat_cfg["max_w"] * w)
    max_h = round(cat_cfg["max_h"] * h)
    fitted = ImageOps.contain(cat_img, (max_w, max_h), method=Image.Resampling.LANCZOS)
    cw, _ = fitted.size

    overlap_y = round(cat_cfg.get("overlap_y", 0.005) * h)
    cat_x = fx + (fw - cw) // 2 + round(cat_cfg.get("offset_x", 0.0) * w)
    cat_y = fy + fh - overlap_y
    canvas.paste(fitted, (cat_x, cat_y), fitted)


def generate_lottery_numbers(
    used_pair_sets: dict[str, set[frozenset[int]]],
    pair_group: str | None,
    double_header: bool,
) -> dict:
    if pair_group:
        forbidden = used_pair_sets.setdefault(pair_group, set())
        while True:
            a, b = random.sample(range(10), 2)
            pair_set = frozenset((a, b))
            if pair_set not in forbidden:
                forbidden.add(pair_set)
                break
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
    return {
        "header": header,
        "row_a": row_a,
        "row_b": row_b,
        "three_digits": three_digits,
    }


def format_thai_date(now_ict: datetime, fmt: str) -> str:
    be_year = (now_ict.year + 543) % 100
    if fmt == "spaced":
        return f"{now_ict.day:02d} / {now_ict.month:02d} / {be_year:02d}"
    return f"{now_ict.day:02d}/{now_ict.month:02d}/{be_year:02d}"


def render_card(tpl_cfg: dict, market: dict, numbers: dict, date_str: str, out_path: Path) -> None:
    tpl_path = resolve_file(BASE_DIR / "assets" / "templates", tpl_cfg["base_image"])
    canvas = Image.open(tpl_path).convert("RGBA")
    layout = tpl_cfg["layout"]

    flag_rect = None
    if market.get("flag") and "flag_box" in layout:
        flag_rect = paste_flag_contain(canvas, market["flag"], layout["flag_box"])
    if flag_rect is not None and "flag_cat" in layout:
        paste_cat_below_flag(canvas, flag_rect, layout["flag_cat"])

    title_base = layout["title"]
    if "title_lines" in market:
        for line in market["title_lines"]:
            draw_styled_text(
                canvas,
                text=line["text"],
                x_norm=line.get("x", title_base["x"]),
                y_norm=line.get("y", title_base["y"]),
                font_name=line.get("font", title_base["font"]),
                size_norm=line.get("size", title_base["size"]),
                fill=line.get("fill", title_base["fill"]),
                anchor=line.get("anchor", title_base["anchor"]),
                stroke_color=line.get("stroke_color", title_base.get("stroke_color")),
                stroke_width_norm=line.get("stroke_width", title_base.get("stroke_width", 0.0)),
            )
    else:
        draw_styled_text(
            canvas,
            text=market["title"],
            x_norm=market.get("title_x", title_base["x"]),
            y_norm=market.get("title_y", title_base["y"]),
            font_name=title_base["font"],
            size_norm=market.get("title_size", title_base["size"]),
            fill=title_base["fill"],
            anchor=market.get("title_anchor", title_base["anchor"]),
            stroke_color=title_base.get("stroke_color"),
            stroke_width_norm=title_base.get("stroke_width", 0.0),
        )

    d_cfg = layout["date"]
    draw_styled_text(
        canvas,
        text=date_str,
        x_norm=d_cfg["x"],
        y_norm=d_cfg["y"],
        font_name=d_cfg["font"],
        size_norm=d_cfg["size"],
        fill=d_cfg["fill"],
        anchor=d_cfg["anchor"],
        stroke_color=d_cfg.get("stroke_color"),
        stroke_width_norm=d_cfg.get("stroke_width", 0.0),
    )

    if "static_label" in layout:
        lbl = layout["static_label"]
        draw_styled_text(
            canvas,
            text=lbl["text"],
            x_norm=lbl["x"],
            y_norm=lbl["y"],
            font_name=lbl["font"],
            size_norm=lbl["size"],
            fill=lbl["fill"],
            anchor=lbl["anchor"],
        )

    mp = layout["main_pair"]
    draw_styled_text(
        canvas,
        text=numbers["header"],
        x_norm=mp["x"],
        y_norm=mp["y"],
        font_name=mp["font"],
        size_norm=mp["size"],
        fill=mp["fill"],
        anchor=mp["anchor"],
        stroke_color=mp.get("stroke_color"),
        stroke_width_norm=mp.get("stroke_width", 0.0),
        gradient=mp.get("gradient"),
    )

    g_cfg = layout["grid"]
    if g_cfg["mode"] == "columns":
        for row_idx, val in enumerate(numbers["row_a"]):
            draw_styled_text(
                canvas,
                text=val,
                x_norm=g_cfg["col_a_x"],
                y_norm=g_cfg["row_ys"][row_idx],
                font_name=g_cfg["font"],
                size_norm=g_cfg["size"],
                fill=g_cfg["fill"],
                anchor=g_cfg["anchor"],
                stroke_color=g_cfg.get("stroke_color"),
                stroke_width_norm=g_cfg.get("stroke_width", 0.0),
                gradient=g_cfg.get("gradient"),
            )
        for row_idx, val in enumerate(numbers["row_b"]):
            draw_styled_text(
                canvas,
                text=val,
                x_norm=g_cfg["col_b_x"],
                y_norm=g_cfg["row_ys"][row_idx],
                font_name=g_cfg["font"],
                size_norm=g_cfg["size"],
                fill=g_cfg["fill"],
                anchor=g_cfg["anchor"],
                stroke_color=g_cfg.get("stroke_color"),
                stroke_width_norm=g_cfg.get("stroke_width", 0.0),
                gradient=g_cfg.get("gradient"),
            )
    else:
        for row_idx, pair_list in enumerate((numbers["row_a"], numbers["row_b"])):
            line_text = " ".join(pair_list)
            draw_styled_text(
                canvas,
                text=line_text,
                x_norm=g_cfg["x"],
                y_norm=g_cfg["row_ys"][row_idx],
                font_name=g_cfg["font"],
                size_norm=g_cfg["size"],
                fill=g_cfg["fill"],
                anchor=g_cfg["anchor"],
                stroke_color=g_cfg.get("stroke_color"),
                stroke_width_norm=g_cfg.get("stroke_width", 0.0),
            )

    td_cfg = layout["three_digits"]
    if td_cfg["mode"] == "stack":
        for idx, val in enumerate(numbers["three_digits"]):
            draw_styled_text(
                canvas,
                text=val,
                x_norm=td_cfg["x"],
                y_norm=td_cfg["ys"][idx],
                font_name=td_cfg["font"],
                size_norm=td_cfg["size"],
                fill=td_cfg["fill"],
                anchor=td_cfg["anchor"],
                stroke_color=td_cfg.get("stroke_color"),
                stroke_width_norm=td_cfg.get("stroke_width", 0.0),
            )
    else:
        draw_styled_text(
            canvas,
            text=" ".join(numbers["three_digits"]),
            x_norm=td_cfg["x"],
            y_norm=td_cfg["y"],
            font_name=td_cfg["font"],
            size_norm=td_cfg["size"],
            fill=td_cfg["fill"],
            anchor=td_cfg["anchor"],
            stroke_color=td_cfg.get("stroke_color"),
            stroke_width_norm=td_cfg.get("stroke_width", 0.0),
            gradient=td_cfg.get("gradient"),
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.convert("RGB").save(out_path, format="JPEG", quality=95)


def telegram_request(url: str, data: dict, files: dict | None = None) -> str:
    for _ in range(5):
        if files:
            for item in files.values():
                f_obj = item[1] if isinstance(item, tuple) else item
                f_obj.seek(0)
        resp = requests.post(url, data=data, files=files, timeout=60)
        if resp.status_code == 429:
            retry_after = resp.json().get("parameters", {}).get("retry_after", 5)
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
        if not resp.ok:
            raise RuntimeError(f"Telegram HTTP {resp.status_code}: {resp.text}")
        payload = resp.json()
        if not payload.get("ok"):
            raise RuntimeError(f"Telegram API error: {payload}")
        return str(data.get("chat_id", ""))
    raise RuntimeError(f"Telegram API failed after retries: {url}")


def send_template_group(
    token: str, chat_id: str, header_text: str, image_paths: list[Path]
) -> str:
    base_url = f"https://api.telegram.org/bot{token}"

    # 1. Send header text as its own message before the images
    chat_id = telegram_request(
        f"{base_url}/sendMessage",
        data={"chat_id": chat_id, "text": header_text},
    )
    time.sleep(1.0)

    # 2. Send images cleanly without captions
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=["dry_run_sample", "dry_run_all", "live_send"],
        default="dry_run_sample",
    )
    args = parser.parse_args()

    print(f"libraqm active: {features.check('raqm')}")

    with open(BASE_DIR / "config.json", "r", encoding="utf-8") as f:
        cfg = json.load(f)

    now_ict = datetime.now(ICT)
    out_root = BASE_DIR / "output"
    generated_by_template: list[tuple[str, str, list[Path]]] = []

    for tpl in cfg["templates"]:
        tpl_id = tpl["id"]
        used_pair_sets: dict[str, set[frozenset[int]]] = {}
        date_str = format_thai_date(now_ict, tpl["date_format"])
        markets = tpl["markets"]

        if args.mode == "dry_run_sample":
            sample_indices = {0, 4, 9, 10, 11, 14, 16, len(markets) - 1}
            selected_markets = [(i, markets[i]) for i in sorted(sample_indices) if i < len(markets)]
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

    if args.mode == "live_send":
        token = os.environ["TELEGRAM_BOT_TOKEN"].strip().strip('"').strip("'")
        chat_id = os.environ["TELEGRAM_CHAT_ID"].strip().strip('"').strip("'")
        for tpl_id, header_text, paths in generated_by_template:
            print(f"Sending {tpl_id} ({len(paths)} images) to Telegram...")
            chat_id = send_template_group(token, chat_id, header_text, paths)
        print("All 97 images delivered.")


if __name__ == "__main__":
    main()
