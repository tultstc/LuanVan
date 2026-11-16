"""Download candidate clips for the hpd classes the public datasets lack (foot on table,
sleeping at a desk, fighting) from Pexels (stock footage, Pexels License) and YouTube.

Every clip is only a candidate: search results are noisy, so each one gets a contact sheet
(sheets/<class>/<id>.jpg) and a row in manifest.csv with keep="" to fill in by hand
(y / n) before any of it is converted to training data. The manifest also keeps source,
url, author and license for attribution. Re-running skips clips already in the manifest.

Sources:
  pexels   needs a free API key in the PEXELS_API_KEY env var (https://www.pexels.com/api/).
  youtube  yt-dlp search; only Creative Commons videos unless --any-license.

Usage (repo root; needs yt-dlp >= 2026 and ffmpeg):
    python projects/fgtn_ava/crawl_web.py --out data/web_extra --sources pexels youtube --per-query 15
"""
import argparse
import csv
import json
import os
import re
import subprocess
import urllib.parse
import urllib.request
from pathlib import Path

# hpd class id -> (name, search queries)
QUERIES = {
    2: ("sleep_on_table", ["sleeping at desk", "fell asleep at work desk", "tired worker sleeping on table",
                           "employee asleep office", "student sleeping on desk classroom"]),
    3: ("foot_on_table", ["feet on desk", "feet up on table office", "man with legs on desk",
                          "businessman feet on table", "relaxing feet on office desk"]),
    4: ("dangerous_action", ["fight caught on cctv", "workplace fight cctv", "street fight security camera",
                             "people fighting", "men fighting punching", "brawl surveillance camera"]),
}
# titles that are almost never real footage of the action (fetish, ASMR, games, music, tutorials)
BLOCK = re.compile(r"asmr|shoeplay|dangl|heels|socks|barefoot|toes|sole|fetish|stinky|game|gameplay|anime|"
                   r"animation|cartoon|minecraft|roblox|music|song|lyrics|trailer|tutorial|how to|sleep sounds|"
                   r"rain sounds|lofi|meditation|green ?screen|boxing match|ufc|mma|wwe|"
                   r"drone|war\b|military|soldier|marine|explo|accident|crash|fails|how it works|privacy", re.I)
FIELDS = ["cls", "cls_name", "source", "id", "url", "title", "author", "license", "duration", "query", "path", "keep"]


def blocked(title):
    return bool(BLOCK.search(title or ""))


def pexels_search(query, n, key):
    url = "https://api.pexels.com/videos/search?" + urllib.parse.urlencode({"query": query, "per_page": n})
    req = urllib.request.Request(url, headers={"Authorization": key, "User-Agent": "hpd-research-crawler"})
    for v in json.load(urllib.request.urlopen(req, timeout=30))["videos"]:
        files = [f for f in v["video_files"] if f.get("height") and f["file_type"] == "video/mp4"]
        best = min(files, key=lambda f: abs(f["height"] - 720))  # ~720p: enough for 384x256 input
        title = v["url"].rstrip("/").split("/")[-1]
        yield dict(source="pexels", id=str(v["id"]), url=v["url"], title=title, author=v["user"]["name"],
                   license="Pexels License", duration=v["duration"], download=best["link"])


def youtube_search(query, n, any_license, ytdlp):
    sp = "" if any_license else "&sp=EgIwAQ%253D%253D"  # YouTube's Creative Commons search filter
    url = f"https://www.youtube.com/results?search_query={urllib.parse.quote_plus(query)}{sp}"
    out = subprocess.run([ytdlp, "--flat-playlist", "--playlist-end", str(n), "--print",
                          "%(id)s\t%(duration)s\t%(channel)s\t%(title)s", url],
                         capture_output=True, text=True, timeout=300).stdout
    for line in out.splitlines():
        vid, dur, chan, title = (line.split("\t") + ["", "", ""])[:4]
        yield dict(source="youtube", id=vid, url=f"https://www.youtube.com/watch?v={vid}", title=title,
                   author=chan, license="CC BY (YouTube filter)" if not any_license else "unknown",
                   duration=float(dur) if dur not in ("", "NA", "None") else 0.0, download=None)


def download(c, dst, ytdlp):
    if c["source"] == "pexels":
        req = urllib.request.Request(c["download"], headers={"User-Agent": "hpd-research-crawler"})
        with urllib.request.urlopen(req, timeout=120) as r, open(dst, "wb") as f:
            f.write(r.read())
    else:
        subprocess.run([ytdlp, "-q", "--no-warnings", "-f", "bv*[height<=720][ext=mp4]+ba[ext=m4a]/b[height<=720]/b",
                        "--merge-output-format", "mp4", "-o", str(dst), c["url"]], check=True, timeout=900)


def contact_sheet(video, sheet, duration):
    # 8 frames spread over the clip, 4x2 tiles, each 320 px wide
    fps = 8 / max(float(duration), 1.0)
    subprocess.run(["ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-i", str(video), "-vf",
                    f"fps={fps:.4f},scale=320:-2,tile=4x2", "-frames:v", "1", str(sheet)], check=True, timeout=300)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/web_extra")
    ap.add_argument("--sources", nargs="+", default=["pexels", "youtube"], choices=["pexels", "youtube"])
    ap.add_argument("--classes", nargs="+", type=int, default=sorted(QUERIES))
    ap.add_argument("--per-query", type=int, default=15, help="search results looked at per query and source")
    ap.add_argument("--max-duration", type=float, default=180, help="seconds; longer clips are skipped")
    ap.add_argument("--any-license", action="store_true", help="YouTube: also non-Creative-Commons videos")
    ap.add_argument("--ytdlp", default="yt-dlp")
    args = ap.parse_args()

    key = os.environ.get("PEXELS_API_KEY")
    assert "pexels" not in args.sources or key, "set PEXELS_API_KEY or drop pexels from --sources"
    out = Path(args.out)
    manifest = out / "manifest.csv"
    seen = set()
    if manifest.exists():
        seen = {(r["source"], r["id"]) for r in csv.DictReader(open(manifest))}
    else:
        out.mkdir(parents=True, exist_ok=True)
        csv.writer(open(manifest, "w", newline="")).writerow(FIELDS)

    for cls in args.classes:
        name, queries = QUERIES[cls]
        for q in queries:
            for src in args.sources:
                try:
                    found = list(pexels_search(q, args.per_query, key) if src == "pexels"
                                 else youtube_search(q, args.per_query, args.any_license, args.ytdlp))
                except Exception as e:  # one failed search must not stop the crawl
                    print(f"search failed {src} '{q}': {e}")
                    continue
                for c in found:
                    if (src, c["id"]) in seen or blocked(c["title"]) or not 0 < c["duration"] <= args.max_duration:
                        continue
                    seen.add((src, c["id"]))  # also a clip found again under another class
                    video = out / "videos" / name / f"{src}_{c['id']}.mp4"
                    sheet = out / "sheets" / name / f"{src}_{c['id']}.jpg"
                    video.parent.mkdir(parents=True, exist_ok=True)
                    sheet.parent.mkdir(parents=True, exist_ok=True)
                    try:
                        download(c, video, args.ytdlp)
                        contact_sheet(video, sheet, c["duration"])
                    except Exception as e:
                        print(f"download failed {c['url']}: {e}")
                        video.unlink(missing_ok=True)
                        continue
                    row = dict(c, cls=cls, cls_name=name, query=q, path=str(video.relative_to(out)), keep="")
                    with open(manifest, "a", newline="") as f:
                        csv.DictWriter(f, FIELDS, extrasaction="ignore").writerow(row)
                    print(f"[{name}] {src} {c['id']} {c['duration']:.0f}s {c['title'][:60]}")

    rows = list(csv.DictReader(open(manifest)))
    for cls in args.classes:
        print(f"{QUERIES[cls][0]}: {sum(r['cls'] == str(cls) for r in rows)} candidates")


if __name__ == "__main__":
    assert blocked("Shoeplay ASMR at the office") and blocked("3D anime fight scene") and not blocked("Caught on CCTV | Fist fight")
    main()
