"""Render the wiki drafts (docs/wiki/*.md, GitHub wiki pages) as ONE self-contained HTML preview, opened from disk:
pages in wiki order with a page bar, GitHub-style headings / tables / code, images from docs/wiki/images embedded as
base64 (a missing image shows a 'screenshot pending' box), image alt text as the caption, and wiki links rewritten
for the single file ('README#seatable' -> the SeaTable section of the README page; heading ids follow GitHub's slugs,
prefixed per page). Writes figures/wiki/<stamp>_wiki_preview.html. Usage: wiki_render.py [--out PATH]"""
import argparse
import base64
import html
import mimetypes
import os
import re

import markdown

from faab.config import FIGURES, REPO
from faab.partners.sheet import utc_stamp

WIKI = os.path.join(REPO, "docs", "wiki")
PAGES = [("README", "README"), ("SeaTable-dos-and-donts", "SeaTable dos and don'ts"),
         ("Proofreading-guide", "Proofreading guide")]


def slug(text, sep="-"):
    """GitHub's heading anchor: lowercase, punctuation dropped (except - and _), spaces -> '-'"""
    t = re.sub(r"<[^>]+>", "", text).strip().lower()
    t = re.sub(r"[^\w\- ]", "", t)
    return t.replace(" ", sep)


def page_html(key, md_text):
    md = markdown.Markdown(extensions=["tables", "fenced_code", "toc", "sane_lists"],
                           extension_configs={"toc": {"slugify": lambda v, s: f"{key}--{slug(v)}"}})
    body = md.convert(md_text)
    keys = {k for k, _ in PAGES}

    def link(m):
        href = html.unescape(m.group(1))
        if href.startswith("#"):
            return f'href="#{key}--{href[1:]}"'
        page, _, anchor = href.partition("#")
        if page in keys:
            return f'href="#{page}--{anchor}"' if anchor else f'href="#page-{page}"'
        return f'href="{html.escape(href)}" target="_blank" rel="noopener"'

    body = re.sub(r'href="([^"]*)"', link, body)
    # markdown renders ![alt](src) on its own line as <p><img alt="..." src="..." /></p>
    return re.sub(r'<p><img alt="(?P<alt>[^"]*)" src="(?P<src>[^"]*)" ?/?></p>',
                  lambda m: figure(m.group("src"), m.group("alt")), body)


def figure(src, alt):
    """an image file embedded as base64 with its alt text as the caption, or a 'pending' box"""
    path = os.path.join(WIKI, html.unescape(src))
    alt = html.unescape(alt)
    cap = f"<figcaption>{html.escape(alt)}</figcaption>"
    if os.path.exists(path):
        mime = mimetypes.guess_type(path)[0] or "image/png"
        data = base64.b64encode(open(path, "rb").read()).decode()
        return f'<figure><img src="data:{mime};base64,{data}" alt="{html.escape(alt)}">{cap}</figure>'
    return f'<figure><div class="pending">screenshot pending: {html.escape(src)}</div>{cap}</figure>'


CSS = """
:root { --fg:#1f2328; --muted:#59636e; --bg:#ffffff; --line:#d1d9e0; --soft:#f6f8fa; --link:#0969da; }
* { box-sizing: border-box; }
body { margin:0; background:var(--bg); color:var(--fg);
  font: 16px/1.5 -apple-system, BlinkMacSystemFont, "Segoe UI", "Noto Sans", Helvetica, Arial, sans-serif; }
header { position: sticky; top:0; background:var(--soft); border-bottom:1px solid var(--line); padding:10px 16px; z-index:2; }
header .title { font-weight:600; margin-right:16px; }
header a { margin-right:14px; color:var(--link); text-decoration:none; }
header a:hover { text-decoration:underline; }
main { max-width: 980px; margin: 0 auto; padding: 8px 16px 64px; }
section.page { border-bottom: 1px solid var(--line); padding-bottom: 32px; margin-bottom: 24px; }
.stamp { color:var(--muted); font-size: 13px; margin: 12px 0 0; }
h1 { font-size: 2em; border-bottom:1px solid var(--line); padding-bottom:.3em; margin: 24px 0 16px; }
h2 { font-size: 1.5em; border-bottom:1px solid var(--line); padding-bottom:.3em; margin: 24px 0 16px; }
h3 { font-size: 1.25em; margin: 24px 0 16px; }
a { color: var(--link); }
table { border-collapse: collapse; display:block; overflow-x:auto; margin: 0 0 16px; }
th, td { border:1px solid var(--line); padding:6px 13px; vertical-align: top; }
tr:nth-child(2n) td { background: var(--soft); }
code { background: rgba(129,139,152,.15); padding: .2em .4em; border-radius: 6px; font-size: 85%; }
pre { background: var(--soft); padding: 16px; border-radius: 6px; overflow-x: auto; }
pre code { background: none; padding: 0; }
figure { margin: 16px 0 24px; }
figure img { max-width: 100%; border: 1px solid var(--line); border-radius: 6px; }
figcaption { color: var(--muted); font-size: 14px; margin-top: 6px; }
.pending { border: 2px dashed var(--line); border-radius: 6px; padding: 40px 16px; text-align: center; color: var(--muted); }
"""


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out")
    a = ap.parse_args()
    stamp = utc_stamp()
    secs, nav = [], []
    for key, title in PAGES:
        body = page_html(key, open(os.path.join(WIKI, f"{key}.md"), encoding="utf-8").read())
        secs.append(f'<section class="page" id="page-{key}">{body}</section>')
        nav.append(f'<a href="#page-{key}">{html.escape(title)}</a>')
    doc = (f'<!doctype html><html lang="en"><head><meta charset="utf-8">'
           f'<meta name="viewport" content="width=device-width, initial-scale=1"><title>FAAB wiki preview</title>'
           f'<style>{CSS}</style></head><body><header><span class="title">FAAB wiki (draft preview)</span>'
           f'{"".join(nav)}</header><main><p class="stamp">Rendered {stamp} from faab/docs/wiki; '
           f'links between pages jump within this file.</p>{"".join(secs)}</main></body></html>')
    out = a.out or os.path.join(FIGURES, "wiki", f"{stamp}_wiki_preview.html")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    open(out, "w", encoding="utf-8").write(doc)
    n_img = doc.count("<figure>")
    print(f"written {out} ({len(doc) // 1024} KB, {n_img} figures, {doc.count('screenshot pending')} pending)")


if __name__ == "__main__":
    main()
