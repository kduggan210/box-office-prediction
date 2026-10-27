import re, pathlib

for name, path in [("Endgame", "rl3059975681"), ("Ocean's Twelve", "rl426149377")]:
    html = pathlib.Path(f"data/raw/bom_releases/{path}.html").read_text(encoding="utf-8")
    print("==", name)
    for m in re.finditer(r"tt\d{6,9}", html):
        s = max(0, m.start() - 70)
        print(m.group(0), "|", html[s:m.end() + 30].replace("\n", " "))