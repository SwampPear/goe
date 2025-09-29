"""
from src.io.listing import list_local_tifs, http_listdir, enrich_with_head

# Local
paths = list_local_tifs("/path/to/volumes/scroll2")
print(f"{len(paths)} tifs")
print(paths[:5])

# Remote
items = http_listdir("https://data.server/", "volumes/scroll2")
items = enrich_with_head(items)  # optional HEAD metadata
for it in items[:5]:
    print(it["name"], it["size"], it["modified"])
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


from src.io.listing import autoindex_listdir

def main():
    items = autoindex_listdir("https://dl.ash2txt.org/", "full-scrolls/Scroll1/PHercParis4.volpkg")


if __name__ == '__main__':
    main()
