import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


from src.io.listing import list_volume_files

def main():
    DS = "https://dl.ash2txt.org/full-scrolls/Scroll1/PHercParis4.volpkg/"

    paths = list_volume_files(DS)

    print(paths)


if __name__ == '__main__':
    main()
