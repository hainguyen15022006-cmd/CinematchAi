"""
Gom tat ca code trong project thanh 1 file .txt duy nhat,
chia ro theo tung thu muc de de doc.
"""

import os

# ==== CAU HINH ====
ROOT_DIR = "."                      # thu muc goc can quet (mac dinh: thu muc hien tai)
OUTPUT_FILE = "all_code.txt"        # file txt xuat ra
EXTENSIONS = (".py", ".md", ".toml", ".yml", ".yaml", ".txt")
EXCLUDE_DIRS = {".git", "__pycache__", "node_modules", ".venv", "venv", "data", "dist", "build"}
EXCLUDE_FILES = {OUTPUT_FILE}       # tranh tu gom chinh file output vao


def collect_files(root_dir):
    collected = []
    for current_dir, dirs, files in os.walk(root_dir):
        # loai bo thu muc khong can quet (sua truc tiep list dirs de os.walk khong di vao)
        dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS]

        for filename in sorted(files):
            if filename in EXCLUDE_FILES:
                continue
            if filename.lower().endswith(EXTENSIONS):
                full_path = os.path.join(current_dir, filename)
                collected.append(full_path)
    return collected


def main():
    files = collect_files(ROOT_DIR)

    # encoding="utf-8" de khong bi loi font tieng Viet
    with open(OUTPUT_FILE, "w", encoding="utf-8") as out:
        last_dir = None
        for file_path in files:
            current_dir = os.path.dirname(file_path)

            # in header thu muc khi chuyen sang thu muc moi
            if current_dir != last_dir:
                out.write(f"\n\n{'#' * 60}\n# FOLDER: {current_dir}\n{'#' * 60}\n")
                last_dir = current_dir

            out.write(f"\n----- FILE: {os.path.basename(file_path)} -----\n")
            try:
                with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                    out.write(f.read())
            except Exception as e:
                out.write(f"[Khong doc duoc file: {e}]\n")

    print(f"Xong! Da gom {len(files)} file vao: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()