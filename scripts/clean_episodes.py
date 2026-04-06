"""
Clean episode data in experience.db:
  1. Delete episodes where assistant_response has >5% Chinese characters
  2. Fix underscore→apostrophe artifacts in all remaining responses
"""

import re
import sqlite3
import sys
from pathlib import Path

DB_PATH = Path(__file__).parent.parent / "data" / "experience.db"


def has_chinese(text: str, threshold: float = 0.05) -> bool:
    if not text:
        return False
    chinese_count = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
    return chinese_count / len(text) > threshold


def fix_underscores(text: str) -> str:
    # Regex: underscore between letter and contraction suffix
    text = re.sub(
        r"([a-zA-Z])[\x5f\uff3f]+(t|s|d|m|ll|re|ve)(?=\W|$)",
        r"\1'\2",
        text,
    )
    # Direct replacements for edge cases
    contraction_map = {
        "n_t": "n't", "_re ": "'re ", "_ve ": "'ve ", "_ll ": "'ll ",
        "_s ": "'s ", "_d ": "'d ", "_m ": "'m ",
        "_re.": "'re.", "_ve.": "'ve.", "_ll.": "'ll.",
        "_s.": "'s.", "_s,": "'s,", "_re,": "'re,",
        "_t ": "'t ", "_s\n": "'s\n", "_re\n": "'re\n",
    }
    for old, new in contraction_map.items():
        text = text.replace(old, new)
    return text


def main():
    if not DB_PATH.exists():
        print(f"DB not found: {DB_PATH}")
        sys.exit(1)

    conn = sqlite3.connect(str(DB_PATH))
    cur = conn.cursor()

    # Count before
    cur.execute("SELECT COUNT(*) FROM episodes")
    total_before = cur.fetchone()[0]
    print(f"Total episodes before cleaning: {total_before}")

    # Step 1: Delete Chinese episodes
    cur.execute("SELECT id, assistant_response FROM episodes")
    rows = cur.fetchall()

    chinese_ids = []
    for row_id, response in rows:
        if has_chinese(response):
            chinese_ids.append(row_id)

    if chinese_ids:
        placeholders = ",".join("?" for _ in chinese_ids)
        cur.execute(f"DELETE FROM episodes WHERE id IN ({placeholders})", chinese_ids)
        conn.commit()
    print(f"Deleted {len(chinese_ids)} episodes with Chinese characters")

    # Step 2: Fix underscores in remaining episodes
    cur.execute("SELECT id, assistant_response FROM episodes")
    rows = cur.fetchall()

    fixed_count = 0
    for row_id, response in rows:
        cleaned = fix_underscores(response)
        if cleaned != response:
            cur.execute("UPDATE episodes SET assistant_response = ? WHERE id = ?", (cleaned, row_id))
            fixed_count += 1

    conn.commit()
    print(f"Fixed underscores in {fixed_count} episodes")

    # Also clean semantic_memories content
    cur.execute("SELECT rowid, content FROM semantic_memories")
    sem_rows = cur.fetchall()
    sem_chinese = 0
    sem_fixed = 0
    sem_delete_ids = []
    for rowid, content in sem_rows:
        if has_chinese(content):
            sem_delete_ids.append(rowid)
            sem_chinese += 1
        else:
            cleaned = fix_underscores(content)
            if cleaned != content:
                cur.execute("UPDATE semantic_memories SET content = ? WHERE rowid = ?", (cleaned, rowid))
                sem_fixed += 1

    if sem_delete_ids:
        placeholders = ",".join("?" for _ in sem_delete_ids)
        cur.execute(f"DELETE FROM semantic_memories WHERE rowid IN ({placeholders})", sem_delete_ids)

    conn.commit()
    print(f"Semantics: deleted {sem_chinese} Chinese, fixed {sem_fixed} underscores")

    # Final counts
    cur.execute("SELECT COUNT(*) FROM episodes")
    total_after = cur.fetchone()[0]
    cur.execute("SELECT COUNT(*) FROM semantic_memories")
    sem_after = cur.fetchone()[0]
    print(f"\nFinal: {total_after} episodes, {sem_after} semantic memories")

    conn.close()


if __name__ == "__main__":
    main()
