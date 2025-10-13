#!/usr/bin/env python3
# tests/test_rag_code_ingestion.py

import os
import sys
import glob
import argparse
import importlib
from pathlib import Path
from typing import Iterable, List, Optional

# Load env early
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import exa23ai_rag_agent as rag  # type: ignore


class TestRAGCodeIngestion:
    """Test suite for RAG agent code file ingestion and inference"""

    # default test table
    test_table = "SLACKBOT_CODE_VECTORS_TEST"
    original_table = None
    keep_table = True  # default: keep table unless teardown/cleanup called

    # ingestion knobs
    MAX_CHUNK_SIZE = 3500
    MAX_CHUNKS_PER_FILE = 200
    SKIP_EXTS = {".xml", ".json"}  # skip these text formats explicitly

    @classmethod
    def setup_class(cls) -> None:
        """Setup test environment"""
        cls.original_table = os.environ.get("ORA_TABLE")

        # Force test table
        os.environ["ORA_TABLE"] = cls.test_table

        # IMPORTANT: reload rag so it reads the new ORA_TABLE
        importlib.reload(rag)

        # Reset RAG globals
        rag._CONN = None
        rag._EMB = None
        rag._VS = None
        rag._RETRIEVER = None
        rag._QA = None

        # Initialize
        result = rag.init_once()
        assert result.get("ok"), f"RAG initialization failed: {result}"

        print(f"[TEST] Using test table: {cls.test_table}")

    @classmethod
    def teardown_class(cls) -> None:
        """Drop the test table and close connections iff keep_table == False"""
        if cls.keep_table:
            # restore env and leave resources intact
            if cls.original_table:
                os.environ["ORA_TABLE"] = cls.original_table
            else:
                os.environ.pop("ORA_TABLE", None)
            return

        print("[CLEANUP] Starting teardown (dropping table and closing connection)")
        try:
            if rag._CONN:
                cursor = rag._CONN.cursor()
                try:
                    cursor.execute(f"DROP TABLE {cls.test_table}")
                    print(f"[CLEANUP] ? Dropped table {cls.test_table}")
                except Exception as e:
                    msg = str(e)
                    if "ORA-00942" in msg:
                        print(f"[CLEANUP] Table {cls.test_table} doesn't exist")
                    else:
                        print(f"[CLEANUP] Failed to drop table: {e}")
                cursor.close()
        except Exception as e:
            print(f"[CLEANUP] Error during table cleanup: {e}")

        try:
            if rag._CONN:
                rag._CONN.close()
                print("[CLEANUP] ? Closed Oracle connection")
        except Exception as e:
            print(f"[CLEANUP] Error closing connection: {e}")

        # Reset rag state
        rag._CONN = None
        rag._EMB = None
        rag._VS = None
        rag._RETRIEVER = None
        rag._QA = None

        # Restore original ORA_TABLE
        if cls.original_table:
            os.environ["ORA_TABLE"] = cls.original_table
        else:
            os.environ.pop("ORA_TABLE", None)

        print("[CLEANUP] ? Teardown complete")

    # ---------- Helpers ----------

    @staticmethod
    def is_text_file(path: str, probe_bytes: int = 2048) -> bool:
        """
        Heuristic: treat as binary if it contains NUL or too many non-printables.
        """
        try:
            with open(path, "rb") as f:
                chunk = f.read(probe_bytes)
            if b"\x00" in chunk:
                return False
            # ratio of printable-ish bytes
            printable = sum(32 <= b <= 126 or b in (9, 10, 13) for b in chunk)
            # if less than 85% printable, assume binary
            return (len(chunk) == 0) or (printable / max(1, len(chunk)) >= 0.85)
        except Exception:
            return False

    @staticmethod
    def _expand_inputs(file_patterns: Iterable[str]) -> List[str]:
        """
        Expand patterns into a list of file paths, recursing into directories.
        """
        files: List[str] = []
        seen = set()

        def add_file(p: str):
            ap = os.path.abspath(p)
            if ap not in seen:
                seen.add(ap)
                files.append(ap)

        for patt in file_patterns:
            if os.path.isfile(patt):
                add_file(patt)
                continue

            matches = glob.glob(patt)
            if not matches and os.path.isdir(patt):
                # exact dir
                matches = [patt]

            for m in matches:
                if os.path.isdir(m):
                    # walk recursively
                    for root, _, fnames in os.walk(m):
                        for fn in fnames:
                            add_file(os.path.join(root, fn))
                elif os.path.isfile(m):
                    add_file(m)
        return files

    def cleanup_table_data(self) -> None:
        """DELETE rows but keep the table (schema & indexes intact)."""
        try:
            if rag._CONN:
                cursor = rag._CONN.cursor()
                cursor.execute(f"DELETE FROM {self.test_table}")
                rag._CONN.commit()
                cursor.close()
                print(f"[CLEANUP] ? Cleared all data from {self.test_table}")
        except Exception as e:
            print(f"[CLEANUP] Error clearing table data: {e}")

    def get_table_stats(self) -> int:
        """Return COUNT(*) from test table."""
        try:
            if rag._CONN:
                cursor = rag._CONN.cursor()
                cursor.execute(f"SELECT COUNT(*) FROM {self.test_table}")
                count = int(cursor.fetchone()[0])
                cursor.close()
                print(f"[INFO] Table {self.test_table} contains {count} records")
                return count
        except Exception as e:
            print(f"[INFO] Error getting table stats: {e}")
        return 0

    # ---------- Tests ----------

    def test_do_ingest(self, file_patterns: Optional[Iterable[str]] = None) -> int:
        """Ingest code/text files with directory recursion, exclusions, and size guard."""
        print("\n[TEST] Starting code file ingestion...")

        if file_patterns is None:
            file_patterns = [
                "oss/test/*",
                "oss/oeda/test/*",
            ]

        all_files = self._expand_inputs(file_patterns)
        print(f"[TEST] Total inputs expanded (pre-filter): {len(all_files)}")

        if not all_files:
            print("[WARN] No files found matching the patterns")
            return 0

        ingested_count = 0
        for file_path in all_files:
            try:
                # skip directories (shouldn't happen after expand, but be safe)
                if os.path.isdir(file_path):
                    continue

                ext = Path(file_path).suffix.lower()
                if ext in self.SKIP_EXTS:
                    print(f"[SKIP] Excluded extension {ext}: {file_path}")
                    continue

                if not self.is_text_file(file_path):
                    print(f"[SKIP] Binary/non-text detected: {file_path}")
                    continue

                # read
                with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read()

                if not content.strip():
                    print(f"[SKIP] Empty file: {file_path}")
                    continue

                # chunking
                max_chunk = self.MAX_CHUNK_SIZE
                if len(content) <= max_chunk:
                    chunks = [content]
                else:
                    chunks = [content[i:i + max_chunk] for i in range(0, len(content), max_chunk)]

                # guard: skip if would exceed MAX_CHUNKS_PER_FILE
                if len(chunks) > self.MAX_CHUNKS_PER_FILE:
                    print(f"[SKIP] {Path(file_path).name} too large ({len(chunks)} chunks > {self.MAX_CHUNKS_PER_FILE})")
                    continue

                if len(chunks) == 1:
                    doc_id = f"code_file_{Path(file_path).stem}_{hash(file_path) % 10000}"
                    metadata = {
                        "file_path": file_path,
                        "file_type": Path(file_path).suffix,
                        "file_name": Path(file_path).name,
                        "content_type": "code",
                    }
                    result = rag.upsert_text(doc_id, chunks[0], **metadata)
                    if result.get("ok"):
                        ingested_count += 1
                        print(f"[TEST] ? Ingested: {Path(file_path).name}")
                    else:
                        print(f"[WARN] Failed to ingest {file_path}: {result}")
                else:
                    print(f"[TEST] Splitting {Path(file_path).name} into {len(chunks)} chunks")
                    chunk_success = 0
                    for idx, chunk in enumerate(chunks):
                        doc_id = f"code_file_{Path(file_path).stem}_chunk{idx}_{hash(file_path) % 10000}"
                        metadata = {
                            "file_path": file_path,
                            "file_type": Path(file_path).suffix,
                            "file_name": Path(file_path).name,
                            "content_type": "code",
                            "chunk_index": idx,
                            "total_chunks": len(chunks),
                        }
                        result = rag.upsert_text(doc_id, chunk, **metadata)
                        if result.get("ok"):
                            chunk_success += 1

                    if chunk_success > 0:
                        ingested_count += 1
                        print(f"[TEST] ? Ingested: {Path(file_path).name} ({chunk_success}/{len(chunks)} chunks)")
                    else:
                        print(f"[WARN] Failed to ingest any chunks from {file_path}")

            except Exception as e:
                print(f"[ERROR] Failed to process {file_path}: {e}")
                continue

        print(f"[TEST] Successfully ingested { ingested_count }/{ len(all_files) } files")
        assert ingested_count > 0, "No files were successfully ingested"

        # Optional light query to verify pipeline
        if ingested_count > 0:
            try:
                test_result = rag.query("exascale volumes", k=2)
                if "answer" not in test_result:
                    print(f"[WARN] Test query failed: {test_result}")
            except Exception as e:
                print(f"[WARN] Error performing test query: {e}")

        return ingested_count

    def test_infer(self, queries=None, output_dir="/tmp"):
        """Unchanged from your original (omitted here for brevity)?keep your version"""
        # You can paste your existing test_infer implementation here
        raise NotImplementedError("test_infer unchanged; keep your existing implementation.")

    def test_health_check(self):
        """Verify RAG system health"""
        print("\n[TEST] Running health check...")

        health_status = rag.health()

        assert health_status.get("oracle_connected"), "Oracle connection failed"
        assert health_status.get("initialized"), "RAG system not initialized"
        assert health_status.get("table") == self.test_table, f"Wrong table: {health_status.get('table')}"

        print(f"[TEST] ? Health check passed")
        print(f"[TEST]   Table: {health_status.get('table')}")
        print(f"[TEST]   Model: {health_status.get('embed_model')}")

    # ----------- Simple REPL -----------

    def interactive_repl(self, k: int = 4):
        print("\n[REPL] Initialising interactive session...")
        TestRAGCodeIngestion.setup_class()
        try:
            print(f"[TEST] Using test table: {self.test_table}")
            while True:
                q = input("Query (type 'exit' to quit)> ").strip()
                if q.lower() in ("exit", "quit", ":q"):
                    break
                if not q:
                    continue
                try:
                    result = rag.query(q, k=k)
                    if "answer" in result:
                        print("\n--- ANSWER ---\n")
                        print(result["answer"])
                        print("\n--------------\n")
                    else:
                        print(f"[ERROR] Query failed: {result}")
                except Exception as e:
                    print(f"[ERROR] Query failed: {e}")
        finally:
            # keep table by default
            pass


# -------- CLI --------

def main():
    parser = argparse.ArgumentParser(description="RAG code ingestion test harness")
    sub = parser.add_subparsers(dest="cmd")

    # health
    sub.add_parser("health")

    # ingest (alias do_ingest)
    p_ing = sub.add_parser("ingest")
    p_ing.add_argument("--patterns", nargs="+", required=False,
                       help="File patterns or directories (recursive). Default: oss/test/* oss/oeda/test/*")

    p_ing2 = sub.add_parser("do_ingest")
    p_ing2.add_argument("--patterns", nargs="+", required=False)

    # repl
    p_repl = sub.add_parser("interactive")
    p_repl.add_argument("-k", type=int, default=4)

    # delete rows but keep table
    sub.add_parser("cleanup_table_data")

    # drop table + close connection
    sub.add_parser("teardown")

    args = parser.parse_args()
    test_inst = TestRAGCodeIngestion()

    if args.cmd is None:
        parser.print_help()
        sys.exit(1)

    if args.cmd == "health":
        TestRAGCodeIngestion.setup_class()
        try:
            test_inst.test_health_check()
        finally:
            # keep table by default
            pass
        return 0

    if args.cmd in ("ingest", "do_ingest"):
        TestRAGCodeIngestion.setup_class()
        try:
            patterns = args.patterns or ["oss/test/*", "oss/oeda/test/*"]
            test_inst.test_do_ingest(patterns)
        finally:
            # keep table by default
            pass
        return 0

    if args.cmd == "interactive":
        test_inst.interactive_repl(k=args.k)
        return 0

    if args.cmd == "cleanup_table_data":
        TestRAGCodeIngestion.setup_class()
        try:
            test_inst.cleanup_table_data()
        finally:
            pass
        return 0

    if args.cmd == "teardown":
        # signal teardown to drop table
        TestRAGCodeIngestion.keep_table = False
        TestRAGCodeIngestion.setup_class()
        try:
            pass
        finally:
            TestRAGCodeIngestion.teardown_class()
        return 0

    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())

