"""Command-line entry for building the highD master frame table."""
from builder import HighDDataBuilder
from config import NUM_WORKERS, OUTPUT_PARQUET_DIR, RAW_DATA_DIR, TEST_MODE, TEST_RECORDINGS


def main() -> None:
    builder = HighDDataBuilder(RAW_DATA_DIR, OUTPUT_PARQUET_DIR, num_workers=NUM_WORKERS)
    if TEST_MODE:
        print(f"TEST MODE ON. Only processing recordings: {TEST_RECORDINGS}")
        builder.process_all_recordings(TEST_RECORDINGS)
    else:
        builder.process_all_recordings(list(range(1, 61)))


if __name__ == "__main__":
    main()

