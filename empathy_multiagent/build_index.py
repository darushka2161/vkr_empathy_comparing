# build_index.py
# Скрипт построения FAISS-индекса для EmpathyMAS-C (и empathy_rag).
# Запускать один раз перед первым использованием EmpathyMAS-C.
#
# Usage: python build_index.py [--cache-dir retriever_cache]

import os
os.environ["OMP_NUM_THREADS"] = "1"

import argparse
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

from src.load_dataset import prepare_examples
from architectures.empathy_rag import EmpathyRetriever


def build(cache_dir: str = "retriever_cache"):
    cache_path = Path(cache_dir)

    if (cache_path / "examples.pkl").exists():
        print(f"Index already exists at '{cache_dir}'. Nothing to do.")
        print("To rebuild, delete the directory and run again.")
        return

    print("Loading EmpatheticDialogues train set...")
    train_examples = prepare_examples("train")
    print(f"Building FAISS index for {len(train_examples)} examples...")
    retriever = EmpathyRetriever(train_examples)
    retriever.save(cache_dir)
    print(f"\nDone. Index saved to '{cache_dir}'.")
    print("You can now run: python run_experiment.py --arch empathy_mas_c --limit 10")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build FAISS retriever index for EmpathyMAS-C")
    parser.add_argument("--cache-dir", default="retriever_cache",
                        help="Directory to save the FAISS index (default: retriever_cache)")
    args = parser.parse_args()
    build(args.cache_dir)
