#!/usr/bin/env python3

import os
import random


def main():
    for i in range(100):
        # Build the main blob
        blob = bytearray()

        # Fill the tensor data with some random bytes
        for _ in range(5000):
            blob.append(random.randint(0, 255))

        # Write the seed file
        corpus_dir = "corpus"
        os.makedirs(corpus_dir, exist_ok=True)
        path = os.path.join(corpus_dir, f"seed{i}.bin")
        with open(path, "wb") as f:
            f.write(blob)
        print(f"Generated {path}")


if __name__ == "__main__":
    main()
