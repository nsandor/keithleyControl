#!/usr/bin/env python3
"""
unscramble_array.py

Unscramble a 10×10 sensor array CSV using a wiring key mapping original indices
(1–100, row‑major, top‑left to bottom‑right) to their *scrambled* (observed)
indices, or vice‑versa.

INPUTS
=======
1. Scrambled data CSV (default interpretation):
   A 10×10 grid (100 values) in row‑major order *as acquired* with the incorrect wiring.
   Accepts optional headers; non‑numeric cells are ignored.

2. Mapping (key) CSV:
   Two integer columns. By default we assume the *first* column is the ORIGINAL
   (intended) index and the *second* column is the NEW / SCRAMBLED index actually
   observed after wiring error.

   Format examples (with or without header):
       original,new
       1,37
       2,5
       ...

   or whitespace separated:
       1 37
       2 5

OUTPUT
======
A reconstructed 10×10 CSV in correct original spatial order.

USAGE
======
    python unscramble_array.py \
        --data scrambled.csv \
        --map key.csv \
        --out unscrambled.csv

ADVANCED
========
If your mapping file is reversed (first column = scrambled, second = original),
auto‑detection will attempt to catch this. Use --columns to force:
   --columns orig,new        (default)
   --columns new,orig        (explicit reversal)

Optionally output both forms (matrix grid and a long index/value list) with
--long-out.

VALIDATION
==========
Script checks for:
  * Duplicate indices
  * Missing indices (must cover 1..100)
  * Shape mismatches

EXIT STATUS
===========
 0 success; non‑zero on error.
"""
from __future__ import annotations
import argparse
import csv
import sys
from pathlib import Path
from typing import List, Tuple, Dict

NUM_PIXELS = 100
SIDE = 10

class MappingError(Exception):
    pass

def read_numeric_grid_csv(path: Path) -> List[float]:
    """Read up to NUM_PIXELS numeric values from a CSV presumed to contain the scrambled grid.
    Non‑numeric tokens are skipped. Returns a flat list length == NUM_PIXELS.
    """
    values: List[float] = []
    with path.open(newline='') as f:
        reader = csv.reader(f)
        for row in reader:
            for cell in row:
                if cell.strip() == '':
                    continue
                try:
                    v = float(cell)
                except ValueError:
                    # Ignore non-numeric (header or text)
                    continue
                values.append(v)
                if len(values) == NUM_PIXELS:
                    break
            if len(values) == NUM_PIXELS:
                break
    if len(values) != NUM_PIXELS:
        raise MappingError(f"Data file {path} yielded {len(values)} numeric values; expected {NUM_PIXELS} (10x10).")
    return values

def read_mapping_csv(path: Path) -> List[Tuple[int,int]]:
    """Read a two‑column mapping file. Returns list of (col1, col2) ints.
    Skips non‑integer rows (e.g., header)."""
    pairs: List[Tuple[int,int]] = []
    with path.open(newline='') as f:
        dialect = csv.Sniffer().sniff(f.read(4096), delimiters=",;\t ")
        f.seek(0)
        reader = csv.reader(f, dialect)
        for row in reader:
            if len(row) < 2:
                continue
            a, b = row[0].strip(), row[1].strip()
            try:
                pairs.append((int(a), int(b)))
            except ValueError:
                # Probably a header
                continue
    if not pairs:
        raise MappingError(f"Mapping file {path} has no integer pairs.")
    return pairs

def build_index_maps(pairs: List[Tuple[int,int]], columns: str) -> Tuple[Dict[int,int], Dict[int,int]]:
    """Given raw (a,b) pairs and a column role spec ('orig,new' or 'new,orig'),
    build forward and reverse dictionaries:
      orig_to_new[original] = scrambled_index
      new_to_orig[scrambled] = original
    Performs validation (coverage 1..100, no duplicates)."""
    if columns not in ("orig,new", "new,orig"):
        raise MappingError("--columns must be either 'orig,new' or 'new,orig'")
    orig_to_new: Dict[int,int] = {}
    new_to_orig: Dict[int,int] = {}
    for a,b in pairs:
        if columns == "orig,new":
            orig, new = a, b
        else:
            orig, new = b, a
        if not (1 <= orig <= NUM_PIXELS and 1 <= new <= NUM_PIXELS):
            raise MappingError(f"Index out of range in mapping: ({orig},{new})")
        if orig in orig_to_new:
            raise MappingError(f"Duplicate original index {orig}")
        if new in new_to_orig:
            raise MappingError(f"Duplicate new (scrambled) index {new}")
        orig_to_new[orig] = new
        new_to_orig[new] = orig
    # Coverage check
    missing_orig = sorted(set(range(1,NUM_PIXELS+1)) - orig_to_new.keys())
    missing_new  = sorted(set(range(1,NUM_PIXELS+1)) - new_to_orig.keys())
    if missing_orig:
        raise MappingError(f"Mapping missing original indices: {missing_orig[:10]}{'...' if len(missing_orig)>10 else ''}")
    if missing_new:
        raise MappingError(f"Mapping missing new/scrambled indices: {missing_new[:10]}{'...' if len(missing_new)>10 else ''}")
    return orig_to_new, new_to_orig

def auto_detect_columns(pairs: List[Tuple[int,int]]) -> str:
    """Heuristic: If first elements are a permutation with many in natural order, assume orig,new.
    If second column looks more ordered, maybe reversed. This is just a hint; user can override."""
    first_in_place = sum(1 for i,(a,_) in enumerate(pairs, start=1) if a == i)
    second_in_place = sum(1 for i,(_,b) in enumerate(pairs, start=1) if b == i)
    # If second column matches sequence more than first, guess reversed.
    if second_in_place > first_in_place and second_in_place - first_in_place > 5:
        return "new,orig"
    return "orig,new"

def unscramble(scrambled_values: List[float], orig_to_new: Dict[int,int]) -> List[float]:
    """Return list of length NUM_PIXELS in *original* order.
    We know: orig_to_new[original] = scrambled_index.
    Scrambled values are 0-based list; indices are 1-based, so adjust.
    So: value for original i = scrambled_values[ orig_to_new[i] - 1 ]"""
    output = [0.0]*NUM_PIXELS
    for orig in range(1, NUM_PIXELS+1):
        scrambled_idx = orig_to_new[orig]
        output[orig-1] = scrambled_values[scrambled_idx-1]
    return output

def write_grid_csv(values: List[float], out_path: Path):
    if len(values) != NUM_PIXELS:
        raise ValueError("Values length must be 100")
    with out_path.open('w', newline='') as f:
        writer = csv.writer(f)
        for r in range(SIDE):
            row_vals = values[r*SIDE:(r+1)*SIDE]
            writer.writerow(row_vals)

def write_long_csv(values: List[float], out_path: Path):
    with out_path.open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["original_index", "value"])
        for i,v in enumerate(values, start=1):
            writer.writerow([i,v])

def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Unscramble a 10x10 sensor array CSV using a mapping key.")
    p.add_argument('--data', required=True, type=Path, help='Scrambled data CSV file (10x10 values).')
    p.add_argument('--map', required=True, dest='mapfile', type=Path, help='Mapping CSV file (two columns).')
    p.add_argument('--out', required=True, type=Path, help='Output CSV path for unscrambled 10x10 grid.')
    p.add_argument('--long-out', type=Path, help='Optional output CSV listing (original_index,value).')
    p.add_argument('--columns', choices=['auto','orig,new','new,orig'], default='auto',
                   help="Column meaning in mapping file: 'orig,new' (default), 'new,orig', or 'auto'.")
    p.add_argument('--print-map', action='store_true', help='Print resolved mapping summary and exit.')
    return p.parse_args(argv)

def main(argv=None) -> int:
    args = parse_args(argv)
    try:
        scrambled = read_numeric_grid_csv(args.data)
        raw_pairs = read_mapping_csv(args.mapfile)
        columns = args.columns
        if columns == 'auto':
            columns = auto_detect_columns(raw_pairs)
        orig_to_new, new_to_orig = build_index_maps(raw_pairs, columns)
        if args.print_map:
            print(f"Mapping interpretation: {columns}")
            print("original -> scrambled")
            for i in range(1, NUM_PIXELS+1):
                print(f"{i:3d} -> {orig_to_new[i]:3d}")
            return 0
        unscrambled = unscramble(scrambled, orig_to_new)
        write_grid_csv(unscrambled, args.out)
        if args.long_out:
            write_long_csv(unscrambled, args.long_out)
        print(f"Unscrambled grid written to {args.out}")
        if args.long_out:
            print(f"Long form written to {args.long_out}")
        return 0
    except MappingError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2
    except Exception as e:
        print(f"Unhandled error: {e}", file=sys.stderr)
        return 1

if __name__ == '__main__':  # pragma: no cover
    sys.exit(main())
