#!/bin/bash
cd "$(dirname "$0")"
cd char_level
python3 -W ignore generate_sentence.py
