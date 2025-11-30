#!/bin/bash

# Download and extract Google Fonts data
# This will download ~1.6GB and extract to ~2GB+

set -e

OUTPUT_DIR="${1:-./google-fonts}"

echo "=== Google Fonts Downloader ==="
echo "Output directory: $OUTPUT_DIR"
echo ""

# Create output directory
mkdir -p "$OUTPUT_DIR"
cd "$OUTPUT_DIR"

# Download
echo "Downloading Google Fonts (~1GB)..."
curl -L -o fonts.zip https://github.com/google/fonts/archive/main.zip

# Unzip
echo "Extracting..."
unzip -q fonts.zip

# Move contents up one level (removes fonts-main wrapper folder)
mv fonts-main/* .
rmdir fonts-main
rm fonts.zip

# Count fonts
TTF_COUNT=$(find . -name "*.ttf" | wc -l)
echo ""
echo "Done! Found $TTF_COUNT .ttf files"
echo "Fonts are in: $OUTPUT_DIR"
