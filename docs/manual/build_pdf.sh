#!/bin/sh
# Build rx_performance_manual.pdf from the HTML source with headless Chrome.
set -eu
cd "$(dirname "$0")"
CHROME="${CHROME:-/Applications/Google Chrome.app/Contents/MacOS/Google Chrome}"
"$CHROME" --headless --disable-gpu --no-pdf-header-footer \
  --print-to-pdf=rx_performance_manual.pdf rx_performance_manual.html
echo "rx_performance_manual.pdf"
