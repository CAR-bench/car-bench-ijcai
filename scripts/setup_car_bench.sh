#!/bin/bash
# Clone the external CAR-bench repository required by the evaluator.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
CAR_BENCH_DIR="$PROJECT_ROOT/third_party/car-bench"
CAR_BENCH_REF="${CAR_BENCH_REF:-6bc1900750e561e79831dd6a8e757ac37080ba0f}"

if [ -d "$CAR_BENCH_DIR" ]; then
    echo "car-bench already exists at $CAR_BENCH_DIR"
    echo "To re-download, remove the directory first: rm -rf $CAR_BENCH_DIR"
    exit 0
fi

mkdir -p "$(dirname "$CAR_BENCH_DIR")"

echo "Cloning car-bench repository at $CAR_BENCH_REF..."
git clone https://github.com/CAR-bench/car-bench.git "$CAR_BENCH_DIR"
git -C "$CAR_BENCH_DIR" checkout --detach "$CAR_BENCH_REF"


echo ""
echo "✅ Setup complete! car-bench is ready at:"
echo "   $CAR_BENCH_DIR"
echo ""
echo "📝 Note: Tasks and mock data are automatically loaded from HuggingFace"
