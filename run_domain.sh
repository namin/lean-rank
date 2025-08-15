#!/usr/bin/env bash
set -euo pipefail

# -----------------------------------------------------------------------------
# Domain-Specific Lean Rank Pipeline
# -----------------------------------------------------------------------------
# Usage: ./run_domain.sh <domain>
# Example: ./run_domain.sh number_theory
# Available domains: number_theory, topology, algebra, analysis, category_theory, order

if [[ $# -ne 1 ]]; then
    echo "Usage: $0 <domain>"
    echo "Available domains: number_theory, topology, algebra, analysis, category_theory, order"
    exit 1
fi

DOMAIN="$1"
PYTHON_BIN="${PYTHON_BIN:-$(python -c 'import sys; print(sys.executable)')}"

# Validate domain
case "$DOMAIN" in
    number_theory|topology|algebra|analysis|category_theory|order)
        echo "==> Running pipeline for domain: $DOMAIN"
        ;;
    *)
        echo "ERROR: Unknown domain '$DOMAIN'"
        echo "Available domains: number_theory, topology, algebra, analysis, category_theory, order"
        exit 1
        ;;
esac

# Set up directories
BASE_DATA_DIR="${BASE_DATA_DIR:-data}"
DOMAIN_DATA_DIR="${BASE_DATA_DIR}/${DOMAIN}_filtered"
DOMAIN_PROC_DIR="${DOMAIN_DATA_DIR}/processed"
DOMAIN_OUT_DIR="outputs/${DOMAIN}_filtered"

# Check if we have base data
if [[ ! -f "${BASE_DATA_DIR}/premises.txt" ]]; then
    echo "ERROR: Missing ${BASE_DATA_DIR}/premises.txt"
    echo "Please run the data generation first (see README.md)"
    exit 1
fi

# Check if we should force regeneration
FORCE="${FORCE:-0}"

# Step 1: Filter the data to the domain (if needed)
echo ""
# Check all three required files
if [[ "$FORCE" == "1" || ! -f "${DOMAIN_DATA_DIR}/premises.txt" || ! -f "${DOMAIN_DATA_DIR}/declaration_types.txt" || ! -f "${DOMAIN_DATA_DIR}/declaration_structures.jsonl" ]]; then
    if [[ "$FORCE" == "1" ]]; then
        echo "==> Step 1: Force regenerating filtered data for $DOMAIN domain..."
    else
        echo "==> Step 1: Filtering data to $DOMAIN domain..."
        # Show which files are missing
        [[ ! -f "${DOMAIN_DATA_DIR}/premises.txt" ]] && echo "    Missing: premises.txt"
        [[ ! -f "${DOMAIN_DATA_DIR}/declaration_types.txt" ]] && echo "    Missing: declaration_types.txt"  
        [[ ! -f "${DOMAIN_DATA_DIR}/declaration_structures.jsonl" ]] && echo "    Missing: declaration_structures.jsonl"
    fi
    echo "    Output directory: $DOMAIN_DATA_DIR"
    
    "$PYTHON_BIN" -m src.tasks.filter_domain \
        --domain "$DOMAIN" \
        --input "$BASE_DATA_DIR" \
        --output "$DOMAIN_DATA_DIR"
    
    # Check filtering succeeded
    if [[ ! -f "${DOMAIN_DATA_DIR}/premises.txt" ]]; then
        echo "ERROR: Domain filtering failed - no premises.txt generated"
        exit 1
    fi
else
    echo "==> Step 1: Using existing filtered data for $DOMAIN domain"
    echo "    Directory: $DOMAIN_DATA_DIR"
    echo "    Files: premises.txt, declaration_types.txt, declaration_structures.jsonl"
    echo "    (Use FORCE=1 ./run_domain.sh $DOMAIN to regenerate)"
fi

echo ""
echo "==> Step 2: Running full pipeline on filtered $DOMAIN data..."
echo "    Data directory: $DOMAIN_DATA_DIR"
echo "    Output directory: $DOMAIN_OUT_DIR"

# Run the main walkthrough with domain-specific directories
# No TARGET_PREFIXES needed - we're working with pre-filtered data
DATA_DIR="$DOMAIN_DATA_DIR" \
PROC_DIR="$DOMAIN_PROC_DIR" \
OUT_DIR="$DOMAIN_OUT_DIR" \
TARGET_PREFIXES="" \
./run_walkthrough.sh

echo ""
echo "==> Domain-specific pipeline complete!"
echo ""
echo "Results available in:"
echo "  - Filtered data: $DOMAIN_DATA_DIR/"
echo "  - Processed data: $DOMAIN_PROC_DIR/"
echo "  - Models/outputs: $DOMAIN_OUT_DIR/"
echo ""
echo "Key outputs:"
echo "  - Rankings: ${DOMAIN_PROC_DIR}/rankings_explained.csv"
echo "  - What-if report: ${DOMAIN_PROC_DIR}/whatif.md"
echo "  - Productivity scores: See terminal output from Step 7"