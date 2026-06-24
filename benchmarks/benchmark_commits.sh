#!/bin/bash
# Automated benchmarking across lazybgen commits.
# Builds a Docker image per commit (lazybgen built from that commit) and runs
# the reader benchmark inside it against the shared test_data.
#
# Usage:
#   ./benchmark_commits.sh [options] <commit> [<commit> ...]
#   ./benchmark_commits.sh --file commits.txt
#
# Run from the repository root.

set -euo pipefail

PARALLEL_BUILDS=2
OUTPUT_DIR="benchmark_results"
DOCKERFILE="benchmarks/Dockerfile.benchmark"
DATA_DIR="$(pwd)/benchmarks/test_data"
LOG_DIR="benchmark_logs"
FORCE_REBUILD=false
NUM_RUNS=3
MODE="standard"
REGION_SIZE="medium"
PROFILE_MODE=false

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'
print_status()  { echo -e "${GREEN}[$(date '+%H:%M:%S')]${NC} $1"; }
print_error()   { echo -e "${RED}[$(date '+%H:%M:%S')] ERROR:${NC} $1" >&2; }
print_warning() { echo -e "${YELLOW}[$(date '+%H:%M:%S')] WARN:${NC} $1"; }

usage() {
    cat <<EOF
Usage: $0 [OPTIONS] [COMMITS...]

Benchmark multiple lazybgen commits via Docker.

OPTIONS:
    -f, --file FILE        Read commits from file (one per line)
    -p, --parallel NUM     Parallel builds (default: $PARALLEL_BUILDS)
    -o, --output-dir DIR   Results dir (default: $OUTPUT_DIR)
    -d, --data-dir DIR     Test data dir (default: $DATA_DIR)
    -n, --num-runs NUM     Measured runs per workload (default: $NUM_RUNS)
    -m, --mode MODE        quick|standard|compression|scaling|comprehensive (default: $MODE)
    -r, --region-size SZ   small|medium|large (default: $REGION_SIZE)
    --dockerfile FILE      Dockerfile (default: $DOCKERFILE)
    --force-rebuild        Rebuild images even if present
    --profile              Profiling mode (cProfile + perf); implies one run
    -h, --help             Show this help

EXAMPLES:
    # initial-extraction baseline vs HEAD
    $0 dd01e34 07505b6
    ./compare_results.py benchmark_results/benchmark_*.json
EOF
    exit 1
}

COMMITS=()
while [[ $# -gt 0 ]]; do
    case $1 in
        -f|--file)
            [[ -f "$2" ]] || { print_error "File not found: $2"; exit 1; }
            while IFS= read -r c; do [[ -z "$c" || "$c" =~ ^# ]] && continue; COMMITS+=("$c"); done < "$2"; shift 2 ;;
        -p|--parallel)    PARALLEL_BUILDS="$2"; shift 2 ;;
        -o|--output-dir)  OUTPUT_DIR="$2"; shift 2 ;;
        -d|--data-dir)    DATA_DIR="$2"; shift 2 ;;
        -n|--num-runs)    NUM_RUNS="$2"; shift 2 ;;
        -m|--mode)        MODE="$2"; shift 2 ;;
        -r|--region-size) REGION_SIZE="$2"; shift 2 ;;
        --dockerfile)     DOCKERFILE="$2"; shift 2 ;;
        --force-rebuild)  FORCE_REBUILD=true; shift ;;
        --profile)        PROFILE_MODE=true; shift ;;
        -h|--help)        usage ;;
        -*)               print_error "Unknown option: $1"; usage ;;
        *)                COMMITS+=("$1"); shift ;;
    esac
done

[[ ${#COMMITS[@]} -eq 0 ]] && { print_error "No commits specified"; usage; }
[[ -d "$DATA_DIR" ]] || { print_error "Test data not found: $DATA_DIR"; exit 1; }
mkdir -p "$OUTPUT_DIR" "$LOG_DIR"

build_image() {
    local commit=$1
    local tag="lazybgen:benchmark-${commit:0:8}"
    local log_file="$LOG_DIR/build_${commit:0:8}.log"
    print_status "Building image for $commit ..."
    if [[ "$FORCE_REBUILD" != "true" ]] && docker image inspect "$tag" >/dev/null 2>&1; then
        print_warning "Image $tag exists, skipping (use --force-rebuild)"; return 0
    fi
    local repo_root; repo_root=$(git rev-parse --show-toplevel 2>/dev/null || pwd)
    if (cd "$repo_root" && docker build --build-arg GIT_COMMIT="$commit" \
            -f "$DOCKERFILE" -t "$tag" . > "$log_file" 2>&1); then
        print_status "Built $tag"; return 0
    else
        print_error "Build failed for $commit (see $log_file)"; return 1
    fi
}

run_benchmark() {
    local commit=$1
    local tag="lazybgen:benchmark-${commit:0:8}"
    local log_file="$LOG_DIR/run_${commit:0:8}.log"
    print_status "Running benchmark for $commit ..."
    docker image inspect "$tag" >/dev/null 2>&1 || { print_error "Image $tag missing"; return 1; }
    local repo_root; repo_root=$(git rev-parse --show-toplevel 2>/dev/null || pwd)
    local abs_output_dir; abs_output_dir=$(cd "$OUTPUT_DIR" && pwd)

    local docker_cmd=(
        docker run --rm
        -v "$abs_output_dir:/results"
        -v "$DATA_DIR:/data/test_data:ro"
        -v "$repo_root/benchmarks:/app/benchmarks:ro"
        -e GIT_COMMIT="$commit"
    )
    [[ "$PROFILE_MODE" == "true" ]] && docker_cmd+=(--cap-add=SYS_ADMIN)
    docker_cmd+=("$tag" python /app/benchmarks/run_benchmark.py
        --data-dir /data/test_data --output-dir /results
        --num-runs "$NUM_RUNS" --mode "$MODE" --region-size "$REGION_SIZE")
    [[ "$PROFILE_MODE" == "true" ]] && docker_cmd+=(--profile)

    if "${docker_cmd[@]}" > "$log_file" 2>&1; then
        print_status "Done $commit"; return 0
    else
        print_error "Benchmark failed for $commit (see $log_file)"; return 1
    fi
}

print_status "Benchmarking ${#COMMITS[@]} commit(s): ${COMMITS[*]}"
echo "  mode=$MODE region=$REGION_SIZE runs=$NUM_RUNS profile=$PROFILE_MODE data=$DATA_DIR"

failed_builds=(); failed_runs=()
for commit in "${COMMITS[@]}"; do
    build_image "$commit" || failed_builds+=("$commit")
done

successful_builds=()
for commit in "${COMMITS[@]}"; do
    if docker image inspect "lazybgen:benchmark-${commit:0:8}" >/dev/null 2>&1; then
        successful_builds+=("$commit")
    fi
done
print_status "Built ${#successful_builds[@]}/${#COMMITS[@]} images"

for commit in "${successful_builds[@]}"; do
    run_benchmark "$commit" || failed_runs+=("$commit")
    sleep 1
done

echo ""
print_status "Done. Results in $OUTPUT_DIR, logs in $LOG_DIR"
[[ ${#failed_builds[@]} -gt 0 ]] && { print_error "Failed builds: ${failed_builds[*]}"; }
[[ ${#failed_runs[@]}   -gt 0 ]] && { print_error "Failed runs: ${failed_runs[*]}"; }
print_status "Compare with: ./benchmarks/compare_results.py $OUTPUT_DIR/benchmark_*.json"
[[ ${#failed_builds[@]} -gt 0 || ${#failed_runs[@]} -gt 0 ]] && exit 1 || exit 0
