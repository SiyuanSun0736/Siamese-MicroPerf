#!/bin/bash
# =============================================================================
# bolt_optimize.sh — 对 train_set/bin/O2-bolt/ 中的预编译二进制执行 BOLT 优化
# =============================================================================
#
# 流程（对每个基准）：
#   阶段 A  perf record (LBR) 采样，运行时长 PMU_WINDOW 秒
#   阶段 B  perf2bolt  将 perf.data 转换为 bolt.fdata
#   阶段 C  llvm-bolt  应用 profile 生成优化后二进制
#
# 前提：
#   - 二进制已用 -Wl,--emit-relocs -no-pie 编译（见 cmake/caches/O2-bolt.cmake）
#   - llvm-bolt / perf2bolt 已安装并在 PATH 中
#   - perf 支持 LBR（Intel CPU 或 AMD Zen3+）
#     如果不支持 LBR，设置 USE_INSTRUMENTATION=1 退回到插桩模式
#   - sudo 权限（或 perf_event_paranoid ≤ 1）
#
# 用法：
#   cd /path/to/Siamese-MicroPerf
#   bash train_set/bolt_optimize.sh            # 默认处理 O2-bolt
#   VARIANT=O3-bolt bash train_set/bolt_optimize.sh
#   DRYRUN=1       bash train_set/bolt_optimize.sh   # 试运行：只打印命令
#
# 环境变量：
#   VARIANT               源变体目录名（默认 O2-bolt）
#   SRC_VARIANT           等同 VARIANT（别名）
#   OUT_VARIANT           输出变体目录名（默认 O2-bolt-opt）
#   BIN_DIR               输入二进制目录（默认 train_set/bin/<VARIANT>）
#   OUT_DIR               输出二进制目录（默认 train_set/bin/<OUT_VARIANT>）
#   TEST_DIR              测试规格目录（默认 train_set/test/<VARIANT>）
#   PROFILE_DIR           perf/fdata 临时目录（默认 train_set/bolt_profiles）
#   PMU_WINDOW            perf record 采样时长，秒（默认 30）
#   USE_INSTRUMENTATION   1=插桩模式（无 LBR），0=LBR 模式（默认 0）
#   OVERWRITE             1=覆盖已有优化二进制，0=已存在则跳过（默认 0）
#   DRYRUN                1=只打印命令不执行（默认 0）
#   BOLT_EXTRA_FLAGS      追加到 llvm-bolt 的额外参数（默认空）
# =============================================================================

set -euo pipefail

# ── 路径配置 ─────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

VARIANT="${VARIANT:-O2-bolt}"
OUT_VARIANT="${OUT_VARIANT:-}"       # 依赖 VARIANT，在 getopts 后填充
BIN_DIR="${BIN_DIR:-}"               # 依赖 VARIANT，在 getopts 后填充
OUT_DIR="${OUT_DIR:-}"               # 依赖 OUT_VARIANT，在 getopts 后填充
TEST_DIR="${TEST_DIR:-}"             # 依赖 VARIANT，在 getopts 后填充
OUT_TEST_DIR="${OUT_TEST_DIR:-}"     # 依赖 OUT_VARIANT，在 getopts 后填充
PROFILE_DIR="${PROFILE_DIR:-}"       # 依赖 VARIANT，在 getopts 后填充

PMU_WINDOW="${PMU_WINDOW:-30}"
USE_INSTRUMENTATION="${USE_INSTRUMENTATION:-0}"
OVERWRITE="${OVERWRITE:-0}"
DRYRUN="${DRYRUN:-0}"
BOLT_EXTRA_FLAGS="${BOLT_EXTRA_FLAGS:-}"

# ── 参数解析 ─────────────────────────────────────────────────────────────────
usage() {
    echo "用法: $0 [-v VARIANT] [-V OUT_VARIANT] [-b BIN_DIR] [-o OUT_DIR]"
    echo "         [-t TEST_DIR] [-T OUT_TEST_DIR] [-p PROFILE_DIR]"
    echo "         [-w PMU_WINDOW] [-I] [-r] [-n]"
    echo "选项:"
    echo "  -v VARIANT      源变体名称（默认 O2-bolt）"
    echo "  -V OUT_VARIANT  输出变体名称（默认 <VARIANT>-opt）"
    echo "  -b BIN_DIR      输入二进制目录"
    echo "  -o OUT_DIR      输出二进制目录"
    echo "  -t TEST_DIR     测试规格目录"
    echo "  -T OUT_TEST_DIR 输出测试目录"
    echo "  -p PROFILE_DIR  perf/fdata 临时目录"
    echo "  -w PMU_WINDOW   采样时长，秒（默认 30）"
    echo "  -I              使用插桩模式（无 LBR）"
    echo "  -r              覆盖已有优化二进制"
    echo "  -n              DRYRUN 模式"
    echo "  -h              显示帮助"
    exit 0
}

while getopts "v:V:b:o:t:T:p:w:Irnh" opt; do
    case $opt in
        v) VARIANT="$OPTARG" ;;
        V) OUT_VARIANT="$OPTARG" ;;
        b) BIN_DIR="$OPTARG" ;;
        o) OUT_DIR="$OPTARG" ;;
        t) TEST_DIR="$OPTARG" ;;
        T) OUT_TEST_DIR="$OPTARG" ;;
        p) PROFILE_DIR="$OPTARG" ;;
        w) PMU_WINDOW="$OPTARG" ;;
        I) USE_INSTRUMENTATION=1 ;;
        r) OVERWRITE=1 ;;
        n) DRYRUN=1 ;;
        h) usage ;;
        *) echo "未知选项：-$OPTARG" >&2; exit 1 ;;
    esac
done

# 用最终 VARIANT / OUT_VARIANT 填充未显式指定的依赖项
[[ -z "$OUT_VARIANT"  ]] && OUT_VARIANT="${VARIANT}-opt"
[[ -z "$BIN_DIR"      ]] && BIN_DIR="$SCRIPT_DIR/bin/$VARIANT"
[[ -z "$OUT_DIR"      ]] && OUT_DIR="$SCRIPT_DIR/bin/$OUT_VARIANT"
[[ -z "$TEST_DIR"     ]] && TEST_DIR="$SCRIPT_DIR/test/$VARIANT"
[[ -z "$OUT_TEST_DIR" ]] && OUT_TEST_DIR="$SCRIPT_DIR/test/$OUT_VARIANT"
[[ -z "$PROFILE_DIR"  ]] && PROFILE_DIR="$SCRIPT_DIR/bolt_profiles/$VARIANT"

# ── 颜色 ─────────────────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
CYAN='\033[0;36m'; BOLD='\033[1m'; NC='\033[0m'

info()  { printf "${CYAN}[INFO]${NC}  %s\n" "$*"; }
pass()  { printf "${GREEN}[OK]${NC}    %s\n" "$*"; }
warn()  { printf "${YELLOW}[WARN]${NC}  %s\n" "$*"; }
err()   { printf "${RED}[FAIL]${NC}  %s\n" "$*"; }
bold()  { printf "${BOLD}%s${NC}\n" "$*"; }

# ── dry_run 包装：DRYRUN=1 时只打印，否则执行 ────────────────────────────────
run_cmd() {
    if [[ "$DRYRUN" -eq 1 ]]; then
        printf "${YELLOW}  [DRYRUN]${NC} %s\n" "$*"
    else
        eval "$@"
    fi
}

# ── 从 .test 文件解析 RUN: 命令（复用 collect_dataset_testbench.sh 约定）────
# 参数: $1=test_file  $2=binary_path  $3=test_data_dir
# 输出: 展开后的命令（stdout）
parse_run_cmd() {
    local test_file="$1" binary="$2" test_data="$3"
    local run_raw cmd_part first_word

    run_raw=$(grep '^RUN:' "$test_file" | head -1 | sed 's/^RUN: //') || return 1
    [[ -z "$run_raw" ]] && return 1

    cmd_part="$run_raw"
    if [[ "$run_raw" == "cd %S ;"* ]]; then
        cmd_part="${run_raw#cd %S ; }"
    fi

    first_word=$(echo "$cmd_part" | awk '{print $1}')
    if [[ "$first_word" == "%S/"* ]]; then
        local bin_ref_name="${first_word#%S/}"
        cmd_part=$(printf '%s' "$cmd_part" \
            | sed "s|%S/${bin_ref_name}|${binary}|g")
    fi

    cmd_part=$(printf '%s' "$cmd_part" | sed "s|%S|${test_data}|g")
    printf '%s' "$cmd_part"
}

# ── 清理钩子 ─────────────────────────────────────────────────────────────────
_BENCH_PID=""
_CLEANUP_DONE=0
cleanup() {
    [[ "$_CLEANUP_DONE" -eq 1 ]] && return
    _CLEANUP_DONE=1
    [[ -n "$_BENCH_PID" ]] && kill "$_BENCH_PID" 2>/dev/null || true
}
trap 'cleanup; exit 130' INT
trap 'cleanup; exit 143' TERM
trap cleanup EXIT

# ── 前置检查 ─────────────────────────────────────────────────────────────────
cd "$PROJECT_ROOT"

[[ -d "$BIN_DIR" ]]  || { echo "Error: BIN_DIR 不存在: $BIN_DIR"   >&2; exit 1; }
[[ -d "$TEST_DIR" ]] || { echo "Error: TEST_DIR 不存在: $TEST_DIR" >&2; exit 1; }

if ! command -v llvm-bolt &>/dev/null; then
    echo "Error: llvm-bolt 未找到，请确认已安装并在 PATH 中" >&2; exit 1
fi
if ! command -v perf2bolt &>/dev/null; then
    echo "Error: perf2bolt 未找到，请确认已安装并在 PATH 中" >&2; exit 1
fi
if [[ "$USE_INSTRUMENTATION" -eq 0 ]] && ! command -v perf &>/dev/null; then
    echo "Error: perf 未找到，请安装 linux-perf 或设置 USE_INSTRUMENTATION=1" >&2; exit 1
fi

[[ "$DRYRUN" -eq 1 ]] || mkdir -p "$OUT_DIR" "$OUT_TEST_DIR" "$PROFILE_DIR"

# ── 统计 ─────────────────────────────────────────────────────────────────────
COUNT_TOTAL=0; COUNT_OK=0; COUNT_SKIP=0; COUNT_FAIL=0

bold "======================================================"
bold "     BOLT 优化：$VARIANT  →  $OUT_VARIANT"
bold "======================================================"
info "BIN_DIR:      $BIN_DIR"
info "OUT_DIR:      $OUT_DIR"
info "TEST_DIR:     $TEST_DIR"
info "OUT_TEST_DIR: $OUT_TEST_DIR"
info "PROFILE_DIR:  $PROFILE_DIR"
info "采样时长:     ${PMU_WINDOW}s"
info "采样模式:     $( [[ $USE_INSTRUMENTATION -eq 1 ]] && echo '插桩 (instrumentation)' || echo 'LBR (perf record)')"
info "覆盖已有:     $( [[ $OVERWRITE -eq 1 ]] && echo '是' || echo '否')"
[[ -n "$BOLT_EXTRA_FLAGS" ]] && info "BOLT 额外参数: $BOLT_EXTRA_FLAGS"
[[ "$DRYRUN" -eq 1 ]] && warn "★ DRYRUN 模式：仅打印命令，不执行 ★"
echo

# ── 主处理循环 ───────────────────────────────────────────────────────────────
for test_subdir in "$TEST_DIR"/*/; do
    [[ -d "$test_subdir" ]] || continue
    bench_name="$(basename "$test_subdir")"
    ((COUNT_TOTAL++)) || true

    info "[$COUNT_TOTAL] 处理: $bench_name"

    # ── 查找可执行文件 ────────────────────────────────────────────────────────
    binary="$BIN_DIR/${bench_name}_${VARIANT}"
    if [[ ! -x "$binary" ]]; then
        err "可执行文件不存在: $binary，跳过"
        ((COUNT_SKIP++)) || true; echo; continue
    fi

    # ── 查找 .test 文件 ───────────────────────────────────────────────────────
    test_file=$(find "$test_subdir" -maxdepth 1 -name "*.test" | head -1)
    if [[ -z "$test_file" ]]; then
        err "未找到 .test 文件，跳过"
        ((COUNT_SKIP++)) || true; echo; continue
    fi

    bench_cmd=$(parse_run_cmd "$test_file" "$binary" "$test_subdir") || {
        err ".test 文件中未找到 RUN: 行，跳过"
        ((COUNT_SKIP++)) || true; echo; continue
    }

    info "  CMD:    $bench_cmd"
    info "  SOURCE: ${binary#$PROJECT_ROOT/}"

    out_binary="$OUT_DIR/${bench_name}_${OUT_VARIANT}"
    prof_dir="$PROFILE_DIR/$bench_name"
    perf_data="$prof_dir/perf.data"
    fdata="$prof_dir/bolt.fdata"
    inst_binary="$prof_dir/${bench_name}.inst"
    inst_fdata="$prof_dir/inst.fdata"

    # ── 跳过已存在的优化二进制 ───────────────────────────────────────────────
    if [[ "$OVERWRITE" -eq 0 && -x "$out_binary" && "$DRYRUN" -eq 0 ]]; then
        info "  已存在优化二进制，跳过（OVERWRITE=0）"
        pass "$bench_name  [跳过]"
        ((COUNT_SKIP++)) || true; echo; continue
    fi

    [[ "$DRYRUN" -eq 1 ]] || mkdir -p "$prof_dir"

    # =========================================================================
    # 阶段 A：采样
    # =========================================================================
    if [[ "$USE_INSTRUMENTATION" -eq 0 ]]; then
        # ── A-1：LBR 硬件采样 ─────────────────────────────────────────────────
        info "  [A] LBR 采样 (${PMU_WINDOW}s)..."
        # 将 benchmark 在后台循环运行，同时用 perf record 采样
        (
            cd "$test_subdir"
            while true; do
                eval "$bench_cmd" >/dev/null 2>&1 || true
            done
        ) &
        _BENCH_PID=$!

        # 等进程稳定启动
        sleep 0.3

        run_cmd "perf record -e cycles:u -j any,u -m 256M -o '$perf_data' \
            -p '$_BENCH_PID' -- sleep '${PMU_WINDOW}'" \
            "2>&1 | grep -v '^\\[' || true"

        kill "$_BENCH_PID" 2>/dev/null || true
        wait "$_BENCH_PID" 2>/dev/null || true
        _BENCH_PID=""

        if [[ "$DRYRUN" -eq 0 && ! -f "$perf_data" ]]; then
            err "perf.data 未生成，跳过"
            ((COUNT_FAIL++)) || true; echo; continue
        fi

        # ── A-2：perf2bolt 转换 ───────────────────────────────────────────────
        info "  [B] perf2bolt 转换..."
        run_cmd "perf2bolt -p '$perf_data' -o '$fdata' '$binary' 2>&1 || true"

        profile_arg="-data '$fdata'"

    else
        # ── A-3：插桩模式（无 LBR 时回退）────────────────────────────────────
        info "  [A] 生成插桩二进制..."
        run_cmd "llvm-bolt '$binary' -instrument -o '$inst_binary'"

        info "  [A] 运行插桩二进制 (${PMU_WINDOW}s)..."
        (
            cd "$test_subdir"
            BOLT_FDATA_FILE="$inst_fdata"
            export BOLT_FDATA_FILE
            while true; do
                eval "$(echo "$bench_cmd" | sed "s|${binary}|${inst_binary}|g")" \
                    >/dev/null 2>&1 || true
            done
        ) &
        _BENCH_PID=$!
        sleep "$PMU_WINDOW"
        kill "$_BENCH_PID" 2>/dev/null || true
        wait "$_BENCH_PID" 2>/dev/null || true
        _BENCH_PID=""

        if [[ "$DRYRUN" -eq 0 && ! -f "$inst_fdata" ]]; then
            err "插桩 fdata 未生成，跳过"
            ((COUNT_FAIL++)) || true; echo; continue
        fi

        profile_arg="-data '$inst_fdata'"
    fi

    # =========================================================================
    # 阶段 C：llvm-bolt 优化
    # =========================================================================
    info "  [C] llvm-bolt 优化 → ${out_binary#$PROJECT_ROOT/}"
    run_cmd "llvm-bolt '$binary' \
        ${profile_arg} \
        -o '$out_binary' \
        -reorder-blocks=ext-tsp \
        -reorder-functions=hfsort \
        -split-functions \
        -split-all-cold \
        -split-eh \
        -dyno-stats \
        -plt=hot \
        ${BOLT_EXTRA_FLAGS} \
        2>&1"

    if [[ "$DRYRUN" -eq 0 ]]; then
        if [[ -x "$out_binary" ]]; then
            pass "$bench_name  →  ${out_binary#$PROJECT_ROOT/}"
            ((COUNT_OK++)) || true
            # ── 复制测试文件到 OUT_TEST_DIR ──────────────────────────────────
            out_test_subdir="$OUT_TEST_DIR/$bench_name"
            if [[ -d "$test_subdir" ]]; then
                cp -r "$test_subdir" "$out_test_subdir"
                info "  测试文件已复制 → ${out_test_subdir#$PROJECT_ROOT/}"
            fi
        else
            err "llvm-bolt 未生成可执行文件: $out_binary"
            ((COUNT_FAIL++)) || true
        fi
    else
        run_cmd "cp -r '$test_subdir' '$OUT_TEST_DIR/$bench_name'"
        pass "$bench_name  [DRYRUN OK]"
        ((COUNT_OK++)) || true
    fi
    echo
done

# ── 汇总 ─────────────────────────────────────────────────────────────────────
bold "======================================================"
bold "汇总：总计 $COUNT_TOTAL  |  成功 $COUNT_OK  |  跳过 $COUNT_SKIP  |  失败 $COUNT_FAIL"
bold "======================================================"
[[ "$DRYRUN" -eq 0 && $COUNT_OK -gt 0 ]] && {
    info "优化二进制已写入: $OUT_DIR"
    info "测试文件已复制到: $OUT_TEST_DIR"
}
