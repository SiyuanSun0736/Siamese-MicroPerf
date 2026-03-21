#!/bin/bash
# =============================================================================
# collect_dataset.sh — Siamese-MicroPerf 训练集采集主脚本
# =============================================================================
#
# 功能概述：
#   1. 扫描 BENCH_ROOTS 下所有含 int main() 的程序目录（同时覆盖
#      SingleSource/Benchmarks、MultiSource/Benchmarks、MultiSource/Applications）
#   2. 将目录内所有 .c/.cpp 源文件 + loop_main.c 一起编译：
#      - 含 main() 的文件用 -Dmain=bench_entry 重命名入口
#      → v1（-O3）和 v2（-O2）两个可执行文件
#   3. 标签生成：在固定时间窗口 LABEL_WINDOW 内统计各版本执行次数，
#      N_v1 > N_v2 → label=1（v1/-O3 更快），否则 label=0
#   4. PMU 采集：用 pmu_monitor 分别监控 v1 和 v2，保存 CSV 时序文件
#   5. 将 (program, label, v1_csv, v2_csv) 写入 manifest.jsonl
#
# 依赖：
#   - clang/clang++（或 gcc/g++，通过 CC/CXX 变量切换）
#   - ./pmu_monitor（已编译，位于项目根目录）
#   - sudo 权限（或 perf_event_paranoid ≤ 1）
#   - train_set/loop_main.c
#
# 用法：
#   cd /path/to/Siamese-MicroPerf
#   sudo bash train_set/collect_dataset.sh
#
# 环境变量（可覆盖默认值）：
#   CC              C 编译器（默认 clang，回退 gcc）
#   CXX             C++ 编译器（默认 clang++，回退 g++）
#   BENCH_ROOTS     空格分隔的扫描根目录（相对于 PROJECT_ROOT，默认见下）
#   OUT_DIR         编译输出目录（默认 train_set/bin）
#   DATA_DIR        PMU CSV 输出目录（默认 train_set/data）
#   LABEL_WINDOW    标签生成时间窗口，秒（默认 10）
#   PMU_WINDOW      PMU 采集时间窗口，秒（默认 30）
#   INTERVAL_MS     pmu_monitor 采样间隔，毫秒（默认 500）
#   MIN_EXECS       最小执行次数阈值，低于此值视为基准太慢跳过（默认 2）
#   MAX_SRCS        单程序最多允许的源文件数，超出则跳过（默认 50）
#
# 输出文件结构：
#   train_set/
#   ├── bin/               编译后的可执行文件
#   │   ├── Bubblesort_v1  (-O3)
#   │   ├── Bubblesort_v2  (-O2)
#   │   └── ...
#   ├── data/              PMU CSV 时序文件
#   │   ├── Bubblesort_v1.csv
#   │   ├── Bubblesort_v2.csv
#   │   └── ...
#   └── manifest.jsonl     数据集配对清单（每行一条 JSON）
# =============================================================================

set -euo pipefail

# ── 路径配置 ─────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

CC="${CC:-}"
CXX="${CXX:-}"

# 默认扫描三个根目录（SingleSource + MultiSource）
BENCH_ROOTS="${BENCH_ROOTS:-\
llvm-test-suite/SingleSource/Benchmarks \
llvm-test-suite/MultiSource/Benchmarks \
llvm-test-suite/MultiSource/Applications}"

OUT_DIR="${OUT_DIR:-$SCRIPT_DIR/bin}"
DATA_DIR="${DATA_DIR:-$SCRIPT_DIR/data}"
PMU_MONITOR="${PMU_MONITOR:-$PROJECT_ROOT/pmu_monitor}"
LOOP_MAIN="$SCRIPT_DIR/loop_main.c"
MANIFEST="$SCRIPT_DIR/manifest.jsonl"

LABEL_WINDOW="${LABEL_WINDOW:-10}"
PMU_WINDOW="${PMU_WINDOW:-30}"
INTERVAL_MS="${INTERVAL_MS:-500}"
MIN_EXECS="${MIN_EXECS:-2}"
MAX_SRCS="${MAX_SRCS:-50}"

# v1 = 更激进优化（对应 README 中 label=1 方向）
OPT_V1="${OPT_V1:--O3}"
OPT_V2="${OPT_V2:--O2}"

# ── 颜色 ─────────────────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
CYAN='\033[0;36m'; BOLD='\033[1m'; NC='\033[0m'

info()  { printf "${CYAN}[INFO]${NC}  %s\n" "$*"; }
pass()  { printf "${GREEN}[OK]${NC}    %s\n" "$*"; }
warn()  { printf "${YELLOW}[WARN]${NC}  %s\n" "$*"; }
err()   { printf "${RED}[SKIP]${NC}  %s\n" "$*"; }
bold()  { printf "${BOLD}%s${NC}\n" "$*"; }

# ── 统计计数器 ───────────────────────────────────────────────────────────────
COUNT_TOTAL=0
COUNT_OK=0
COUNT_SKIP=0

# ── 清理钩子 ─────────────────────────────────────────────────────────────────
BENCH_PID_V=""
MON_PID_V=""
cleanup() {
    [[ -n "$BENCH_PID_V" ]] && kill "$BENCH_PID_V" 2>/dev/null || true
    [[ -n "$MON_PID_V"   ]] && kill "$MON_PID_V"   2>/dev/null || true
    wait 2>/dev/null || true
}
trap cleanup EXIT INT TERM

# ── 前置检查 ─────────────────────────────────────────────────────────────────
cd "$PROJECT_ROOT"

# 自动检测 C 编译器
if [[ -z "$CC" ]]; then
    if   command -v clang   &>/dev/null; then CC=clang
    elif command -v gcc     &>/dev/null; then CC=gcc
    else echo "Error: 未找到 C 编译器，请设置 CC=" >&2; exit 1; fi
fi

# 自动检测 C++ 编译器
if [[ -z "$CXX" ]]; then
    if   command -v clang++ &>/dev/null; then CXX=clang++
    elif command -v g++     &>/dev/null; then CXX=g++
    else CXX=""; fi
fi

info "C 编译器:   $CC"
[[ -n "$CXX" ]] && info "C++ 编译器: $CXX"

# pmu_monitor 可执行文件
if [[ ! -x "$PMU_MONITOR" ]]; then
    info "pmu_monitor 未找到，尝试编译..."
    make -C "$PROJECT_ROOT" 2>&1 | tail -5
    [[ -x "$PMU_MONITOR" ]] || { echo "Error: pmu_monitor 编译失败" >&2; exit 1; }
fi

# perf_event_paranoid 检查
PARANOID=$(cat /proc/sys/kernel/perf_event_paranoid 2>/dev/null || echo "2")
if [[ "$PARANOID" -gt 1 ]] && [[ "$EUID" -ne 0 ]]; then
    warn "perf_event_paranoid=$PARANOID，建议以 root 运行或执行："
    warn "  echo 1 | sudo tee /proc/sys/kernel/perf_event_paranoid"
fi

# 目录创建
mkdir -p "$OUT_DIR" "$DATA_DIR" "log"

# ── 预编译 loop_main.o（C 和 C++ 各一份）────────────────────────────────────
LOOP_OBJ_C="$OUT_DIR/loop_main_c.o"
LOOP_OBJ_CXX=""

$CC -O2 -c "$LOOP_MAIN" -o "$LOOP_OBJ_C" 2>/dev/null \
    || { echo "Error: loop_main.c (C 模式) 编译失败" >&2; exit 1; }
info "loop_main (C):   $LOOP_OBJ_C"

if [[ -n "$CXX" ]]; then
    LOOP_OBJ_CXX="$OUT_DIR/loop_main_cxx.o"
    $CXX -O2 -x c++ -c "$LOOP_MAIN" -o "$LOOP_OBJ_CXX" 2>/dev/null \
        || { warn "loop_main.c (C++ 模式) 编译失败，C++ 程序将被跳过"; LOOP_OBJ_CXX=""; }
    [[ -n "$LOOP_OBJ_CXX" ]] && info "loop_main (C++): $LOOP_OBJ_CXX"
fi

info "扫描根目录: $BENCH_ROOTS"
info "优化对比: v1=$OPT_V1 vs v2=$OPT_V2"
info "标签窗口: ${LABEL_WINDOW}s  |  PMU 窗口: ${PMU_WINDOW}s  |  采样间隔: ${INTERVAL_MS}ms"
echo

# ── 辅助函数 ─────────────────────────────────────────────────────────────────

# 从基准输出的 stderr 中提取 BENCH_EXECUTIONS=<n>
extract_executions() {
    local output="$1"
    echo "$output" | grep -oP '(?<=BENCH_EXECUTIONS=)\d+' || echo "0"
}

# 判断目录是否包含 C++ 源文件
is_cpp_dir() {
    find "$1" -maxdepth 1 \( -name "*.cpp" -o -name "*.cxx" -o -name "*.cc" \) \
        -print -quit 2>/dev/null | grep -q .
}

# 找出目录中第一个含 int main() 的源文件（C 优先，再 C++）
find_main_file() {
    local dir="$1"
    {
        find "$dir" -maxdepth 1 -name "*.c"                                         | sort
        find "$dir" -maxdepth 1 \( -name "*.cpp" -o -name "*.cc" -o -name "*.cxx" \) | sort
    } | while IFS= read -r f; do
        grep -qP '\bint\s+main\s*\(' "$f" 2>/dev/null && echo "$f" && return 0
    done
    return 1
}

# 编译多文件基准为可执行文件
# 参数: <prog_dir> <bench_name> <output_binary> <opt_flags>
# 返回: 0 成功，非 0 失败
compile_bench() {
    local dir="$1"
    local name="$2"
    local out="$3"
    local opt_flags="$4"

    local main_file
    main_file=$(find_main_file "$dir") || return 1

    local is_cpp=0
    is_cpp_dir "$dir" && is_cpp=1

    # C++ 程序需要 loop_main C++ 版本
    if [[ "$is_cpp" -eq 1 && -z "$LOOP_OBJ_CXX" ]]; then
        warn "$name: C++ 程序但 C++ loop_main 不可用，跳过"
        return 1
    fi

    # 收集同目录下所有源文件（不递归，过滤测试文件）
    local -a srcs=()
    while IFS= read -r -d '' f; do
        srcs+=("$f")
    done < <(find "$dir" -maxdepth 1 \
        \( -name "*.c" -o -name "*.cpp" -o -name "*.cc" -o -name "*.cxx" \) \
        -not -name "*.test.*" \
        -print0 | sort -z)

    local ns="${#srcs[@]}"
    [[ "$ns" -eq 0 ]] && return 1
    if [[ "$ns" -gt "$MAX_SRCS" ]]; then
        warn "$name: 源文件过多 ($ns > $MAX_SRCS)，跳过"
        return 1
    fi

    local COMPILER LOOP_OBJ
    if [[ "$is_cpp" -eq 1 ]]; then
        COMPILER="$CXX"; LOOP_OBJ="$LOOP_OBJ_CXX"
    else
        COMPILER="$CC";  LOOP_OBJ="$LOOP_OBJ_C"
    fi

    local obj_dir="$OUT_DIR/obj_${name}_$$"
    mkdir -p "$obj_dir"

    local -a objs=()
    for src in "${srcs[@]}"; do
        local bn obj extra_flag=""
        bn=$(basename "$src")
        obj="$obj_dir/${bn%.*}.o"
        [[ "$src" == "$main_file" ]] && extra_flag="-Dmain=bench_entry"

        if ! $COMPILER $opt_flags $extra_flag \
                -I"$dir" -I"$PROJECT_ROOT/llvm-test-suite" \
                -c "$src" -o "$obj" 2>/dev/null; then
            rm -rf "$obj_dir"
            return 1
        fi
        objs+=("$obj")
    done

    if ! $COMPILER $opt_flags "${objs[@]}" "$LOOP_OBJ" -lm -lpthread -o "$out" 2>/dev/null; then
        rm -rf "$obj_dir"
        return 1
    fi

    rm -rf "$obj_dir"
    return 0
}

# 运行基准进程并附加 pmu_monitor，采集 PMU_WINDOW 秒后保存 CSV
# 参数: <binary_path> <output_csv_path>
collect_pmu() {
    local bin="$1"
    local out_csv="$2"

    BENCH_PID_V=""
    MON_PID_V=""

    "$bin" "$PMU_WINDOW" >/dev/null 2>/dev/null &
    BENCH_PID_V=$!

    sleep 0.5

    if ! kill -0 "$BENCH_PID_V" 2>/dev/null; then
        warn "基准进程已在 0.5s 内退出，PMU 采集窗口不足"
        BENCH_PID_V=""
        return 1
    fi

    "$PMU_MONITOR" "$BENCH_PID_V" -i "$INTERVAL_MS" >/dev/null 2>/dev/null &
    MON_PID_V=$!

    sleep "$PMU_WINDOW"

    kill "$BENCH_PID_V" 2>/dev/null || true
    kill "$MON_PID_V"   2>/dev/null || true
    wait "$BENCH_PID_V" 2>/dev/null || true
    wait "$MON_PID_V"   2>/dev/null || true
    BENCH_PID_V=""
    MON_PID_V=""

    sleep 0.5

    if [[ -f "log/pmu_monitor.csv" ]]; then
        cp -L "log/pmu_monitor.csv" "$out_csv"
        return 0
    else
        warn "log/pmu_monitor.csv 未生成"
        return 1
    fi
}

# ── 主循环：按程序目录扫描 ───────────────────────────────────────────────────
bold "======================================================"
bold "         开始扫描基准并采集训练数据"
bold "======================================================"

> "$MANIFEST"

while IFS= read -r PROG_DIR; do
    # 生成简短基准名：取相对路径，去掉已知公共前缀后 / 转 _
    REL="${PROG_DIR#$PROJECT_ROOT/llvm-test-suite/}"
    BENCH_NAME="${REL//\//_}"
    BENCH_NAME="${BENCH_NAME#SingleSource_Benchmarks_}"
    BENCH_NAME="${BENCH_NAME#MultiSource_Benchmarks_}"
    BENCH_NAME="${BENCH_NAME#MultiSource_Applications_}"

    ((COUNT_TOTAL++)) || true
    info "[$COUNT_TOTAL] $BENCH_NAME"
    info "  目录: $REL"

    BIN_V1="$OUT_DIR/${BENCH_NAME}_v1"
    BIN_V2="$OUT_DIR/${BENCH_NAME}_v2"

    # ── 编译 v1（-O3）────────────────────────────────────────────────────────
    if ! compile_bench "$PROG_DIR" "$BENCH_NAME" "$BIN_V1" "$OPT_V1"; then
        err "$BENCH_NAME: v1 编译失败，跳过"
        ((COUNT_SKIP++)) || true
        continue
    fi

    # ── 编译 v2（-O2）────────────────────────────────────────────────────────
    if ! compile_bench "$PROG_DIR" "${BENCH_NAME}_v2" "$BIN_V2" "$OPT_V2"; then
        err "$BENCH_NAME: v2 编译失败，跳过"
        ((COUNT_SKIP++)) || true
        continue
    fi

    # ── 标签生成：固定时间窗口内执行次数对比 ────────────────────────────────
    OUT_V1=$(timeout $((LABEL_WINDOW + 5)) "$BIN_V1" "$LABEL_WINDOW" 2>&1 || true)
    OUT_V2=$(timeout $((LABEL_WINDOW + 5)) "$BIN_V2" "$LABEL_WINDOW" 2>&1 || true)

    N_V1=$(extract_executions "$OUT_V1")
    N_V2=$(extract_executions "$OUT_V2")

    if [[ "$N_V1" -lt "$MIN_EXECS" && "$N_V2" -lt "$MIN_EXECS" ]]; then
        err "$BENCH_NAME: 执行次数过少 (v1=$N_V1, v2=$N_V2)，基准运行时间可能超过窗口，跳过"
        ((COUNT_SKIP++)) || true
        continue
    fi

    if [[ "$N_V1" -eq "$N_V2" ]]; then
        warn "$BENCH_NAME: v1=$N_V1 = v2=$N_V2，标签模糊，跳过"
        ((COUNT_SKIP++)) || true
        continue
    fi

    LABEL=$(( N_V1 > N_V2 ? 1 : 0 ))
    info "  标签: v1(${OPT_V1})=$N_V1 次  vs  v2(${OPT_V2})=$N_V2 次  → label=$LABEL"

    # ── PMU 采集 v1 ──────────────────────────────────────────────────────────
    CSV_V1="$DATA_DIR/${BENCH_NAME}_v1.csv"
    if ! collect_pmu "$BIN_V1" "$CSV_V1"; then
        err "$BENCH_NAME: v1 PMU 采集失败，跳过"
        ((COUNT_SKIP++)) || true
        continue
    fi
    info "  v1 CSV: ${CSV_V1#$PROJECT_ROOT/}"

    # ── PMU 采集 v2 ──────────────────────────────────────────────────────────
    CSV_V2="$DATA_DIR/${BENCH_NAME}_v2.csv"
    if ! collect_pmu "$BIN_V2" "$CSV_V2"; then
        err "$BENCH_NAME: v2 PMU 采集失败，跳过"
        ((COUNT_SKIP++)) || true
        continue
    fi
    info "  v2 CSV: ${CSV_V2#$PROJECT_ROOT/}"

    # ── 写入 manifest ─────────────────────────────────────────────────────────
    printf '{"program":"%s","label":%d,"n_v1":%d,"n_v2":%d,"opt_v1":"%s","opt_v2":"%s","v1_csv":"%s","v2_csv":"%s","source":"%s"}\n' \
        "$BENCH_NAME" "$LABEL" "$N_V1" "$N_V2" \
        "$OPT_V1" "$OPT_V2" \
        "${CSV_V1#$PROJECT_ROOT/}" \
        "${CSV_V2#$PROJECT_ROOT/}" \
        "$REL" \
        >> "$MANIFEST"

    pass "$BENCH_NAME 完成"
    ((COUNT_OK++)) || true
    echo

done < <(
    for ROOT_REL in $BENCH_ROOTS; do
        ROOT_ABS="$PROJECT_ROOT/$ROOT_REL"
        [[ -d "$ROOT_ABS" ]] || { warn "目录不存在: $ROOT_ABS"; continue; }

        find "$ROOT_ABS" -type f \
            \( -name "*.c" -o -name "*.cpp" -o -name "*.cc" -o -name "*.cxx" \) \
            -not -path "*/CMakeFiles/*" \
            -not -path "*/.git/*" \
            | while IFS= read -r src; do
                grep -lP '\bint\s+main\s*\(' "$src" 2>/dev/null || true
            done \
            | xargs -I{} dirname {} \
            | sort -u
    done | sort -u
)

# ── 汇总 ─────────────────────────────────────────────────────────────────────
echo
bold "======================================================"
printf "  总计: %d  |  成功: %d  |  跳过: %d\n" \
    "$COUNT_TOTAL" "$COUNT_OK" "$COUNT_SKIP"
echo "  数据集清单: $MANIFEST"
bold "======================================================"
