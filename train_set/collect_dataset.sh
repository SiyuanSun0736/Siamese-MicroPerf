#!/bin/bash
# =============================================================================
# collect_dataset.sh — Siamese-MicroPerf 训练集采集主脚本
# =============================================================================
#
# 功能概述：
#   1. 从 LLVM test suite SingleSource/Benchmarks 中发现所有 .c 基准
#   2. 分别用 -O3（v1）和 -O2（v2）编译，链接 loop_main.c 使基准可循环运行
#   3. 标签生成：在固定时间窗口 LABEL_WINDOW 内统计各版本执行次数，
#      N_v1 > N_v2 → label=1（v1/O3 更快），否则 label=0
#   4. PMU 采集：用 pmu_monitor 分别监控 v1 和 v2，保存 CSV 时序文件
#   5. 将 (program, label, v1_csv, v2_csv) 写入 manifest.jsonl
#
# 依赖：
#   - clang（或 gcc，通过 CC 变量切换）
#   - ./pmu_monitor（已编译，位于项目根目录）
#   - sudo 权限（或 perf_event_paranoid ≤ 1）
#   - train_set/loop_main.c
#
# 用法：
#   cd /path/to/Siamese-MicroPerf
#   sudo bash train_set/collect_dataset.sh
#
# 环境变量（可覆盖默认值）：
#   CC              编译器（默认 clang，回退 gcc）
#   BENCH_DIR       基准源文件根目录（默认 llvm-test-suite/SingleSource/Benchmarks）
#   OUT_DIR         编译输出目录（默认 train_set/bin）
#   DATA_DIR        PMU CSV 输出目录（默认 train_set/data）
#   LABEL_WINDOW    标签生成时间窗口，秒（默认 10）
#   PMU_WINDOW      PMU 采集时间窗口，秒（默认 30）
#   INTERVAL_MS     pmu_monitor 采样间隔，毫秒（默认 500）
#   MIN_EXECS       最小执行次数阈值，低于此值视为基准太慢跳过（默认 2）
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
BENCH_DIR="${BENCH_DIR:-$PROJECT_ROOT/llvm-test-suite/SingleSource/Benchmarks}"
OUT_DIR="${OUT_DIR:-$SCRIPT_DIR/bin}"
DATA_DIR="${DATA_DIR:-$SCRIPT_DIR/data}"
PMU_MONITOR="${PMU_MONITOR:-$PROJECT_ROOT/pmu_monitor}"
LOOP_MAIN="$SCRIPT_DIR/loop_main.c"
MANIFEST="$SCRIPT_DIR/manifest.jsonl"

LABEL_WINDOW="${LABEL_WINDOW:-10}"
PMU_WINDOW="${PMU_WINDOW:-30}"
INTERVAL_MS="${INTERVAL_MS:-500}"
MIN_EXECS="${MIN_EXECS:-2}"

# v1 = 更激进优化（对应 README 中 label=1 方向）
OPT_V1="${OPT_V1:--O3}"
OPT_V2="${OPT_V2:--O2}"

# ── 颜色 ─────────────────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
CYAN='\033[0;36m'; NC='\033[0m'

info()  { printf "${CYAN}[INFO]${NC}  %s\n" "$*"; }
pass()  { printf "${GREEN}[OK]${NC}    %s\n" "$*"; }
warn()  { printf "${YELLOW}[WARN]${NC}  %s\n" "$*"; }
err()   { printf "${RED}[SKIP]${NC}  %s\n" "$*"; }

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

# 自动检测编译器
if [[ -z "$CC" ]]; then
    if command -v clang &>/dev/null; then
        CC=clang
    elif command -v gcc &>/dev/null; then
        CC=gcc
    else
        echo "Error: neither clang nor gcc found. Set CC= manually." >&2
        exit 1
    fi
fi
info "编译器: $CC"

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

# ── 预编译 loop_main.o（不带 -Dmain=bench_entry，避免符号冲突）────────────────
LOOP_OBJ="$OUT_DIR/loop_main.o"
if ! $CC -O2 -c "$LOOP_MAIN" -o "$LOOP_OBJ" 2>/dev/null; then
    echo "Error: loop_main.c 编译失败" >&2; exit 1
fi
info "loop_main.o 已预编译: $LOOP_OBJ"

info "基准目录: $BENCH_DIR"
info "优化对比: v1=$OPT_V1 vs v2=$OPT_V2"
info "标签窗口: ${LABEL_WINDOW}s  |  PMU 窗口: ${PMU_WINDOW}s  |  采样间隔: ${INTERVAL_MS}ms"
echo

# ── 辅助函数 ─────────────────────────────────────────────────────────────────

# 从基准输出的 stderr 中提取 BENCH_EXECUTIONS=<n>
extract_executions() {
    local output="$1"
    echo "$output" | grep -oP '(?<=BENCH_EXECUTIONS=)\d+' || echo "0"
}

# 运行基准进程并附加 pmu_monitor，采集 PMU_WINDOW 秒后保存 CSV
# 参数: <binary_path> <output_csv_path>
collect_pmu() {
    local bin="$1"
    local out_csv="$2"

    BENCH_PID_V=""
    MON_PID_V=""

    # 启动基准（忽略 stdout；stderr 写入 /dev/null 避免干扰）
    "$bin" "$PMU_WINDOW" >/dev/null 2>/dev/null &
    BENCH_PID_V=$!

    # 等待进程启动
    sleep 0.5

    # 检查进程是否还活着（防止极快基准已退出）
    if ! kill -0 "$BENCH_PID_V" 2>/dev/null; then
        warn "基准进程已在 0.5s 内退出，PMU 采集窗口不足"
        BENCH_PID_V=""
        return 1
    fi

    # 启动 pmu_monitor，附加到基准 PID
    "$PMU_MONITOR" "$BENCH_PID_V" -i "$INTERVAL_MS" >/dev/null 2>/dev/null &
    MON_PID_V=$!

    # 等待 PMU 窗口结束
    sleep "$PMU_WINDOW"

    # 终止两个进程
    kill "$BENCH_PID_V" 2>/dev/null || true
    kill "$MON_PID_V"   2>/dev/null || true
    wait "$BENCH_PID_V" 2>/dev/null || true
    wait "$MON_PID_V"   2>/dev/null || true
    BENCH_PID_V=""
    MON_PID_V=""

    # 等待文件系统刷新
    sleep 0.5

    # 复制最新生成的 CSV（pmu_monitor.csv 软链接指向最新文件）
    if [[ -f "log/pmu_monitor.csv" ]]; then
        cp -L "log/pmu_monitor.csv" "$out_csv"
        return 0
    else
        warn "log/pmu_monitor.csv 未生成"
        return 1
    fi
}

# ── 主循环：遍历所有 .c 基准 ──────────────────────────────────────────────────
echo "======================================================"
echo "         开始扫描基准并采集训练数据"
echo "======================================================"

# 清空旧 manifest（追加模式），若为全新运行则重建
> "$MANIFEST"

while IFS= read -r -d '' SRC; do
    BENCH_NAME=$(basename "$SRC" .c)
    ((COUNT_TOTAL++)) || true

    info "[$COUNT_TOTAL] $BENCH_NAME  ($SRC)"

    # ── 编译 v1 和 v2 ────────────────────────────────────────────────────────
    BIN_V1="$OUT_DIR/${BENCH_NAME}_v1"
    BIN_V2="$OUT_DIR/${BENCH_NAME}_v2"

    # 两步编译：基准 .o 用 -Dmain=bench_entry，loop_main.o 独立编译后链接
    if ! $CC $OPT_V1 -Dmain=bench_entry -c "$SRC" -o "$BIN_V1.o" 2>/dev/null ||
       ! $CC "$BIN_V1.o" "$LOOP_OBJ" -lm -o "$BIN_V1" 2>/dev/null; then
        rm -f "$BIN_V1.o"
        err "$BENCH_NAME: v1 编译失败，跳过"
        ((COUNT_SKIP++)) || true
        continue
    fi
    rm -f "$BIN_V1.o"
    if ! $CC $OPT_V2 -Dmain=bench_entry -c "$SRC" -o "$BIN_V2.o" 2>/dev/null ||
       ! $CC "$BIN_V2.o" "$LOOP_OBJ" -lm -o "$BIN_V2" 2>/dev/null; then
        rm -f "$BIN_V2.o"
        err "$BENCH_NAME: v2 编译失败，跳过"
        ((COUNT_SKIP++)) || true
        continue
    fi
    rm -f "$BIN_V2.o"

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

    if [[ "$N_V1" -gt "$N_V2" ]]; then
        LABEL=1
    else
        LABEL=0
    fi

    info "  标签: v1(${OPT_V1})=$N_V1 次  vs  v2(${OPT_V2})=$N_V2 次  → label=$LABEL"

    # ── PMU 采集 v1 ──────────────────────────────────────────────────────────
    CSV_V1="$DATA_DIR/${BENCH_NAME}_v1.csv"
    if ! collect_pmu "$BIN_V1" "$CSV_V1"; then
        err "$BENCH_NAME: v1 PMU 采集失败，跳过"
        ((COUNT_SKIP++)) || true
        continue
    fi
    info "  v1 CSV: $CSV_V1"

    # ── PMU 采集 v2 ──────────────────────────────────────────────────────────
    CSV_V2="$DATA_DIR/${BENCH_NAME}_v2.csv"
    if ! collect_pmu "$BIN_V2" "$CSV_V2"; then
        err "$BENCH_NAME: v2 PMU 采集失败，跳过"
        ((COUNT_SKIP++)) || true
        continue
    fi
    info "  v2 CSV: $CSV_V2"

    # ── 写入 manifest ─────────────────────────────────────────────────────────
    printf '{"program":"%s","label":%d,"n_v1":%d,"n_v2":%d,"opt_v1":"%s","opt_v2":"%s","v1_csv":"%s","v2_csv":"%s"}\n' \
        "$BENCH_NAME" "$LABEL" "$N_V1" "$N_V2" \
        "$OPT_V1" "$OPT_V2" \
        "${CSV_V1#$PROJECT_ROOT/}" \
        "${CSV_V2#$PROJECT_ROOT/}" \
        >> "$MANIFEST"

    pass "$BENCH_NAME 完成"
    ((COUNT_OK++)) || true
    echo

done < <(find "$BENCH_DIR" -name "*.c" -not -name "CMakeLists*" -print0 | sort -z)

# ── 汇总 ─────────────────────────────────────────────────────────────────────
echo "======================================================"
printf "  总计: %d  |  成功: %d  |  跳过: %d\n" \
    "$COUNT_TOTAL" "$COUNT_OK" "$COUNT_SKIP"
echo "  数据集清单: $MANIFEST"
echo "======================================================"
