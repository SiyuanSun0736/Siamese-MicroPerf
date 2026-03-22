#!/bin/bash
# =============================================================================
# collect_dataset_testbench.sh — 基于预编译可执行文件 + .test 文件的 PMU 采集脚本
# =============================================================================
#
# 功能概述：
#   使用 train_set/test/<VARIANT>/ 目录下的 .test 文件作为运行规格，
#   对 train_set/bin/<VARIANT>/ 中预编译的可执行文件执行 PMU 采集。
#   只执行 RUN: 行，忽略 VERIFY: 行。
#
# .test 文件格式（LLVM 测试套件规范）：
#   RUN: cd %S ; %S/<binary> [args...]
#   VERIFY: ...（忽略）
#
#   %S  = 测试数据目录（即 .test 文件所在目录）
#   %b  = 构建目录（忽略，仅出现在 VERIFY: 行）
#   %o  = 输出文件（忽略，仅出现在 VERIFY: 行）
#
# 二进制文件命名规则：
#   $BIN_DIR/<bench_name>_<VARIANT>
#   例如：train_set/bin/O1-g/aha_O1-g
#
# 依赖：
#   - ./pmu_monitor（已编译，且 pmu_counters.c 中 pe.inherit=1）
#   - sudo 权限（或 perf_event_paranoid ≤ 1）
#
# 用法：
#   cd /path/to/Siamese-MicroPerf
#   sudo bash train_set/collect_dataset_testbench.sh
#   sudo bash train_set/collect_dataset_testbench.sh  # 默认 VARIANT=O1-g
#   VARIANT=O2-g bash train_set/collect_dataset_testbench.sh
#
# 环境变量（可覆盖默认值）：
#   VARIANT       变体名称（默认 O1-g），决定 bin/test 子目录名
#   BIN_DIR       可执行文件目录（默认 train_set/bin/<VARIANT>）
#   TEST_DIR      测试规格目录（默认 train_set/test/<VARIANT>）
#   DATA_DIR      PMU CSV 输出目录（默认 train_set/data/<VARIANT>）
#   PMU_WINDOW    PMU 采集时间窗口，秒（默认 30）
#   INTERVAL_MS   pmu_monitor 采样间隔，毫秒（默认 500）
#   CONTINUOUS    持续循环模式：1=无限循环直到 Ctrl+C，0=跑一遍退出（默认 0）
#   OVERWRITE     是否覆盖已有数据：1=总是重新采集，0=已有足够数据则跳过（默认 1）
#   MIN_ROWS      最少有效数据行数，不足则重试（默认 50，约 30s/500ms×0.83）
#   RETRY_MAX     每个基准最多重试次数（默认 3）
#   DRYRUN        干跑模式：1=只解析并打印命令，不执行任何采集（默认 0）
#
# 输出文件结构：
#   train_set/
#   ├── data/<VARIANT>/        PMU CSV 时序文件
#   │   ├── aha_O1-g.csv
#   │   └── ...
#   └── manifest_<VARIANT>.jsonl   数据集清单（每行一条 JSON）
# =============================================================================

set -euo pipefail

# ── 路径配置 ─────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

VARIANT="${VARIANT:-O1-g}"
BIN_DIR="${BIN_DIR:-$SCRIPT_DIR/bin/$VARIANT}"
TEST_DIR="${TEST_DIR:-$SCRIPT_DIR/test/$VARIANT}"
DATA_DIR="${DATA_DIR:-$SCRIPT_DIR/data/$VARIANT}"
PMU_MONITOR="${PMU_MONITOR:-$PROJECT_ROOT/pmu_monitor}"
MANIFEST="${MANIFEST:-$SCRIPT_DIR/manifest_${VARIANT}.jsonl}"

PMU_WINDOW="${PMU_WINDOW:-30}"
INTERVAL_MS="${INTERVAL_MS:-500}"
CONTINUOUS="${CONTINUOUS:-0}"
OVERWRITE="${OVERWRITE:-1}"
MIN_ROWS="${MIN_ROWS:-50}"
RETRY_MAX="${RETRY_MAX:-3}"
DRYRUN="${DRYRUN:-0}"

# ── 颜色 ─────────────────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
CYAN='\033[0;36m'; BOLD='\033[1m'; NC='\033[0m'

info()  { printf "${CYAN}[INFO]${NC}  %s\n" "$*"; }
pass()  { printf "${GREEN}[OK]${NC}    %s\n" "$*"; }
warn()  { printf "${YELLOW}[WARN]${NC}  %s\n" "$*"; }
err()   { printf "${RED}[SKIP]${NC}  %s\n" "$*"; }
bold()  { printf "${BOLD}%s${NC}\n" "$*"; }
retry() { printf "${YELLOW}[RETRY]${NC} %s\n" "$*"; }

# ── 统计计数器 ───────────────────────────────────────────────────────────────
COUNT_TOTAL=0
COUNT_OK=0
COUNT_SKIP=0
ROUND=0      # 持续循环轮次

declare -A BENCH_LOOP_PIDS  # 持续模式：基准名 → 工作负载循环 PID（复用）

# ── 清理钩子 ─────────────────────────────────────────────────────────────────
LOOP_PID=""
MON_PID=""
TEE_PID=""
_CLEANUP_DONE=0

# 强力清理：尝试杀掉同一进程组内的所有子进程（保留当前脚本）
cleanup() {
    # 幂等保护：防止 INT/TERM trap 与 EXIT trap 重复执行
    [[ "$_CLEANUP_DONE" -eq 1 ]] && return
    _CLEANUP_DONE=1

    # 终止已追踪的前台子进程
    [[ -n "$LOOP_PID" ]] && kill "$LOOP_PID" 2>/dev/null || true
    [[ -n "$MON_PID"  ]] && kill "$MON_PID"  2>/dev/null || true
    local _pid
    for _pid in "${BENCH_LOOP_PIDS[@]:-}"; do
        [[ -n "$_pid" ]] && kill "$_pid" 2>/dev/null || true
    done

    # 获取当前脚本的进程组 ID，逐一 TERM 再 KILL（排除自身）
    local pgid
    pgid=$(ps -o pgid= "$$" 2>/dev/null | tr -d ' ')
    if [[ -n "$pgid" ]]; then
        local p
        for p in $(ps -o pid= -g "$pgid" 2>/dev/null || true); do
            if [[ "$p" -ne "$$" ]]; then
                kill -TERM "$p" 2>/dev/null || true
            fi
        done
        sleep 0.2
        for p in $(ps -o pid= -g "$pgid" 2>/dev/null || true); do
            if [[ "$p" -ne "$$" ]]; then
                kill -KILL "$p" 2>/dev/null || true
            fi
        done
    fi

    # 显式终止并等待 tee 子进程，避免 wait 因管道未关闭而永久阻塞
    [[ -n "$TEE_PID" ]] && kill "$TEE_PID" 2>/dev/null || true
    [[ -n "$TEE_PID" ]] && wait "$TEE_PID" 2>/dev/null || true
}

# 重新设置 trap：SIGINT 使用 130，SIGTERM 使用 143
trap 'cleanup; exit 130' INT
trap 'cleanup; exit 143' TERM
trap cleanup EXIT

# ── 前置检查 ─────────────────────────────────────────────────────────────────
cd "$PROJECT_ROOT"

[[ -d "$BIN_DIR"  ]] || { echo "Error: BIN_DIR 不存在: $BIN_DIR"  >&2; exit 1; }
[[ -d "$TEST_DIR" ]] || { echo "Error: TEST_DIR 不存在: $TEST_DIR" >&2; exit 1; }

if [[ ! -x "$PMU_MONITOR" ]]; then
    info "pmu_monitor 未找到，尝试编译..."
    make -C "$PROJECT_ROOT" 2>&1 | tail -5
    [[ -x "$PMU_MONITOR" ]] || { echo "Error: pmu_monitor 编译失败" >&2; exit 1; }
fi

PARANOID=$(cat /proc/sys/kernel/perf_event_paranoid 2>/dev/null || echo "2")
if [[ "$PARANOID" -gt 1 ]] && [[ "$EUID" -ne 0 ]]; then
    warn "perf_event_paranoid=$PARANOID，建议以 root 运行或执行："
    warn "  echo 1 | sudo tee /proc/sys/kernel/perf_event_paranoid"
fi

mkdir -p "$DATA_DIR" "$PROJECT_ROOT/log"

# 将整个脚本的 stdout/stderr 记录到 log 文件（包含时间戳与 PID）
LOG_TS=$(date +%Y%m%d_%H%M%S)
LOGFILE="$PROJECT_ROOT/log/collect_dataset_testbench_${VARIANT}_${LOG_TS}_$$.log"
exec > >(tee -a "$LOGFILE") 2>&1
TEE_PID=$!   # 记录 tee 子进程 PID，供 cleanup 精确终止
info "记录脚本输出到: ${LOGFILE#$PROJECT_ROOT/}"

# ── PMU 采集函数 ──────────────────────────────────────────────────────────────
# 参数:
#   $1  bench_name  — 基准名称
#   $2  bench_cmd   — 已展开变量的基准运行命令（不含 cd）
#   $3  run_dir     — 运行目录（对应 .test 中的 cd %S）
#   $4  out_csv     — 输出 CSV 路径
# 执行次数 = pmu_monitor 就绪后到采集窗口结束期间，循环内单个程序重新执行的次数
#           （cnt_before 在窗口开始时读取，cnt_after 在窗口结束时读取，取差值）
collect_pmu() {
    local bench_name="$1"
    local bench_cmd="$2"
    local run_dir="$3"
    local out_csv="$4"
    local reuse_loop=0

    MON_PID=""

    # ── 持续模式：若该基准的工作负载循环仍存活则直接复用，无需重建进程树 ────
    if [[ "$CONTINUOUS" -eq 1 && -n "${BENCH_LOOP_PIDS[$bench_name]:-}" ]]; then
        if kill -0 "${BENCH_LOOP_PIDS[$bench_name]}" 2>/dev/null; then
            LOOP_PID="${BENCH_LOOP_PIDS[$bench_name]}"
            reuse_loop=1
            info "  复用工作负载循环 (PID $LOOP_PID)"
        else
            warn "  已记录的循环进程 (PID ${BENCH_LOOP_PIDS[$bench_name]}) 已终止，重新启动"
            unset "BENCH_LOOP_PIDS[$bench_name]"
        fi
    fi

    if [[ "$reuse_loop" -eq 0 ]]; then
        LOOP_PID=""
        # 计数文件：每次执行 +1，持续模式下跨轮累加（取差值得本窗口次数）
        local cnt_file="$PROJECT_ROOT/log/run_count_${bench_name//[^a-zA-Z0-9_]/_}_$$.txt"
        printf '0' > "$cnt_file"
        # 将计数文件路径存入以 bench_name 命名的动态变量，供复用时读取
        local _safe_name="${bench_name//[^a-zA-Z0-9_]/_}"
        eval "BENCH_CNT_${_safe_name}=\$cnt_file"

        # 在后台以 shell 循环持续运行基准，直到被 kill
        # pmu_counters.c 中 pe.inherit=1，pmu_monitor 可捕获子进程 PMU 事件
        #
        # 关键：pe.inherit 只对 perf_event_open() 之后 fork 的子进程生效。
        # 因此循环进程必须等 pmu_monitor 就绪（ready_flag 出现）后再启动第一次 benchmark，
        # 否则 benchmark 子进程在挂载前已 fork，导致整个采集窗口全为 0。
        local ready_flag="$PROJECT_ROOT/log/pmu_ready_${bench_name//[^a-zA-Z0-9_]/_}_$$.flag"
        rm -f "$ready_flag"

        (
            cd "$run_dir"
            _cnt=0
            # 等待 pmu_monitor 开始监控后再运行第一次 benchmark
            while [[ ! -f "$ready_flag" ]]; do sleep 0.05; done
            while true; do
                eval "$bench_cmd" >/dev/null 2>&1 || true
                (( _cnt++ )) || true
                printf '%s' "$_cnt" > "$cnt_file"
            done
        ) &
        LOOP_PID=$!

        # 等待循环进程稳定启动（此时它在 ready_flag 上自旋等待，尚未 fork benchmark）
        sleep 0.3

        if ! kill -0 "$LOOP_PID" 2>/dev/null; then
            warn "循环进程启动后立即退出"
            LOOP_PID=""
            return 1
        fi

        # 持续模式：记录循环 PID，供后续轮次直接复用（不重建进程）
        [[ "$CONTINUOUS" -eq 1 ]] && BENCH_LOOP_PIDS[$bench_name]=$LOOP_PID
    fi

    # 获取本次采集对应的计数文件路径（新建或复用）
    local _safe_name="${bench_name//[^a-zA-Z0-9_]/_}"
    local _cnt_var="BENCH_CNT_${_safe_name}"
    local cnt_file="${!_cnt_var:-}"
    local cnt_before=0  # 在 pmu_monitor 就绪后再读取，见下方

    # 启动 pmu_monitor，监控循环进程（含 inherit 子进程）
    # 每轮重新挂载以获取独立的 PMU 计数窗口；循环进程 PID 不变
    # stderr 写入日志以便诊断计数器打开失败等问题
    local pmu_err_log="$PROJECT_ROOT/log/pmu_monitor_stderr_$$.txt"
    "$PMU_MONITOR" "$LOOP_PID" -i "$INTERVAL_MS" >/dev/null 2>"$pmu_err_log" &
    MON_PID=$!

    # 短暂等待确认 pmu_monitor 成功启动
    sleep 0.3 || true
    if ! kill -0 "$MON_PID" 2>/dev/null; then
        warn "pmu_monitor 意外退出，错误信息："
        cat "$pmu_err_log" >&2 || true
        MON_PID=""
        # pmu_monitor 启动失败：终止刚启动的循环进程并从记录中清除
        if [[ "$reuse_loop" -eq 0 ]]; then
            kill "$LOOP_PID" 2>/dev/null || true
            wait "$LOOP_PID" 2>/dev/null || true
            unset "BENCH_LOOP_PIDS[$bench_name]" 2>/dev/null || true
        fi
        LOOP_PID=""
        return 1
    fi
    if [[ -s "$pmu_err_log" ]]; then
        warn "pmu_monitor 有警告信息："
        cat "$pmu_err_log" >&2 || true
    fi

    # pmu_monitor 已就绪（perf_event_open 已完成）：通知循环进程可以开始 fork benchmark
    # 此后所有子进程都会被 inherit 捕获，解决"全 0"问题
    if [[ "$reuse_loop" -eq 0 ]]; then
        local _ready_flag_var="$PROJECT_ROOT/log/pmu_ready_${bench_name//[^a-zA-Z0-9_]/_}_$$.flag"
        touch "$_ready_flag_var"
        info "  pmu_monitor 已就绪，通知 loop 开始执行 benchmark"
    fi

    # 在 pmu_monitor 就绪、loop 开始执行后读取基线计数
    # 执行次数 = 本窗口内循环内单个程序重新执行的次数（cnt_after - cnt_before）
    # reuse_loop=0 时 loop 此刻才刚被允许执行，cnt 仍为 0，cnt_before=0 亦正确
    [[ -f "$cnt_file" ]] && cnt_before=$(cat "$cnt_file" 2>/dev/null || echo 0)

    # 采集 PMU_WINDOW 秒
    sleep "$PMU_WINDOW" || true

    # 终止监控进程（每轮结束后关闭，下轮重新挂载同一循环 PID）
    kill "$MON_PID"  2>/dev/null || true
    wait "$MON_PID"  2>/dev/null || true
    MON_PID=""

    # 读取本窗口结束时的执行次数，计算差值
    local cnt_after=0
    [[ -f "$cnt_file" ]] && cnt_after=$(cat "$cnt_file" 2>/dev/null || echo 0)
    local run_count=$(( cnt_after - cnt_before ))
    info "  本窗口执行次数: $run_count 次 (${PMU_WINDOW}s)"

    # 单次模式：同时终止循环进程并清理计数文件；持续模式：保持运行供下轮复用
    if [[ "$CONTINUOUS" -eq 0 ]]; then
        kill "$LOOP_PID" 2>/dev/null || true
        wait "$LOOP_PID" 2>/dev/null || true
        [[ -f "$cnt_file" ]] && rm -f "$cnt_file" || true
        # 清理 ready_flag 文件
        local _rf="$PROJECT_ROOT/log/pmu_ready_${bench_name//[^a-zA-Z0-9_]/_}_$$.flag"
        rm -f "$_rf" || true
    fi
    LOOP_PID=""

    sleep 0.3

    if [[ -f "$PROJECT_ROOT/log/pmu_monitor.csv" ]]; then
        cp -L "$PROJECT_ROOT/log/pmu_monitor.csv" "$out_csv"
        # 将执行次数写入同名 .runs 文件（每轮覆盖）
        printf '%s' "$run_count" > "${out_csv%.csv}.runs"
        # 检查收集到的行数（不含表头）
        local row_count
        row_count=$(awk 'NR>1' "$out_csv" | wc -l)
        if [[ "$row_count" -lt "$MIN_ROWS" ]]; then
            warn "数据不足：仅收集到 $row_count 行（期望 ≥ $MIN_ROWS）"
            return 2   # 专用返回码：数据不足
        fi
        _LAST_RUN_COUNT=$run_count
        return 0
    else
        warn "log/pmu_monitor.csv 未生成"
        return 1
    fi
}

# ── 数据质量检查函数 ──────────────────────────────────────────────────────────
# 检查 CSV 文件质量，输出摘要
# 参数: $1 csv_path
check_csv_quality() {
    local csv="$1"
    [[ -f "$csv" ]] || { warn "文件不存在: $csv"; return 1; }
    local n inst_zero lbr_zero row_count
    row_count=$(awk 'NR>1' "$csv" | wc -l)
    inst_zero=$(awk -F, 'NR>1 && $3=="0"' "$csv" | wc -l)
    lbr_zero=$(awk -F, 'NR>1 && $21=="0"' "$csv" | wc -l)
    local lbr_pos=$(( row_count - lbr_zero ))
    local issues=""
    [[ "$row_count" -lt "$MIN_ROWS" ]] && issues+="行数不足(${row_count}<${MIN_ROWS}) "
    [[ "$inst_zero" -gt 1 ]] && issues+="inst=0行过多(${inst_zero}) "
    [[ "$lbr_pos" -eq 0 ]] && issues+="LBR全零 "
    if [[ -z "$issues" ]]; then
        info "    质量OK: ${row_count}行, lbr命中${lbr_pos}/${row_count}"
    else
        warn "    质量问题: $issues (${row_count}行, lbr命中${lbr_pos}/${row_count})"
    fi
}

# ── 从 .test 文件解析并展开 RUN: 命令 ────────────────────────────────────────
# 参数:
#   $1  test_file   — .test 文件路径
#   $2  binary      — 实际可执行文件绝对路径
#   $3  test_data   — 测试数据目录绝对路径（对应 %S）
# 输出: 展开后的命令（stdout），运行目录已剥离（仅 cmd 部分）
# 返回: 0 成功，1 未找到 RUN: 行
parse_run_cmd() {
    local test_file="$1"
    local binary="$2"
    local test_data="$3"

    # 取第一条 RUN: 行，去掉前缀
    local run_raw
    run_raw=$(grep '^RUN:' "$test_file" | head -1 | sed 's/^RUN: //') || return 1
    [[ -z "$run_raw" ]] && return 1

    # 去掉 "cd %S ; " 前缀（后续统一用 run_dir 处理）
    local cmd_part="$run_raw"
    if [[ "$run_raw" == "cd %S ;"* ]]; then
        cmd_part="${run_raw#cd %S ; }"
    fi

    # 找出命令中第一个词作为可执行文件引用（格式 %S/<name>）
    local first_word
    first_word=$(echo "$cmd_part" | awk '{print $1}')

    if [[ "$first_word" == "%S/"* ]]; then
        local bin_ref_name="${first_word#%S/}"
        # 用 sed 将该可执行文件引用替换为实际二进制路径（精确匹配）
        cmd_part=$(printf '%s' "$cmd_part" \
            | sed "s|%S/${bin_ref_name}|${binary}|g")
    fi

    # 将剩余的 %S 引用（如 %S/dbdir）替换为测试数据目录
    cmd_part=$(printf '%s' "$cmd_part" | sed "s|%S|${test_data}|g")

    printf '%s' "$cmd_part"
    return 0
}

# ── 主循环 ───────────────────────────────────────────────────────────────────
bold "======================================================"
bold "     PMU 采集：$VARIANT 预编译基准"
bold "======================================================"
info "BIN_DIR:    $BIN_DIR"
info "TEST_DIR:   $TEST_DIR"
info "DATA_DIR:   $DATA_DIR"
info "PMU 窗口:   ${PMU_WINDOW}s  |  采样间隔: ${INTERVAL_MS}ms"
info "持续循环:   $( [[ $CONTINUOUS -eq 1 ]] && echo '是（Ctrl+C 停止）' || echo '否（单次）')"
info "最少行数:   ${MIN_ROWS}  |  最大重试: ${RETRY_MAX}"
[[ "$DRYRUN" -eq 1 ]] && warn "★ DRYRUN 模式：仅解析命令，不执行采集 ★"
echo

# ── 主采集循环（支持持续运行）────────────────────────────────────────────────
run_one_pass() {
[[ "$DRYRUN" -eq 0 ]] && > "$MANIFEST"

local _pass_ok=0 _pass_skip=0
for test_subdir in "$TEST_DIR"/*/; do
    [[ -d "$test_subdir" ]] || continue
    bench_name="$(basename "$test_subdir")"
    ((COUNT_TOTAL++)) || true

    info "[$COUNT_TOTAL] $bench_name  (轮次 $ROUND)"

    # ── 查找 .test 文件 ───────────────────────────────────────────────────────
    test_file=$(find "$test_subdir" -maxdepth 1 -name "*.test" | head -1)
    if [[ -z "$test_file" ]]; then
        err "$bench_name: 未找到 .test 文件，跳过"
        ((COUNT_SKIP++)) || true
        continue
    fi

    # ── 查找对应可执行文件 ────────────────────────────────────────────────────
    binary="$BIN_DIR/${bench_name}_${VARIANT}"
    if [[ ! -x "$binary" ]]; then
        err "$bench_name: 可执行文件不存在: $binary，跳过"
        ((COUNT_SKIP++)) || true
        continue
    fi

    # ── 解析 RUN: 行 ──────────────────────────────────────────────────────────
    bench_cmd=$(parse_run_cmd "$test_file" "$binary" "$test_subdir") || {
        err "$bench_name: .test 文件中未找到 RUN: 行，跳过"
        ((COUNT_SKIP++)) || true
        continue
    }

    info "  CMD: $bench_cmd"

    # ── DRYRUN：打印完整执行信息后跳过采集 ──────────────────────────────────
    if [[ "$DRYRUN" -eq 1 ]]; then
        printf "${BOLD}  [DRYRUN]${NC} bench=%-20s\n"          "$bench_name"
        printf "  ${CYAN}run_dir${NC}  = %s\n"                   "$test_subdir"
        printf "  ${CYAN}binary${NC}   = %s\n"                   "$binary"
        printf "  ${CYAN}full_cmd${NC} = cd %s && %s\n"          "$test_subdir" "$bench_cmd"
        printf "  ${CYAN}out_csv${NC}  = %s\n"                   "$DATA_DIR/${bench_name}_${VARIANT}.csv"
        echo
        ((_pass_ok++)) || true
        continue
    fi

    _LAST_RUN_COUNT=0

    # ── PMU 采集（含重试）────────────────────────────────────────────────────
    out_csv="$DATA_DIR/${bench_name}_${VARIANT}.csv"

    # 已有足够数据且不强制覆盖时跳过
    # 持续模式第一轮同样跳过已有充足数据的条目；后续轮次（ROUND>1）强制重新采集
    local _effective_overwrite
    _effective_overwrite=$(( OVERWRITE == 1 || (CONTINUOUS == 1 && ROUND > 1) ? 1 : 0 ))
    if [[ "$_effective_overwrite" -eq 0 && -f "$out_csv" ]]; then
        local existing_rows
        existing_rows=$(awk 'NR>1' "$out_csv" | wc -l)
        if [[ "$existing_rows" -ge "$MIN_ROWS" ]]; then
            info "  已有足够数据（${existing_rows}行），跳过采集"
            check_csv_quality "$out_csv"
            ((_pass_ok++)) || true
            continue
        fi
        warn "  已有数据不足（${existing_rows}<${MIN_ROWS}行），重新采集"
    fi

    # 切换程序时终止其他基准的循环进程，防止干扰当前基准的 PMU 采集
    # （若不清理，多个 benchmark 同时跑会污染 PMU 计数）
    for _old_bench in "${!BENCH_LOOP_PIDS[@]}"; do
        [[ "$_old_bench" == "$bench_name" ]] && continue
        _old_pid="${BENCH_LOOP_PIDS[$_old_bench]}"
        if [[ -n "$_old_pid" ]] && kill -0 "$_old_pid" 2>/dev/null; then
            info "  终止旧基准循环进程: $_old_bench (PID $_old_pid)"
            kill "$_old_pid" 2>/dev/null || true
            wait "$_old_pid" 2>/dev/null || true
        fi
        unset "BENCH_LOOP_PIDS[$_old_bench]"
    done

    local attempt=0 collect_rc=0
    while [[ "$attempt" -lt "$RETRY_MAX" ]]; do
        ((attempt++)) || true
        [[ "$attempt" -gt 1 ]] && retry "$bench_name: 第 $attempt 次重试..."
        collect_rc=0
        collect_pmu "$bench_name" "$bench_cmd" "$test_subdir" "$out_csv" \
            || collect_rc=$?
        if [[ "$collect_rc" -eq 0 ]]; then
            break
        elif [[ "$collect_rc" -eq 2 ]]; then
            warn "  数据不足，将重试（$attempt/$RETRY_MAX）"
        else
            warn "  采集失败，将重试（$attempt/$RETRY_MAX）"
        fi
    done

    if [[ "$collect_rc" -ne 0 ]]; then
        err "$bench_name: $RETRY_MAX 次尝试后仍失败，跳过"
        ((_pass_skip++)) || true
        ((COUNT_SKIP++)) || true
        continue
    fi
    info "  CSV: ${out_csv#$PROJECT_ROOT/}  |  本窗口执行 ${_LAST_RUN_COUNT} 次"
    check_csv_quality "$out_csv"

    # ── 写入 manifest（含执行次数）────────────────────────────────────────────
    local_bench_cmd_escaped="${bench_cmd//\"/\\\"}"
    printf '{"program":"%s","variant":"%s","binary":"%s","run_cmd":"%s","csv":"%s","run_count":%d}\n' \
        "$bench_name" \
        "$VARIANT" \
        "${binary#$PROJECT_ROOT/}" \
        "$local_bench_cmd_escaped" \
        "${out_csv#$PROJECT_ROOT/}" \
        "${_LAST_RUN_COUNT:-0}" \
        >> "$MANIFEST"

    pass "$bench_name 完成（本窗口执行 ${_LAST_RUN_COUNT:-0} 次）"
    ((_pass_ok++)) || true
    ((COUNT_OK++)) || true
    echo
done

printf "  本轮: 成功 %d  跳过 %d\n" "$_pass_ok" "$_pass_skip"
} # end run_one_pass

# ── 执行（单次 or 持续）──────────────────────────────────────────────────────
if [[ "$CONTINUOUS" -eq 1 ]]; then
    info "持续模式：Ctrl+C 停止"
    while true; do
        ((ROUND++)) || true
        bold "───────────────── 轮次 $ROUND ─────────────────"
        run_one_pass
        echo
    done
else
    ((ROUND++)) || true
    run_one_pass
fi

# ── 汇总 ─────────────────────────────────────────────────────────────────────
echo
bold "======================================================"
printf "  总计: %d  |  成功: %d  |  跳过: %d\n" \
    "$COUNT_TOTAL" "$COUNT_OK" "$COUNT_SKIP"
echo "  清单: $MANIFEST"
bold "======================================================"
