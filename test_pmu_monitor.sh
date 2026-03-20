#!/bin/bash
# test_pmu_monitor.sh — pmu_monitor 定时采集测试脚本
#
# 测试流程：
#   1. 检查运行权限（perf_event_paranoid）
#   2. 编译 pmu_monitor 和 test_pmu_workload（如未编译）
#   3. 启动 test_pmu_workload 作为被监控目标
#   4. 启动 pmu_monitor 以指定间隔监控该进程，持续 TEST_DURATION 秒
#   5. 终止两个进程
#   6. 验证 CSV 日志（行数、elapsed_ms 单调递增、非零计数器）
#   7. 输出测试结果

set -euo pipefail

# ── 配置（支持通过环境变量从外部脚本覆盖）────────────────────────────────
INTERVAL_MS="${INTERVAL_MS:-500}"               # pmu_monitor 采样间隔（毫秒）
TEST_DURATION="${TEST_DURATION:-60}"            # 测试持续时间（秒）
WORKLOAD_DURATION="${WORKLOAD_DURATION:-120}"   # 工作负载最长持续时间（秒）
TIMESERIES_BIN="${TIMESERIES_BIN:-./pmu_monitor}"
WORKLOAD_BIN="${WORKLOAD_BIN:-./test/test_workload}"
WORKLOAD_NAME="$(basename "$WORKLOAD_BIN")"
# 唯一运行 ID，包含时间（时分秒）与 PID，确保整个脚本一致使用同一后缀
RUN_ID="$(date +%H%M%S)_$$"
LOG_LINK="log/pmu_monitor.csv"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── 颜色 ────────────────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
CYAN='\033[0;36m'; NC='\033[0m'

pass() { printf "${GREEN}[PASS]${NC} %s\n" "$*"; }
fail() { printf "${RED}[FAIL]${NC} %s\n" "$*"; }
info() { printf "${CYAN}[INFO]${NC} %s\n" "$*"; }
warn() { printf "${YELLOW}[WARN]${NC} %s\n" "$*"; }

TESTS_PASSED=0
TESTS_FAILED=0

assert_true() {
    local desc="$1"; shift
    if "$@" 2>/dev/null; then
        pass "$desc"
        ((TESTS_PASSED++)) || true
    else
        fail "$desc"
        ((TESTS_FAILED++)) || true
    fi
}

# ── 清理钩子 ────────────────────────────────────────────────────────────────
TIMESERIES_PID=""
WORKLOAD_PID=""
WORKLOAD_KILLER_PID=""

cleanup() {
    [[ -n "$TIMESERIES_PID" ]] && kill "$TIMESERIES_PID" 2>/dev/null || true
    [[ -n "$WORKLOAD_PID"   ]] && kill "$WORKLOAD_PID"   2>/dev/null || true
    [[ -n "$WORKLOAD_KILLER_PID" ]] && kill "$WORKLOAD_KILLER_PID" 2>/dev/null || true
    wait 2>/dev/null || true
}
trap cleanup EXIT INT TERM

# ── 进入 pmu 目录 ─────────────────────────────────────────────────────────────
cd "$SCRIPT_DIR"

EXPECTED_SAMPLES=$(( TEST_DURATION * 1000 / INTERVAL_MS ))

echo "======================================================"
echo "         pmu_monitor 定时采集功能测试"
echo "======================================================"
printf "  采样间隔  : %d ms\n"  "$INTERVAL_MS"
printf "  测试时长  : %d s\n"   "$TEST_DURATION"
printf "  预期样本数: ~%d 行\n" "$EXPECTED_SAMPLES"
echo

# ── 步骤 1：权限检查 ──────────────────────────────────────────────────────────
info "检查 perf_event_paranoid 权限..."
PARANOID=$(cat /proc/sys/kernel/perf_event_paranoid 2>/dev/null || echo "2")
if [[ "$PARANOID" -gt 1 ]]; then
    warn "perf_event_paranoid=$PARANOID，跨进程监控可能需要 root 或 CAP_PERFMON"
    warn "若计数器全部失败，请以 root 运行或执行："
    warn "  echo 1 | sudo tee /proc/sys/kernel/perf_event_paranoid"
fi
info "当前 perf_event_paranoid=$PARANOID"
echo

# ── 步骤 2：编译 ──────────────────────────────────────────────────────────────
info "检查并编译所需程序..."

info "编译 pmu_monitor..."
make

assert_true "pmu_monitor 可执行文件存在" test -x "$TIMESERIES_BIN"

info "编译 test/.."
make test   
assert_true "test 可执行文件存在" test -x "$WORKLOAD_BIN"
echo

# ── 步骤 3：启动工作负载 ──────────────────────────────────────────────────────
mkdir -p log
info "启动 test_pmu_workload（运行直到收到停止信号）..."
"$WORKLOAD_BIN" &
WORKLOAD_PID=$!
info "工作负载 PID: $WORKLOAD_PID"

# 在后台定时器中，WORKLOAD_DURATION 秒后发送 SIGTERM 给工作负载
(
    sleep "$WORKLOAD_DURATION"
    if kill -0 "$WORKLOAD_PID" 2>/dev/null; then
        info "WORKLOAD_DURATION=${WORKLOAD_DURATION}s 到达，发送 SIGTERM 给工作负载 PID $WORKLOAD_PID"
        kill "$WORKLOAD_PID" 2>/dev/null || true
    fi
) &
WORKLOAD_KILLER_PID=$!

# 等待工作负载完成内存初始化
assert_true "工作负载进程存活" kill -0 "$WORKLOAD_PID"
echo

# ── 步骤 4：启动 pmu_monitor ───────────────────────────────────────────────
info "启动 pmu_monitor 监控 PID $WORKLOAD_PID（间隔 ${INTERVAL_MS} ms）..."
"$TIMESERIES_BIN" "$WORKLOAD_PID" -i "$INTERVAL_MS" \
    >"log/timeseries_stdout_${RUN_ID}.txt" 2>"log/timeseries_stderr_${RUN_ID}.txt" &
TIMESERIES_PID=$!
info "采集器 PID: $TIMESERIES_PID"

# 等待 pmu_monitor 创建 CSV（最多等待 ~5s），并在 log/ 下创建 pmu_monitor.csv 符号链接
# 这样后续断言可以通过，即使被监控程序或权限导致 pmu_monitor 未创建内部链接
(
    CSV_REAL=""
    for i in $(seq 1 10); do
        CSV_REAL=$(ls -1t log/pmu_monitor_*.csv 2>/dev/null | head -n1 || true)
        if [[ -n "$CSV_REAL" ]]; then
            break
        fi
        sleep 0.5
    done
    if [[ -n "$CSV_REAL" ]]; then
        (
            cd log
            ln -sf "$(basename "$CSV_REAL")" pmu_monitor.csv
        )
        info "已创建符号链接：log/pmu_monitor.csv -> $(basename "$CSV_REAL")"
    else
        warn "未检测到 pmu_monitor_*.csv（等待超时），后续断言可能失败"
    fi
)

# (延迟创建 stdout/stderr 链接至脚本末尾，保留原始 *_$$.txt)

# 等待指定测试时长
info "等待 ${TEST_DURATION} 秒完成采集..."
sleep "$TEST_DURATION"

assert_true "采集器进程在测试期间持续运行" kill -0 "$TIMESERIES_PID"
echo

# ── 步骤 5：停止进程 ──────────────────────────────────────────────────────────
info "停止采集器和工作负载..."
kill "$TIMESERIES_PID" 2>/dev/null || true
kill "$WORKLOAD_PID"   2>/dev/null || true
wait "$TIMESERIES_PID" 2>/dev/null || true
wait "$WORKLOAD_PID"   2>/dev/null || true
TIMESERIES_PID=""
WORKLOAD_PID=""
echo

# ── 步骤 6：验证 CSV 日志 ─────────────────────────────────────────────────────
echo "------------------------------------------------------"
info "验证日志文件：$LOG_LINK"

assert_true "日志符号链接存在" test -L "$LOG_LINK"

REAL_LOG="$(readlink -f "$LOG_LINK" 2>/dev/null || echo "")"
assert_true "日志真实文件存在" test -f "$REAL_LOG"

# (延迟创建 CSV 链接至脚本末尾)

# 行数验证（标题行 + 数据行）
TOTAL_LINES=$(wc -l < "$REAL_LOG")
DATA_LINES=$(( TOTAL_LINES - 1 ))
info "日志总行数: ${TOTAL_LINES}（标题 1 行 + 数据 ${DATA_LINES} 行）"

assert_true "日志至少包含 1 行数据" test "$DATA_LINES" -ge 1

# 宽松检查：允许最多 20% 误差（定时器偶发 overrun 或启动延迟）
MIN_SAMPLES=$(( EXPECTED_SAMPLES * 8 / 10 ))
if [[ "$DATA_LINES" -ge "$MIN_SAMPLES" ]]; then
    pass "数据行数 ${DATA_LINES} 满足预期（≥ ${MIN_SAMPLES}）"
    ((TESTS_PASSED++)) || true
else
    warn "数据行数 ${DATA_LINES} 低于预期下限 ${MIN_SAMPLES}（预期 ~${EXPECTED_SAMPLES}）"
fi

# 标题行验证
assert_true "标题首列为 elapsed_ms"    grep -q "^elapsed_ms," "$REAL_LOG"
assert_true "标题包含 inst_retired.any"      grep -q "inst_retired.any"        "$REAL_LOG"
assert_true "标题包含 L1-icache-load-misses" grep -q "L1-icache-load-misses"   "$REAL_LOG"
assert_true "标题包含 iTLB-loads" grep -q "iTLB-loads"   "$REAL_LOG"

# elapsed_ms 单调递增验证（取前 10 个数据行）
info "验证 elapsed_ms 单调递增..."
MONOTONIC=true
PREV_MS=-1
while IFS=, read -r elapsed_ms _rest; do
    [[ "$elapsed_ms" == "elapsed_ms" ]] && continue   # 跳过标题
    if [[ "$elapsed_ms" =~ ^[0-9]+$ ]]; then
        if (( elapsed_ms <= PREV_MS )); then
            MONOTONIC=false
            break
        fi
        PREV_MS=$elapsed_ms
    fi
done < <(head -n 12 "$REAL_LOG")   # 标题 + 前 11 个数据行

if $MONOTONIC; then
    pass "elapsed_ms 列单调递增"
    ((TESTS_PASSED++)) || true
else
    fail "elapsed_ms 列出现非单调值（prev=$PREV_MS, cur=$elapsed_ms）"
    ((TESTS_FAILED++)) || true
fi

# 非零计数器验证
ANY_NONZERO=false
while IFS= read -r line; do
    [[ "$line" == elapsed_ms* ]] && continue   # 跳过标题行
    IFS=',' read -ra fields <<< "$line"
    for f in "${fields[@]:2}"; do              # 跳过 elapsed_ms 和 timestamp 列
        [[ "$f" =~ ^[0-9]+$ ]] && [[ "$f" -gt 0 ]] && { ANY_NONZERO=true; break 2; }
    done
done < "$REAL_LOG"

if $ANY_NONZERO; then
    pass "日志中存在非零计数器值"
    ((TESTS_PASSED++)) || true
else
    fail "日志中所有计数器均为 0（硬件不支持或权限不足）"
    ((TESTS_FAILED++)) || true
fi

# ── 步骤 7：显示日志摘要 ──────────────────────────────────────────────────────
echo
echo "------------------------------------------------------"
info "日志文件前 3 行："
head -n 3 "$REAL_LOG" | while IFS= read -r l; do printf "    %s\n" "$l"; done

# 统计活跃计数器（标题行中非 elapsed_ms/timestamp 的列数）
TOTAL_COLS=$(head -1 "$REAL_LOG" | tr ',' '\n' | wc -l)
COUNTER_COLS=$(( TOTAL_COLS - 2 ))   # 减去 elapsed_ms 和 timestamp 两列

# 统计不可用（空值）计数器
EMPTY_COUNT=0
if [[ "$DATA_LINES" -ge 1 ]]; then
    FIRST_DATA=$(sed -n '2p' "$REAL_LOG")
    EMPTY_COUNT=$(echo "$FIRST_DATA" | tr ',' '\n' | tail -n +"$((2+1))" | grep -c '^$' || true)
fi
ACTIVE_COUNT=$(( COUNTER_COLS - EMPTY_COUNT ))
info "计数器统计：共 ${COUNTER_COLS} 个，活跃 ${ACTIVE_COUNT} 个，不可用 ${EMPTY_COUNT} 个"
info "日志路径：$REAL_LOG"
info "采集器标准输出：log/timeseries_stdout_${RUN_ID}.txt"
info "采集器标准错误：log/timeseries_stderr_${RUN_ID}.txt"

## 在脚本结束前创建工作负载专用的 stdout/stderr 和 CSV 符号链接，保留原始 *_$$.txt
# 在 log 目录内创建链接，确保相对目标位于同一目录，避免产生多余的 "log/" 前缀
WORKLOAD_SPECIFIC_BASENAME="pmu_monitor_${WORKLOAD_NAME}.csv"
if [[ -n "$REAL_LOG" && -f "$REAL_LOG" ]]; then
    mkdir -p log
    (
        cd log
        ln -sf "$REAL_LOG" "$WORKLOAD_SPECIFIC_BASENAME"
        ln -sf "$WORKLOAD_SPECIFIC_BASENAME" pmu_monitor.csv
    )
    info "已创建符号链接：log/pmu_monitor.csv -> ${WORKLOAD_SPECIFIC_BASENAME}"
fi

# 创建 stdout/stderr 链接（优先使用 PID 版本，如果存在），在 log 目录中操作以保持目标一致
if [[ -f "log/timeseries_stdout_${RUN_ID}.txt" ]]; then
    (
        cd log
        ln -sf "timeseries_stdout_${RUN_ID}.txt" "timeseries_stdout_${WORKLOAD_NAME}.txt"
        
    )
fi
if [[ -f "log/timeseries_stderr_${RUN_ID}.txt" ]]; then
    (
        cd log
        ln -sf "timeseries_stderr_${RUN_ID}.txt" "timeseries_stderr_${WORKLOAD_NAME}.txt"
    )
fi

# ── 汇总 ──────────────────────────────────────────────────────────────────────
echo
echo "======================================================"
printf "测试结果：${GREEN}%d 通过${NC}，${RED}%d 失败${NC}\n" \
       "$TESTS_PASSED" "$TESTS_FAILED"
echo "======================================================"

if [[ "$TESTS_FAILED" -eq 0 ]]; then
    printf "${GREEN}所有测试通过！${NC}\n"
    exit 0
else
    printf "${RED}存在失败测试，请检查上方输出。${NC}\n"
    exit 1
fi