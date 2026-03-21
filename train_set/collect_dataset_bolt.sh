#!/bin/bash
# =============================================================================
# collect_dataset_bolt.sh — MultiSource × BOLT 版本变体训练集采集脚本
# =============================================================================
#
# 工作流程（每个多源程序目录）：
#   1. 递归扫描 MultiSource/{Benchmarks,Applications}，找到包含 int main(
#      的叶目录（.c/.cpp 文件与 main 在同一层）
#   2. 将该目录内所有 .c/.cpp 文件 + loop_main.c 一起编译：
#      - 含 main 的文件用 -Dmain=bench_entry 重命名入口
#      - 链接标志加入 -Wl,--emit-relocs（BOLT 必需）
#      → BIN_V1（-O3 基准版本）
#   3. 用 perf LBR 对运行中的 BIN_V1 采样 → perf.data
#      ( 回退：LBR 不可用时改用 llvm-bolt --instrument 插桩模式 )
#   4. perf2bolt -p perf.data → bolt.fdata
#   5. llvm-bolt BIN_V1 -data bolt.fdata → BIN_V2（代码布局重排版本）
#   6. 固定时间窗口内统计执行次数（N_V1, N_V2），生成 label
#   7. pmu_monitor 分别采集 v1/v2 的 PMU 时序 CSV
#   8. 写入 manifest.jsonl（与 process_features.py / siamese_train.py 兼容）
#
# BOLT 先决条件：
#   - llvm-bolt, perf2bolt  在 PATH 中可用
#   - perf record 支持 -j any,u（Intel LBR）
#   - perf_event_paranoid ≤ 1，或以 root 运行
#   - 链接时 -Wl,--emit-relocs（脚本已自动添加）
#
# 用法：
#   cd /path/to/Siamese-MicroPerf
#   sudo bash train_set/collect_dataset_bolt.sh
#
# 环境变量（可覆盖默认值）：
#   CC              C 编译器（默认自动检测 clang/gcc）
#   CXX             C++ 编译器（默认自动检测 clang++/g++）
#   BENCH_ROOTS     空格分隔的根目录（相对于 PROJECT_ROOT，默认见下）
#   OUT_DIR         可执行文件输出目录（默认 train_set/bin_bolt）
#   DATA_DIR        PMU CSV 输出目录（默认 train_set/data_bolt）
#   PROF_DIR        perf.data / bolt.fdata 输出目录（默认 train_set/prof）
#   LABEL_WINDOW    标签生成时间窗口，秒（默认 10）
#   PMU_WINDOW      PMU 采集时间窗口，秒（默认 30）
#   BOLT_PROF_WIN   BOLT 剖析时间窗口，秒（默认 15）
#   INTERVAL_MS     pmu_monitor 采样间隔，毫秒（默认 500）
#   MIN_EXECS       最小执行次数阈值，低于此值跳过（默认 2）
#   MAX_SRCS        单个程序最多允许的源文件数，超出则跳过（默认 50）
#
# 输出文件结构（相对于 train_set/）：
#   bin_bolt/          编译后的可执行文件（*_v1 基准，*_v2 BOLT 优化）
#   data_bolt/         PMU CSV 时序文件
#   prof/              perf.data / bolt.fdata（可安全删除节省空间）
#   manifest_bolt.jsonl   数据集配对清单（每行一条 JSON）
# =============================================================================

set -euo pipefail

# ── 路径配置 ──────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

CC="${CC:-}"
CXX="${CXX:-}"

# 多Source 扫描根目录（用空格分隔多个相对路径）
BENCH_ROOTS="${BENCH_ROOTS:-llvm-test-suite/MultiSource/Benchmarks llvm-test-suite/MultiSource/Applications}"

OUT_DIR="${OUT_DIR:-$SCRIPT_DIR/bin_bolt}"
DATA_DIR="${DATA_DIR:-$SCRIPT_DIR/data_bolt}"
PROF_DIR="${PROF_DIR:-$SCRIPT_DIR/prof}"
PMU_MONITOR="${PMU_MONITOR:-$PROJECT_ROOT/pmu_monitor}"
LOOP_MAIN="$SCRIPT_DIR/loop_main.c"
MANIFEST="$SCRIPT_DIR/manifest_bolt.jsonl"

LABEL_WINDOW="${LABEL_WINDOW:-10}"
PMU_WINDOW="${PMU_WINDOW:-30}"
BOLT_PROF_WIN="${BOLT_PROF_WIN:-15}"
INTERVAL_MS="${INTERVAL_MS:-500}"
MIN_EXECS="${MIN_EXECS:-2}"
MAX_SRCS="${MAX_SRCS:-50}"

# v1 编译标志：-O3 + 必须有 --emit-relocs 供 BOLT 使用
OPT_V1="${OPT_V1:--O3}"
EMIT_RELOCS="-Wl,--emit-relocs"

# llvm-bolt 代码布局重排参数
BOLT_FLAGS="${BOLT_FLAGS:--reorder-blocks=ext-tsp -reorder-functions=hfsort -split-functions -split-all-cold -plt}"

# ── 颜色日志 ─────────────────────────────────────────────────────────────────
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
COUNT_SKIP_COMPILE=0
COUNT_SKIP_BOLT=0
COUNT_SKIP_LABEL=0
COUNT_SKIP_PMU=0

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

# 自动检测 C++ 编译器（与 CC 同系列）
if [[ -z "$CXX" ]]; then
    if   command -v clang++ &>/dev/null; then CXX=clang++
    elif command -v g++     &>/dev/null; then CXX=g++
    else CXX="$CC"; fi  # 回退：C++ 程序跳过
fi

info "C 编译器:   $CC"
info "C++ 编译器: $CXX"

# llvm-bolt / perf2bolt
HAS_BOLT=0
if command -v llvm-bolt &>/dev/null && command -v perf2bolt &>/dev/null; then
    HAS_BOLT=1
    info "BOLT: llvm-bolt=$(command -v llvm-bolt)"
else
    warn "llvm-bolt 或 perf2bolt 未找到 —— 仅生成 v1，跳过 BOLT 变体"
    warn "安装方法（Debian/Ubuntu）: sudo apt install llvm"
fi

# pmu_monitor
if [[ ! -x "$PMU_MONITOR" ]]; then
    info "pmu_monitor 未找到，尝试编译..."
    make -C "$PROJECT_ROOT" 2>&1 | tail -5
    [[ -x "$PMU_MONITOR" ]] || { echo "Error: pmu_monitor 编译失败" >&2; exit 1; }
fi

# perf_event_paranoid 告警
PARANOID=$(cat /proc/sys/kernel/perf_event_paranoid 2>/dev/null || echo "2")
if [[ "$PARANOID" -gt 1 ]] && [[ "$EUID" -ne 0 ]]; then
    warn "perf_event_paranoid=$PARANOID，LBR 采样需要:"
    warn "  echo 1 | sudo tee /proc/sys/kernel/perf_event_paranoid"
fi

# loop_main.c 必须存在
[[ -f "$LOOP_MAIN" ]] || { echo "Error: $LOOP_MAIN 不存在" >&2; exit 1; }

# 目录创建
mkdir -p "$OUT_DIR" "$DATA_DIR" "$PROF_DIR" log

# 预编译 loop_main.o（C 模式和 C++ 模式各一份）
LOOP_OBJ_C="$OUT_DIR/loop_main_c.o"
LOOP_OBJ_CXX="$OUT_DIR/loop_main_cxx.o"

$CC  -O2 -c "$LOOP_MAIN" -o "$LOOP_OBJ_C" 2>/dev/null \
    || { echo "Error: loop_main.c (C 模式) 编译失败" >&2; exit 1; }
# C++ 模式：用 -x c++ 强制 C++ 语言，使 bench_entry 符号具有 C++ linkage
$CXX -O2 -x c++ -c "$LOOP_MAIN" -o "$LOOP_OBJ_CXX" 2>/dev/null \
    || LOOP_OBJ_CXX=""  # CXX 不可用时留空，C++ 程序将被跳过

info "loop_main (C):   $LOOP_OBJ_C"
[[ -n "$LOOP_OBJ_CXX" ]] && info "loop_main (C++): $LOOP_OBJ_CXX"

# ── 辅助函数 ─────────────────────────────────────────────────────────────────

# 从基准 stderr 输出里提取执行次数
extract_executions() {
    echo "$1" | grep -oP '(?<=BENCH_EXECUTIONS=)\d+' || echo "0"
}

# 判断目录是否为 C++ 程序（含 .cpp 文件）
is_cpp_dir() {
    local dir="$1"
    find "$dir" -maxdepth 1 \( -name "*.cpp" -o -name "*.cxx" -o -name "*.cc" \) \
        -print -quit 2>/dev/null | grep -q .
}

# 找出目录中含有 int main( 的首个源文件
find_main_file() {
    local dir="$1"
    # 按文件名排序，找第一个包含 main 的文件，优先 C 再 C++
    {
        find "$dir" -maxdepth 1 -name "*.c"                                 | sort
        find "$dir" -maxdepth 1 \( -name "*.cpp" -o -name "*.cc" -o -name "*.cxx" \) | sort
    } | while IFS= read -r f; do
        if grep -qP '\bint\s+main\s*\(' "$f" 2>/dev/null; then
            echo "$f"
            return 0
        fi
    done
    return 1
}

# 编译多文件基准为 BIN_V1（含 loop_main 集成，加 --emit-relocs）
# 参数: <prog_dir> <bench_name> <output_binary> [opt_flags]
# 返回: 0 成功，非 0 失败
compile_bench() {
    local dir="$1"
    local name="$2"
    local out="$3"
    local opt_flags="${4:-$OPT_V1}"

    # 找到 main 文件
    local main_file
    main_file=$(find_main_file "$dir") || return 1

    local is_cpp=0
    is_cpp_dir "$dir" && is_cpp=1

    # 如果是 C++ 但 loop_main.o (C++ 模式) 未编译成功，跳过
    if [[ "$is_cpp" -eq 1 && -z "$LOOP_OBJ_CXX" ]]; then
        warn "$name: C++ 程序但 C++ loop_main 编译失败，跳过"
        return 1
    fi

    # 收集所有源文件（同一目录下的，不递归）
    local -a srcs=()
    while IFS= read -r -d '' f; do
        srcs+=("$f")
    done < <(find "$dir" -maxdepth 1 \
        \( -name "*.c" -o -name "*.cpp" -o -name "*.cc" -o -name "*.cxx" \) \
        -not -name "*.test.*" \
        -print0 | sort -z)

    local ns="${#srcs[@]}"
    if [[ "$ns" -eq 0 ]]; then
        return 1
    fi
    if [[ "$ns" -gt "$MAX_SRCS" ]]; then
        warn "$name: 源文件过多 ($ns > $MAX_SRCS)，跳过"
        return 1
    fi

    local obj_dir="$OUT_DIR/obj_${name}"
    mkdir -p "$obj_dir"

    # 选择编译器和 loop_main.o
    local COMPILER LOOP_OBJ
    if [[ "$is_cpp" -eq 1 ]]; then
        COMPILER="$CXX"
        LOOP_OBJ="$LOOP_OBJ_CXX"
    else
        COMPILER="$CC"
        LOOP_OBJ="$LOOP_OBJ_C"
    fi

    # 逐文件编译成 .o
    local -a objs=()
    for src in "${srcs[@]}"; do
        local bn
        bn=$(basename "$src")
        local obj="$obj_dir/${bn%.*}.o"

        local extra_flag=""
        if [[ "$src" == "$main_file" ]]; then
            extra_flag="-Dmain=bench_entry"
        fi

        if ! $COMPILER $opt_flags $extra_flag \
            -I"$dir" -I"$PROJECT_ROOT/llvm-test-suite" \
            -c "$src" -o "$obj" 2>/dev/null; then
            rm -rf "$obj_dir"
            return 1
        fi
        objs+=("$obj")
    done

    # 链接（加 --emit-relocs）
    if ! $COMPILER $opt_flags $EMIT_RELOCS \
        "${objs[@]}" "$LOOP_OBJ" \
        -lm -lpthread \
        -o "$out" 2>/dev/null; then
        rm -rf "$obj_dir"
        return 1
    fi

    rm -rf "$obj_dir"
    return 0
}

# BOLT 优化流程：BIN_V1 → BIN_V2
# 参数: <bench_name> <bin_v1> <bin_v2>
# 返回: 0 成功，非 0 失败
bolt_optimize() {
    local name="$1"
    local bin_v1="$2"
    local bin_v2="$3"

    local perf_data="$PROF_DIR/${name}.perf.data"
    local bolt_fdata="$PROF_DIR/${name}.fdata"
    local bolt_log="$PROF_DIR/${name}.bolt.log"

    # ── 阶段 A：perf LBR 采样（在 loop_main 驱动下运行 v1）──────────────────
    # 运行 v1 BOLT_PROF_WIN 秒，同时 perf 以 LBR 模式采样
    if ! perf record \
            -e cycles:u -j any,u \
            -o "$perf_data" \
            -- "$bin_v1" "$BOLT_PROF_WIN" >/dev/null 2>&1; then
        warn "$name: perf LBR 采样失败，尝试插桩回退..."

        # ── 回退：llvm-bolt --instrument 模式 ────────────────────────────────
        if ! llvm-bolt "$bin_v1" -instrument -o "${bin_v1}.instr" 2>/dev/null; then
            err "$name: BOLT 插桩也失败，跳过 BOLT 变体"
            return 1
        fi
        "${bin_v1}.instr" "$BOLT_PROF_WIN" >/dev/null 2>&1 || true
        # 插桩模式产生的 fdata 文件名由 BOLT 决定（通常为 /tmp/prof.fdata）
        for fdata_cand in /tmp/prof.fdata /tmp/bolt-fdata /tmp/merged.fdata; do
            [[ -f "$fdata_cand" ]] && cp "$fdata_cand" "$bolt_fdata" && break
        done
        rm -f "${bin_v1}.instr"
        if [[ ! -f "$bolt_fdata" ]]; then
            err "$name: 未找到插桩生成的 fdata，跳过"
            return 1
        fi
    else
        # ── 阶段 B：perf.data → bolt.fdata 转换 ─────────────────────────────
        if ! perf2bolt \
                -p "$perf_data" \
                -o "$bolt_fdata" \
                "$bin_v1" >"$bolt_log" 2>&1; then
            err "$name: perf2bolt 转换失败 (见 $bolt_log)"
            rm -f "$perf_data"
            return 1
        fi
        rm -f "$perf_data"  # 释放空间（fdata 更小）
    fi

    # ── 阶段 C：llvm-bolt 重排代码布局 → v2 ─────────────────────────────────
    # shellcheck disable=SC2086
    if ! llvm-bolt "$bin_v1" \
            -data "$bolt_fdata" \
            -o "$bin_v2" \
            $BOLT_FLAGS \
            >"$bolt_log" 2>&1; then
        err "$name: llvm-bolt 优化失败 (见 $bolt_log)"
        rm -f "$bolt_fdata"
        return 1
    fi

    # 保留 fdata 以便调试；删除可运行: rm -f "$bolt_fdata"
    return 0
}

# 运行基准并获取执行次数（用于标签生成）
# 参数: <binary_path> <window_secs>
# 输出: 执行次数整数（stdout）
label_run() {
    local bin="$1"
    local window="$2"
    local out
    out=$(timeout $((window + 5)) "$bin" "$window" 2>&1 || true)
    extract_executions "$out"
}

# 附加 pmu_monitor 采集 PMU 时序数据
# 参数: <binary_path> <output_csv_path>
# 返回: 0 成功，非 0 失败
collect_pmu() {
    local bin="$1"
    local out_csv="$2"

    BENCH_PID_V=""
    MON_PID_V=""

    # 启动基准（静默输出）
    "$bin" "$PMU_WINDOW" >/dev/null 2>/dev/null &
    BENCH_PID_V=$!

    sleep 0.5

    if ! kill -0 "$BENCH_PID_V" 2>/dev/null; then
        warn "  基准进程已在 0.5s 内退出"
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
        warn "  log/pmu_monitor.csv 未生成"
        return 1
    fi
}

# ── 主扫描循环 ───────────────────────────────────────────────────────────────
bold "======================================================"
bold "   MultiSource × BOLT 训练集采集"
bold "======================================================"
info "扫描根目录: $BENCH_ROOTS"
info "BOLT 可用: $([ $HAS_BOLT -eq 1 ] && echo '是' || echo '否（仅编译 v1）')"
info "标签窗口: ${LABEL_WINDOW}s | PMU 窗口: ${PMU_WINDOW}s | BOLT 剖析窗口: ${BOLT_PROF_WIN}s"
echo

# 清空旧 manifest
> "$MANIFEST"

# 使用进程替换收集所有需要扫描的程序目录
# 扫描策略：找出所有"含有带 main() 源文件"的最浅层目录（叶目录）
declare -A SEEN_DIRS  # 避免重复处理

while IFS= read -r PROG_DIR; do
    # 防止将父目录重复处理（当子目录也在列表中时）
    for seen in "${!SEEN_DIRS[@]}"; do
        if [[ "$PROG_DIR" == "$seen"* ]]; then
            continue 2
        fi
    done
    SEEN_DIRS["$PROG_DIR"]=1

    # 用目录名作为 bench 名称（取最后两级：Parent/Name）
    REL="${PROG_DIR#$PROJECT_ROOT/llvm-test-suite/}"
    # 将路径中 / 替换为 _ 作为 bench 名
    BENCH_NAME="${REL//\//_}"
    # 去掉 MultiSource_Benchmarks_ 或 MultiSource_Applications_ 前缀简化名称
    BENCH_NAME="${BENCH_NAME#MultiSource_Benchmarks_}"
    BENCH_NAME="${BENCH_NAME#MultiSource_Applications_}"

    ((COUNT_TOTAL++)) || true
    info "[$COUNT_TOTAL] $BENCH_NAME"
    info "  目录: $REL"

    BIN_V1="$OUT_DIR/${BENCH_NAME}_v1"
    BIN_V2="$OUT_DIR/${BENCH_NAME}_v2"

    # ── 1. 编译 v1 ──────────────────────────────────────────────────────────
    if ! compile_bench "$PROG_DIR" "$BENCH_NAME" "$BIN_V1"; then
        err "$BENCH_NAME: 编译失败，跳过"
        ((COUNT_SKIP_COMPILE++)) || true
        continue
    fi
    pass "  v1 编译完成: $BIN_V1"

    # ── 2. BOLT 流程：v1 → v2 ───────────────────────────────────────────────
    if [[ "$HAS_BOLT" -eq 1 ]]; then
        if ! bolt_optimize "$BENCH_NAME" "$BIN_V1" "$BIN_V2"; then
            err "$BENCH_NAME: BOLT 优化失败，跳过"
            ((COUNT_SKIP_BOLT++)) || true
            continue
        fi
        pass "  v2 BOLT 完成: $BIN_V2"
    else
        # BOLT 工具不可用：本脚本以 BOLT 变体为核心训练信号，无法生成有效 v2
        # 请安装 llvm-bolt / perf2bolt 后重新运行，或改用 collect_dataset.sh (-O2 vs -O3)
        err "$BENCH_NAME: llvm-bolt 不可用，无法生成 BOLT 变体，跳过"
        ((COUNT_SKIP_BOLT++)) || true
        continue
    fi

    # ── 3. 标签生成 ─────────────────────────────────────────────────────────
    N_V1=$(label_run "$BIN_V1" "$LABEL_WINDOW")
    N_V2=$(label_run "$BIN_V2" "$LABEL_WINDOW")

    info "  执行次数: v1(base)=$N_V1  v2(bolt)=$N_V2"

    if [[ "$N_V1" -lt "$MIN_EXECS" && "$N_V2" -lt "$MIN_EXECS" ]]; then
        err "$BENCH_NAME: 执行次数过少（均 < $MIN_EXECS），程序运行时间超过窗口，跳过"
        ((COUNT_SKIP_LABEL++)) || true
        continue
    fi
    if [[ "$N_V1" -eq "$N_V2" ]]; then
        warn "$BENCH_NAME: v1=$N_V1 = v2=$N_V2，标签不可分辨，跳过"
        ((COUNT_SKIP_LABEL++)) || true
        continue
    fi

    LABEL=$(( N_V1 > N_V2 ? 1 : 0 ))
    info "  标签: $LABEL  ($([ $LABEL -eq 1 ] && echo 'v1 基准 > v2 BOLT' || echo 'v2 BOLT > v1 基准'))"

    # ── 4. PMU 采集 v1 ──────────────────────────────────────────────────────
    CSV_V1="$DATA_DIR/${BENCH_NAME}_v1.csv"
    if ! collect_pmu "$BIN_V1" "$CSV_V1"; then
        err "$BENCH_NAME: v1 PMU 采集失败，跳过"
        ((COUNT_SKIP_PMU++)) || true
        continue
    fi
    info "  v1 CSV: ${CSV_V1#$PROJECT_ROOT/}"

    # ── 5. PMU 采集 v2 ──────────────────────────────────────────────────────
    CSV_V2="$DATA_DIR/${BENCH_NAME}_v2.csv"
    if ! collect_pmu "$BIN_V2" "$CSV_V2"; then
        err "$BENCH_NAME: v2 PMU 采集失败，跳过"
        ((COUNT_SKIP_PMU++)) || true
        continue
    fi
    info "  v2 CSV: ${CSV_V2#$PROJECT_ROOT/}"

    # ── 6. 写 manifest ──────────────────────────────────────────────────────
    printf '{"program":"%s","label":%d,"n_v1":%d,"n_v2":%d,"opt_v1":"O3-base","opt_v2":"%s","v1_csv":"%s","v2_csv":"%s","source":"%s","method":"bolt"}\n' \
        "$BENCH_NAME" "$LABEL" "$N_V1" "$N_V2" \
        "$([ $HAS_BOLT -eq 1 ] && echo 'bolt' || echo 'O2')" \
        "${CSV_V1#$PROJECT_ROOT/}" \
        "${CSV_V2#$PROJECT_ROOT/}" \
        "$REL" \
        >> "$MANIFEST"

    pass "$BENCH_NAME 完成"
    ((COUNT_OK++)) || true
    echo

done < <(
    # 对每个根目录下，找到所有含有 main() 源文件的"叶目录"
    for ROOT_REL in $BENCH_ROOTS; do
        ROOT_ABS="$PROJECT_ROOT/$ROOT_REL"
        [[ -d "$ROOT_ABS" ]] || { warn "目录不存在: $ROOT_ABS"; continue; }

        # 找到含 main() 源文件的所有目录（按深度排序，浅的优先）
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
printf "  扫描总数:   %d\n"  "$COUNT_TOTAL"
printf "  成功入库:   %d\n"  "$COUNT_OK"
printf "  编译失败:   %d\n"  "$COUNT_SKIP_COMPILE"
printf "  BOLT 失败:  %d\n"  "$COUNT_SKIP_BOLT"
printf "  标签无效:   %d\n"  "$COUNT_SKIP_LABEL"
printf "  PMU 失败:   %d\n"  "$COUNT_SKIP_PMU"
echo "  数据集清单: $MANIFEST"
bold "======================================================"
