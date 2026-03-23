#!/bin/bash
# =============================================================================
# extract_elf.sh — 从 llvm-test-suite build 目录提取 ELF 文件到 train_set
# =============================================================================
#
# 功能：
#   扫描 llvm-test-suite/{BUILD_DIR}/MultiSource/{MS_SUBDIR}/ 下的 ELF 可执行
#   文件，并将其复制到 OUT_DIR，文件名由目录路径各级组件以 "_" 拼接，末尾附加
#   版本后缀 _{VERSION}。
#
#   例：MultiSource/Applications/aha/aha       → bin/aha_v1
#       MultiSource/Applications/JM/ldecod/ldecod → bin/JM_ldecod_v1
#       MultiSource/Applications/ALAC/encode/...  → bin/ALAC_encode_v1
#
# 用法：
#   bash train_set/extract_elf.sh [选项]
#
# 选项：
#   -b <BUILD_DIR>   build 目录名（默认：build-O1-g）
#   -s <MS_SUBDIR>   MultiSource 子目录（默认：Applications）
#   -v <VERSION>     版本后缀（默认：O1-g）
#   -o <OUT_DIR>     ELF 输出目录（默认：train_set/bin/$VERSION）
#   -t <TEST_DIR>    测试文件输出目录（默认：train_set/test/$VERSION）
#   -n               仅预览，不实际复制（dry-run）
#   -h               显示帮助并退出
#
# 示例：
#   bash train_set/extract_elf.sh
#   bash train_set/extract_elf.sh -b build-O3-g -v O3-g
#   bash train_set/extract_elf.sh -b build-O3-bolt -s Benchmarks -v O3-bolt -o train_set/bin_bolt
#   bash train_set/extract_elf.sh -b build-O1-g -s Applications -v O1-g -n
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# ── 默认参数 ──────────────────────────────────────────────────────────────────
VERSION="O2-bolt"
BUILD_DIR=""       # 依赖 VERSION，在 getopts 后填充
MS_SUBDIR="Benchmarks"
# MultiSource 目录名（允许用户修改，例如 "MultiSource" 或 "MultiSource-custom"）
MULTI_DIR="MultiSource"
OUT_DIR=""         # 依赖 VERSION，在 getopts 后填充
TEST_DIR=""        # 依赖 VERSION，在 getopts 后填充
DRY_RUN=0
# 命名方案：
#  - dir  : 使用所在目录层级（默认，保持与现有脚本一致）
#  - full : 在目录层级后追加文件名（例如 JM_ldecod_ldecod_v1）
NAMING_SCHEME="dir"

# ── 参数解析 ──────────────────────────────────────────────────────────────────
usage() {
    grep '^#' "$0" | sed 's/^# \{0,1\}//' | sed -n '2,30p'
    exit 0
}

while getopts "b:s:m:v:o:t:f:nh" opt; do
    case $opt in
        b) BUILD_DIR="$OPTARG" ;;
        s) MS_SUBDIR="$OPTARG" ;;
        m) MULTI_DIR="$OPTARG" ;;
        v) VERSION="$OPTARG" ;;
        o) OUT_DIR="$OPTARG" ;;
        t) TEST_DIR="$OPTARG" ;;
        f) NAMING_SCHEME="$OPTARG" ;;
        n) DRY_RUN=1 ;;
        h) usage ;;
        *) echo "未知选项：-$OPTARG" >&2; exit 1 ;;
    esac
done

# 用最终 VERSION 填充未显式指定的依赖项
[[ -z "$BUILD_DIR" ]] && BUILD_DIR="build-$VERSION"
[[ -z "$OUT_DIR"   ]] && OUT_DIR="$SCRIPT_DIR/bin/$VERSION"
[[ -z "$TEST_DIR"  ]] && TEST_DIR="$SCRIPT_DIR/test/$VERSION"

SEARCH_ROOT="$PROJECT_ROOT/llvm-test-suite/$BUILD_DIR/${MULTI_DIR}/$MS_SUBDIR"

# ── 前置检查 ──────────────────────────────────────────────────────────────────
if [[ ! -d "$SEARCH_ROOT" ]]; then
    echo "错误：扫描目录不存在：$SEARCH_ROOT" >&2
    exit 1
fi

if ! command -v file &>/dev/null; then
    echo "错误：需要 'file' 命令来识别 ELF 文件" >&2
    exit 1
fi

if [[ $DRY_RUN -eq 0 ]]; then
    mkdir -p "$OUT_DIR"
    mkdir -p "$TEST_DIR"
fi

# ── 主逻辑 ────────────────────────────────────────────────────────────────────
count=0
skip=0

echo "MultiSource: $MULTI_DIR"
echo "扫描目录  : $SEARCH_ROOT"
echo "ELF目录   : $OUT_DIR"
echo "测试目录  : $TEST_DIR"
echo "版本后缀  : _${VERSION}"
echo "命名方案  : $NAMING_SCHEME"
[[ $DRY_RUN -eq 1 ]] && echo "模式      : dry-run（仅预览）"
echo "---"

while IFS= read -r -d '' elf_path; do
    # 仅处理 ELF 可执行文件（过滤 shell 脚本、数据文件等）
    if ! file "$elf_path" 2>/dev/null | grep -qE "ELF.*(executable|shared object)"; then
        (( skip++ )) || true
        continue
    fi

    # 计算相对于 SEARCH_ROOT 的路径，例如：aha/aha 或 JM/ldecod/ldecod
    rel_path="${elf_path#"$SEARCH_ROOT"/}"

    # 程序名生成：支持两种命名方案
    dir_part="$(dirname "$rel_path")"
    base_name="$(basename "$rel_path")"
    if [[ "$dir_part" == "." ]]; then
        prog_name_part="${base_name%.*}"
    else
        prog_name_part="${dir_part//\//_}"
    fi

    if [[ "$NAMING_SCHEME" == "full" ]]; then
        # 例如：JM_ldecod_ldecod
        prog_name="${prog_name_part}_$(basename "$rel_path")"
    else
        # 默认：仅目录层级，例如 JM_ldecod
        prog_name="${prog_name_part}"
    fi

    dest="$OUT_DIR/${prog_name}_${VERSION}"

    if [[ $DRY_RUN -eq 1 ]]; then
        echo "[预览] $rel_path  →  ${prog_name}_${VERSION}"
    else
        cp "$elf_path" "$dest"
        echo "已复制：$rel_path  →  ${prog_name}_${VERSION}"
    fi
    (( count++ )) || true

    # ── 复制 .test 文件 ───────────────────────────────────────────────────────
    elf_dir="$(dirname "$elf_path")"
    test_src="$elf_dir/${base_name}.test"
    test_dest_dir="$TEST_DIR/${prog_name}"

    if [[ -f "$test_src" ]]; then
        if [[ $DRY_RUN -eq 1 ]]; then
            echo "[预览] 测试文件 : ${base_name}.test  →  ${prog_name}/${base_name}.test"
        else
            mkdir -p "$test_dest_dir"
            cp "$test_src" "$test_dest_dir/${base_name}.test"
            echo "  测试文件：${base_name}.test  →  ${prog_name}/${base_name}.test"
        fi

        # 扫描 RUN: 行中引用的输出目录（如 Output/），在目标测试目录中预先创建
        while IFS= read -r run_line; do
            # 提取所有形如 "SomeDir/" 的路径前缀（至少一级目录）
            while read -r out_dir; do
                [[ -z "$out_dir" ]] && continue
                if [[ $DRY_RUN -eq 1 ]]; then
                    echo "[预览] 输出目录 : ${prog_name}/${out_dir}"
                else
                    mkdir -p "$test_dest_dir/$out_dir"
                    echo "  输出目录：${prog_name}/${out_dir}"
                fi
            done < <(grep -oP '(?<!\S)[A-Za-z][A-Za-z0-9_.-]*/(?=[^\s/])' <<< "$run_line" \
                     | sed 's|/$||' | sort -u)
        done < <(grep '^RUN:' "$test_src")
    fi

    # ── 复制运行时数据文件 ────────────────────────────────────────────────────
    # 排除构建产物：CMakeFiles/、Output/、cmake_install.cmake、Makefile、
    #               *.link.time、*.size、*.test 以及 ELF 可执行文件本身
    while IFS= read -r -d '' data_file; do
        if file "$data_file" 2>/dev/null | grep -qE "ELF.*(executable|shared object)"; then
            continue
        fi
        data_rel="${data_file#"$elf_dir"/}"
        if [[ $DRY_RUN -eq 1 ]]; then
            echo "[预览] 运行时文件: $data_rel  →  ${prog_name}/$data_rel"
        else
            mkdir -p "$(dirname "$test_dest_dir/$data_rel")"
            if [[ -d "$data_file" ]]; then
                # 符号链接指向目录，递归复制
                cp -r "$data_file" "$test_dest_dir/$data_rel"
            else
                cp "$data_file" "$test_dest_dir/$data_rel"
            fi
            echo "  运行时文件：$data_rel  →  ${prog_name}/$data_rel"
        fi
    done < <(find "$elf_dir" \
        -not -path "*/CMakeFiles/*" \
        -not -path "*/Output/*" \
        -not -name "cmake_install.cmake" \
        -not -name "Makefile" \
        -not -name "*.link.time" \
        -not -name "*.size" \
        -not -name "*.test" \
        \( -type f -o -type l \) -print0 | sort -z)

    # ── 扫描配置文件中的输出目录引用 ───────────────────────────────────────────
    # 某些程序（如 JM/lencod）将输出路径写在 .cfg 文件中而非 RUN: 行，
    # 需额外扫描所有 .cfg 文件，提取引号内含 "/" 的路径并创建其目录部分。
    while IFS= read -r -d '' cfg_file; do
        while read -r cfg_dir; do
            [[ -z "$cfg_dir" || "$cfg_dir" == "." ]] && continue
            if [[ $DRY_RUN -eq 1 ]]; then
                echo "[预览] 配置输出目录: ${prog_name}/${cfg_dir}"
            else
                mkdir -p "$test_dest_dir/$cfg_dir"
                echo "  配置输出目录：${prog_name}/${cfg_dir}"
            fi
        done < <(grep -oP '"[^"]+/[^"]+"' "$cfg_file" \
                 | tr -d '"' | xargs -I{} dirname {} \
                 | grep -v '^\.' | grep -vP '\.' | sort -u)
    done < <(find "$elf_dir" \
        -not -path "*/CMakeFiles/*" \
        -not -path "*/Output/*" \
        -name "*.cfg" \( -type f -o -type l \) -print0)

done < <(find "$SEARCH_ROOT" -type f -executable -print0 | sort -z)

echo "---"
if [[ $DRY_RUN -eq 1 ]]; then
    echo "预览完成：共找到 $count 个 ELF 文件，跳过 $skip 个非 ELF 可执行文件。"
else
    echo "完成：共复制 $count 个 ELF 文件，跳过 $skip 个非 ELF 可执行文件。"
fi
