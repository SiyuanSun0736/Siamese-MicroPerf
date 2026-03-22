/*
 * loop_main.c — 基准程序循环驱动器
 *
 * 将 LLVM test suite 中的基准包裹进一个固定时长的循环，使基准进程在整个
 * PMU 采集窗口内保持稳定的 PID，方便 pmu_monitor 按 PID 连续监控。
 *
 * 支持两种链接模式（通过编译宏选择）：
 *
 * 模式 A：-Dmain=bench_entry（默认，供 shell 脚本编译使用）
 *   将基准的 main() 重命名为 bench_entry()，本文件的 main() 作为真正入口。
 *   编译示例：
 *     clang -O1 -Dmain=bench_entry benchmark.c loop_main.c -o bench
 *   注意：若用 clang++ 编译 C++ 基准，-Dmain=bench_entry 会产生 C++ 链接的
 *         bench_entry（_Z11bench_entryiPPc），而本文件若以 clang++ 编译同样
 *         会匹配，因此 clang++ 单次命令行编译时两种模式等价。
 *
 * 模式 B：-DLOOP_MAIN_WRAP（供 CMake TEST_SUITE_INLINE_LOOP 使用）
 *   利用链接器 --wrap,main 机制，不需要修改基准源码：
 *     - 本文件提供 __wrap_main()，链接器将所有对 main 的外部引用重定向到此
 *     - 基准原有的 main() 以 __real_main() 符号暴露，本文件循环调用之
 *   编译示例：
 *     CMake: target_link_options(... -Wl,--wrap,main)
 *            target_compile_definitions(loop_main_driver LOOP_MAIN_WRAP=1)
 *   优势：适用于 C 和 C++ 基准，无 C/C++ 链接符号不匹配问题。
 *
 * 输出（写入 stderr，供 collect_dataset.sh 解析）：
 *   BENCH_EXECUTIONS=<count>
 *   BENCH_ELAPSED_MS=<ms>
 *
 * 环境变量：
 *   BENCH_DURATION     采集窗口时长（秒），默认 30。所有命令行参数原样透传
 *                      给基准，不消耗任何 argv。
 *   BENCH_STDIN_FILE   若设置，则每次迭代前重新以只读方式打开该文件作为
 *                      stdin，适用于从 stdin 读取输入的基准程序。
 *
 * 注意：
 *   - 迭代之间不重置全局状态。对于某些基准（如排序），第 2 次以后的迭代
 *     输入可能已处于有序/已处理状态，因此 execution count 反映的是
 *     "热路径" 吞吐量，这对微架构训练信号仍然有效。
 *   - 若基准内部调用 exit()，进程会直接退出；collect_dataset.sh 会将此
 *     视为编译/运行失败并跳过该基准。
 *
 * LBR 兼容性说明：
 *   本驱动将 bench_entry() / __real_main() 在同一进程内循环调用（无 fork/exec），
 *   因此 pmu_monitor 的 LBR ring buffer 可以正常采集分支记录。
 */

#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <stdint.h>
#include <unistd.h>

/*
 * 根据编译模式选择基准入口函数的声明方式：
 *
 * 模式 B（LOOP_MAIN_WRAP）：使用链接器 --wrap,main，基准原 main() 以
 *   __real_main() 暴露。__real_main 始终是 C 链接（main 无 name mangling），
 *   对 C 和 C++ 基准均一致，无符号不匹配问题。
 *
 * 模式 A（默认）：基准以 -Dmain=bench_entry 重命名，调用 bench_entry()。
 *   使用 clang++ 编译 C++ 基准时，bench_entry 为 C++ 链接；本文件若同样
 *   由 clang++ 编译（clang++ 对 .c 文件默认以 C++ 模式处理）则自动匹配。
 */
#ifdef LOOP_MAIN_WRAP
/* 模式 B：--wrap,main 方式 */
extern int __real_main(int argc, char **argv);
#define BENCH_CALL(argc, argv) __real_main((argc), (argv))
#define LOOP_MAIN_ENTRY __wrap_main
#else
/* 模式 A：-Dmain=bench_entry 方式 */
extern int bench_entry(int argc, char **argv);
#define BENCH_CALL(argc, argv) bench_entry((argc), (argv))
#define LOOP_MAIN_ENTRY main
#endif

/* 返回 CLOCK_MONOTONIC 毫秒时间戳 */
static uint64_t now_ms(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000ULL + (uint64_t)(ts.tv_nsec / 1000000ULL);
}

int LOOP_MAIN_ENTRY(int argc, char **argv)
{
    /* 采集窗口时长：优先读取环境变量 BENCH_DURATION，默认 30 秒。
     * 所有命令行参数原样透传给基准，不消耗任何 argv。 */
    int duration_secs = 30;
    const char *dur_env = getenv("BENCH_DURATION");
    if (dur_env && atoi(dur_env) > 0)
        duration_secs = atoi(dur_env);

    /* 所有原始 argv 透传给基准 */
    int   bench_argc = argc;
    char **bench_argv = argv;

    /* 若环境变量 BENCH_STDIN_FILE 设置，则每次迭代前重新打开该文件作为 stdin，
     * 避免第一次迭代消耗完 stdin 后后续迭代立即 EOF。 */
    const char *stdin_file = getenv("BENCH_STDIN_FILE");

    uint64_t deadline_ms = now_ms() + (uint64_t)duration_secs * 1000ULL;
    long     count       = 0;

    /* 将 stdout 重定向到 /dev/null，防止基准输出淹没脚本解析 */
    freopen("/dev/null", "w", stdout);

    do {
        if (stdin_file) {
            if (!freopen(stdin_file, "r", stdin)) {
                fprintf(stderr, "loop_main: cannot reopen stdin from '%s'\n",
                        stdin_file);
                return 1;
            }
        }
        BENCH_CALL(bench_argc, bench_argv);
        count++;
    } while (now_ms() < deadline_ms);

    uint64_t elapsed = now_ms() - (deadline_ms - (uint64_t)duration_secs * 1000ULL);

    /* 输出到 stderr，供脚本用 grep 解析 */
    fprintf(stderr, "BENCH_EXECUTIONS=%ld\n", count);
    fprintf(stderr, "BENCH_ELAPSED_MS=%llu\n", (unsigned long long)elapsed);

    return 0;
}

