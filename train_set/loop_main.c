/*
 * loop_main.c — 基准程序循环驱动器
 *
 * 将 LLVM test suite 中的单文件基准（编译时以 -Dmain=bench_entry 重命名其
 * main 函数）包裹进一个固定时长的循环，使基准进程在整个 PMU 采集窗口内
 * 保持稳定的 PID，方便 pmu_monitor 按 PID 连续监控。
 *
 * 用法（编译时链接）：
 *   clang -O3 -Dmain=bench_entry benchmark.c loop_main.c -lm -o bench_O3
 *   ./bench_O3 [duration_secs]
 *
 * 输出（写入 stderr，供 collect_dataset.sh 解析）：
 *   BENCH_EXECUTIONS=<count>
 *   BENCH_ELAPSED_MS=<ms>
 *
 * 注意：
 *   - 迭代之间不重置全局状态。对于某些基准（如排序），第 2 次以后的迭代
 *     输入可能已处于有序/已处理状态，因此 execution count 反映的是
 *     "热路径" 吞吐量，这对微架构训练信号仍然有效。
 *   - 若基准内部调用 exit()，进程会直接退出；collect_dataset.sh 会将此
 *     视为编译/运行失败并跳过该基准。
 */

#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <stdint.h>

/*
 * 由基准源文件提供（通过 -Dmain=bench_entry 重命名后编译为独立 .o）。
 * 注意：loop_main.c 必须在不带 -Dmain=bench_entry 的情况下单独编译，
 * 否则本文件的 main() 也会被重命名为 bench_entry，造成符号冲突。
 */
extern int bench_entry(int argc, char **argv);

/* 返回 CLOCK_MONOTONIC 毫秒时间戳 */
static uint64_t now_ms(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000ULL + (uint64_t)(ts.tv_nsec / 1000000ULL);
}

int main(int argc, char **argv)
{
    int duration_secs = 10;  /* 默认采集窗口：10 秒 */
    if (argc > 1) {
        duration_secs = atoi(argv[1]);
        if (duration_secs <= 0) {
            fprintf(stderr, "Usage: %s [duration_secs]\n", argv[0]);
            return 1;
        }
    }

    uint64_t deadline_ms = now_ms() + (uint64_t)duration_secs * 1000ULL;
    long     count       = 0;

    /* 将 stdout 重定向到 /dev/null，防止基准输出淹没脚本解析 */
    freopen("/dev/null", "w", stdout);

    do {
        bench_entry(0, NULL);
        count++;
    } while (now_ms() < deadline_ms);

    uint64_t elapsed = now_ms() - (deadline_ms - (uint64_t)duration_secs * 1000ULL);

    /* 输出到 stderr，供脚本用 grep 解析 */
    fprintf(stderr, "BENCH_EXECUTIONS=%ld\n", count);
    fprintf(stderr, "BENCH_ELAPSED_MS=%llu\n", (unsigned long long)elapsed);

    return 0;
}
