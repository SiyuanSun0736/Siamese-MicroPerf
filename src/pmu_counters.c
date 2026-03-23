/*
 * pmu_counters.c — perf_event 计数器管理实现
 */

#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/ioctl.h>
#include <linux/perf_event.h>
#include <asm/unistd.h>
#include <stdint.h>
#include <errno.h>

#include "pmu_counters.h"

/* ── 计数器数组定义 ─────────────────────────────────────────────────────── */

perf_counter_t pmu_counters[PMU_NUM_COUNTERS] = {
    {-1, "inst_retired.any",        0, 0, 0, 0, 0, 0, 0},
    {-1, "L1-icache-load-misses",   0, 0, 0, 0, 0, 0, 0},
    {-1, "iTLB-loads",              0, 0, 0, 0, 0, 0, 0},
    {-1, "iTLB-load-misses",        0, 0, 0, 0, 0, 0, 0},
    {-1, "branch-instructions",     0, 0, 0, 0, 0, 0, 0},
    {-1, "branch-misses",           0, 0, 0, 0, 0, 0, 0},
};

/* ── 内部工具 ────────────────────────────────────────────────────────────── */

static int perf_event_open(struct perf_event_attr *hw, pid_t pid,
                           int cpu, int group_fd, unsigned long flags)
{
    return (int)syscall(__NR_perf_event_open, hw, pid, cpu, group_fd, flags);
}

/*
 * open_counter — 打开一个计数器。
 *   group_fd : 组长 fd（-1 = 独立计数器 / 本身为组长）
 * 同组计数器由内核同步调度，时间窗口完全一致，使得
 * MPKI = ΔC/ΔI 中的复用缩放比精确抵消。
 */
static int open_counter(int idx, struct perf_event_attr *pe,
                        pid_t pid, int cpu, int group_fd)
{
    pe->read_format = PERF_FORMAT_TOTAL_TIME_ENABLED |
                      PERF_FORMAT_TOTAL_TIME_RUNNING;
    int fd = perf_event_open(pe, pid, cpu, group_fd, 0);
    if (fd < 0)
        return 0;
    pmu_counters[idx].fd      = fd;
    pmu_counters[idx].enabled = 1;
    ioctl(fd, PERF_EVENT_IOC_ENABLE, 0);
    return 1;
}

/* ── 公共接口 ────────────────────────────────────────────────────────────── */

int pmu_init(pid_t pid, int cpu)
{
    int active = 0;
    int idx    = 0;
    struct perf_event_attr pe;

#define RESET_PE() do {                     \
        memset(&pe, 0, sizeof(pe));         \
        pe.size     = sizeof(pe);           \
        pe.disabled = 1;                    \
        pe.inherit  = 1;                    \
    } while (0)

    /*
     * 分组策略（物理 PMU 寄存器上限 4 个/组）：
     *
     *   Group A（4）: inst_retired.any [组长]
     *                  L1-icache-load-misses
     *                  iTLB-loads
     *                  iTLB-load-misses
     *
     *   Group B（2）: branch-instructions [组长]
     *                  branch-misses
     *
     * 同组所有计数器共享相同的 time_enabled / time_running，
     * 因此 MPKI = (ΔC × scale) / (ΔI × scale) × 1000 中
     * 的复用缩放比精确抵消，消除负值与脉冲放大失真。
     */

    /* ── Group A ─────────────────────────────────────────────────────── */

    /* inst_retired.any — RAW 0x00C0 — Group A 组长 */
    RESET_PE();
    pe.type   = PERF_TYPE_RAW;
    pe.config = 0x00C0;
    active += open_counter(idx++, &pe, pid, cpu, -1);
    int group_a_fd = pmu_counters[0].fd;   /* 若打开失败则为 -1，退化为独立模式 */

    /* L1-icache-load-misses — HW_CACHE — Group A 成员 */
    RESET_PE();
    pe.type   = PERF_TYPE_HW_CACHE;
    pe.config = PERF_COUNT_HW_CACHE_L1I
                | (PERF_COUNT_HW_CACHE_OP_READ    << 8)
                | (PERF_COUNT_HW_CACHE_RESULT_MISS << 16);
    active += open_counter(idx++, &pe, pid, cpu, group_a_fd);

    /* iTLB-loads — HW_CACHE — Group A 成员 */
    pe.config = PERF_COUNT_HW_CACHE_ITLB
                | (PERF_COUNT_HW_CACHE_OP_READ      << 8)
                | (PERF_COUNT_HW_CACHE_RESULT_ACCESS << 16);
    active += open_counter(idx++, &pe, pid, cpu, group_a_fd);

    /* iTLB-load-misses — HW_CACHE — Group A 成员 */
    pe.config = PERF_COUNT_HW_CACHE_ITLB
                | (PERF_COUNT_HW_CACHE_OP_READ    << 8)
                | (PERF_COUNT_HW_CACHE_RESULT_MISS << 16);
    active += open_counter(idx++, &pe, pid, cpu, group_a_fd);

    /* ── Group B ─────────────────────────────────────────────────────── */

    /* branch-instructions — HARDWARE — Group B 组长 */
    RESET_PE();
    pe.type   = PERF_TYPE_HARDWARE;
    pe.config = PERF_COUNT_HW_BRANCH_INSTRUCTIONS;
    active += open_counter(idx++, &pe, pid, cpu, -1);
    int group_b_fd = pmu_counters[4].fd;

    /* branch-misses — HARDWARE — Group B 成员 */
    pe.config = PERF_COUNT_HW_BRANCH_MISSES;
    active += open_counter(idx++, &pe, pid, cpu, group_b_fd);

#undef RESET_PE
    return active;
}

void pmu_read(void)
{
    struct {
        uint64_t value;
        uint64_t time_enabled;
        uint64_t time_running;
    } cnt;

    for (int i = 0; i < PMU_NUM_COUNTERS; i++) {
        if (!pmu_counters[i].enabled) {
            pmu_counters[i].count        = 0;
            pmu_counters[i].time_enabled = 0;
            pmu_counters[i].time_running = 0;
            continue;
        }
        if (read(pmu_counters[i].fd, &cnt, sizeof(cnt)) != (ssize_t)sizeof(cnt)) {
            fprintf(stderr, "read counter[%d] failed: %s\n", i, strerror(errno));
            pmu_counters[i].count = 0;
            continue;
        }

        /* 计算本区间增量（累计值 − 上一轮快照）
         * 这样可以正确处理：
         *   1) inherit=1 时子进程累计量不被 ioctl RESET 覆盖的问题
         *   2) time_enabled/time_running 不受 RESET 影响、始终累计的问题
         */
        uint64_t delta_val     = cnt.value        - pmu_counters[i].prev_raw;
        uint64_t delta_enabled = cnt.time_enabled - pmu_counters[i].prev_time_enabled;
        uint64_t delta_running = cnt.time_running - pmu_counters[i].prev_time_running;

        /* 更新快照供下一轮使用 */
        pmu_counters[i].prev_raw          = cnt.value;
        pmu_counters[i].prev_time_enabled = cnt.time_enabled;
        pmu_counters[i].prev_time_running = cnt.time_running;

        /* 保存本区间时间增量（供 -E 选项输出）*/
        pmu_counters[i].time_enabled = delta_enabled;
        pmu_counters[i].time_running = delta_running;

        /* 应用本区间复用缩放比 */
        if (delta_running > 0 && delta_running < delta_enabled) {
            double scale = (double)delta_enabled / delta_running;
            pmu_counters[i].count = (uint64_t)(delta_val * scale);
        } else {
            pmu_counters[i].count = delta_val;
        }
    }
}

void pmu_reset(void)
{
    struct {
        uint64_t value;
        uint64_t time_enabled;
        uint64_t time_running;
    } cnt;

    /* 读取当前累计值并锁存为 prev 快照，作为下一个采样区间的差值基线。
     * 不再向内核发送 PERF_EVENT_IOC_RESET：
     *   - RESET 无法覆盖 inherit=1 子进程的继承累计量；
     *   - time_enabled / time_running 字段本就不受 RESET 影响。
     */
    for (int i = 0; i < PMU_NUM_COUNTERS; i++) {
        if (!pmu_counters[i].enabled)
            continue;
        if (read(pmu_counters[i].fd, &cnt, sizeof(cnt)) != (ssize_t)sizeof(cnt))
            continue;
        pmu_counters[i].prev_raw          = cnt.value;
        pmu_counters[i].prev_time_enabled = cnt.time_enabled;
        pmu_counters[i].prev_time_running = cnt.time_running;
    }
}

void pmu_close(void)
{
    for (int i = 0; i < PMU_NUM_COUNTERS; i++) {
        if (pmu_counters[i].fd >= 0) {
            close(pmu_counters[i].fd);
            pmu_counters[i].fd      = -1;
            pmu_counters[i].enabled = 0;
        }
    }
}
