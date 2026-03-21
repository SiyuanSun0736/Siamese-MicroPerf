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
    {-1, "inst_retired.any",        0, 0, 0, 0},
    {-1, "L1-icache-load-misses",   0, 0, 0, 0},
    {-1, "iTLB-loads",              0, 0, 0, 0},
    {-1, "iTLB-load-misses",        0, 0, 0, 0},
    {-1, "branch-instructions",     0, 0, 0, 0},
    {-1, "branch-misses",           0, 0, 0, 0},
};

/* ── 内部工具 ────────────────────────────────────────────────────────────── */

static int perf_event_open(struct perf_event_attr *hw, pid_t pid,
                           int cpu, int group_fd, unsigned long flags)
{
    return (int)syscall(__NR_perf_event_open, hw, pid, cpu, group_fd, flags);
}

/*
 * open_counter — 以独立模式（不分组）打开一个计数器，
 * 附带 PERF_FORMAT_TOTAL_TIME_ENABLED | PERF_FORMAT_TOTAL_TIME_RUNNING
 * 用于后续多路复用缩放。
 */
static int open_counter(int idx, struct perf_event_attr *pe, pid_t pid, int cpu)
{
    pe->read_format = PERF_FORMAT_TOTAL_TIME_ENABLED |
                      PERF_FORMAT_TOTAL_TIME_RUNNING;
    int fd = perf_event_open(pe, pid, cpu, -1, 0);
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

    /* inst_retired.any — RAW 0x00C0 */
    RESET_PE();
    pe.type   = PERF_TYPE_RAW;
    pe.config = 0x00C0;
    active += open_counter(idx++, &pe, pid, cpu);

    /* L1-icache-load-misses — HW_CACHE */
    RESET_PE();
    pe.type   = PERF_TYPE_HW_CACHE;
    pe.config = PERF_COUNT_HW_CACHE_L1I
                | (PERF_COUNT_HW_CACHE_OP_READ    << 8)
                | (PERF_COUNT_HW_CACHE_RESULT_MISS << 16);
    active += open_counter(idx++, &pe, pid, cpu);

    /* iTLB-loads — HW_CACHE */
    pe.config = PERF_COUNT_HW_CACHE_ITLB
                | (PERF_COUNT_HW_CACHE_OP_READ      << 8)
                | (PERF_COUNT_HW_CACHE_RESULT_ACCESS << 16);
    active += open_counter(idx++, &pe, pid, cpu);

    /* iTLB-load-misses — HW_CACHE */
    pe.config = PERF_COUNT_HW_CACHE_ITLB
                | (PERF_COUNT_HW_CACHE_OP_READ    << 8)
                | (PERF_COUNT_HW_CACHE_RESULT_MISS << 16);
    active += open_counter(idx++, &pe, pid, cpu);

    /* branch-instructions — HARDWARE */
    RESET_PE();
    pe.type   = PERF_TYPE_HARDWARE;
    pe.config = PERF_COUNT_HW_BRANCH_INSTRUCTIONS;
    active += open_counter(idx++, &pe, pid, cpu);

    /* branch-misses — HARDWARE */
    pe.config = PERF_COUNT_HW_BRANCH_MISSES;
    active += open_counter(idx++, &pe, pid, cpu);

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
        pmu_counters[i].time_enabled = cnt.time_enabled;
        pmu_counters[i].time_running = cnt.time_running;
        if (cnt.time_running > 0 && cnt.time_running < cnt.time_enabled) {
            double scale = (double)cnt.time_enabled / cnt.time_running;
            pmu_counters[i].count = (uint64_t)(cnt.value * scale);
        } else {
            pmu_counters[i].count = cnt.value;
        }
    }
}

void pmu_reset(void)
{
    for (int i = 0; i < PMU_NUM_COUNTERS; i++)
        if (pmu_counters[i].enabled)
            ioctl(pmu_counters[i].fd, PERF_EVENT_IOC_RESET, 0);
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
