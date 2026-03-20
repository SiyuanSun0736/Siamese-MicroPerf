/*
 * pmu_counters.h — perf_event 计数器管理接口
 *
 * 负责打开、读取、重置和关闭 PMU 硬件计数器；
 * 读取时自动按 time_enabled / time_running 缩放以修正多路复用误差。
 */

#ifndef PMU_COUNTERS_H
#define PMU_COUNTERS_H

#include <stdint.h>
#include <stddef.h>
#include <sys/types.h>

/* ── 计数器描述符 ─────────────────────────────────────────────────────────── */

typedef struct {
    int         fd;
    const char *name;
    uint64_t    count;         /* 本采样周期缩放后的计数值                  */
    int         enabled;       /* fd 成功打开则为 1                         */
    uint64_t    time_enabled;  /* 上次读取的 time_enabled（纳秒）           */
    uint64_t    time_running;  /* 上次读取的 time_running（纳秒）           */
} perf_counter_t;

#define PMU_NUM_COUNTERS 6

extern perf_counter_t pmu_counters[PMU_NUM_COUNTERS];

/* ── 接口函数 ─────────────────────────────────────────────────────────────── */

/*
 * pmu_init — 初始化所有 PMU 计数器。
 *   pid : 目标进程（-1 = 全系统）
 *   cpu : perf_event cpu 参数（-1 跟随线程，>=0 指定 CPU）
 * 返回成功打开的计数器数量。
 */
int  pmu_init(pid_t pid, int cpu);

/*
 * pmu_read — 读取所有计数器并按多路复用比例缩放，写入 count 字段。
 */
void pmu_read(void);

/*
 * pmu_reset — 向内核发送 PERF_EVENT_IOC_RESET，将计数器归零。
 */
void pmu_reset(void);

/*
 * pmu_close — 关闭所有计数器文件描述符。
 */
void pmu_close(void);

#endif /* PMU_COUNTERS_H */
