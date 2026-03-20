/*
 * lbr.h — LBR (Last Branch Record) ring buffer 采集接口
 *
 * 通过 perf_event PERF_SAMPLE_BRANCH_STACK + mmap ring buffer 读取
 * 硬件 LBR 寄存器内容，统计每采样周期的跳转跨度分布。
 * 需要 Intel Haswell+ 或 AMD Zen3+ 硬件支持。
 */

#ifndef LBR_H
#define LBR_H

#include <stdint.h>
#include <sys/types.h>

/* ── 采样周期统计结果 ────────────────────────────────────────────────────── */

typedef struct {
    uint64_t total_span;    /* 所有 LBR 跳转跨度之和（字节）              */
    uint64_t entry_count;   /* LBR 跳转条目总数                           */
    uint64_t sample_count;  /* 触发的采样次数（PERF_RECORD_SAMPLE 数）    */
} lbr_stats_t;

/* ── 接口函数 ─────────────────────────────────────────────────────────────── */

/*
 * lbr_init — 打开 LBR perf event 并建立 mmap ring buffer。
 *   pid : 目标进程（-1 = 全系统，此时自动绑定 CPU 0）
 * 成功返回 1，硬件不支持或权限不足返回 0。
 */
int  lbr_init(pid_t pid);

/*
 * lbr_drain — 排空 ring buffer，将本周期的 LBR 记录累加到 *stats。
 * 调用者在每个采样周期结束后负责调用 lbr_stats_reset() 清零。
 */
void lbr_drain(lbr_stats_t *stats);

/*
 * lbr_stats_reset — 将 stats 中的所有字段清零，开始下一个采样周期。
 */
void lbr_stats_reset(lbr_stats_t *stats);

/*
 * lbr_close — 释放 mmap 映射并关闭 perf event fd。
 */
void lbr_close(void);

#endif /* LBR_H */
