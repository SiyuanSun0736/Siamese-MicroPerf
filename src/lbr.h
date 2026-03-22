/*
 * lbr.h — LBR (Last Branch Record) ring buffer 采集接口
 *
 * 通过 perf_event PERF_SAMPLE_BRANCH_STACK + mmap ring buffer 读取
 * 硬件 LBR 寄存器内容，统计每采样周期的跳转跨度分布。
 * 需要 Intel Haswell+ 或 AMD Zen3+ 硬件支持。
 *
 * 工作模式
 * ─────────
 * 注意：Linux 内核不支持 PERF_SAMPLE_BRANCH_STACK 与 inherit=1 同时使用，
 * 因此两种模式均以 inherit=0 打开 perf_event，子线程 / 子进程不会自动
 * 继承 LBR 采集。若需覆盖子线程，须启用外部监控层（tid_monitor）。
 *
 * 标准模式（lbr_init）：
 *   仅为根 PID 开启 LBR 采集（inherit=0）。若目标单线程运行，无需额外
 *   配置；若目标会派生线程，需配合 tid_monitor + lbr_add_tid() 使用。
 *
 * 无继承显式模式（lbr_init_no_inherit）：
 *   语义与 lbr_init 相同，但名称更明确，表示调用方已知晓 inherit 限制，
 *   并计划手动管理每个 TID，配合 tid_monitor 使用。
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

/* ── 标准模式接口 ────────────────────────────────────────────────────────── */

/*
 * lbr_init — 打开 LBR perf event（inherit=0，LBR 不支持继承）并建立
 *   mmap ring buffer。
 *   pid : 目标进程（-1 = 全系统，此时自动绑定 CPU 0）
 * 成功返回 1，硬件不支持或权限不足返回 0。
 * 若目标进程会派生子线程，需配合 tid_monitor + lbr_add_tid() 使用。
 */
int  lbr_init(pid_t pid);

/*
 * lbr_drain — 排空所有活跃槽位的 ring buffer，将本周期 LBR 记录累加到
 * *stats。调用者在每个采样周期结束后负责调用 lbr_stats_reset() 清零。
 */
void lbr_drain(lbr_stats_t *stats);

/*
 * lbr_stats_reset — 将 stats 中的所有字段清零，开始下一个采样周期。
 */
void lbr_stats_reset(lbr_stats_t *stats);

/*
 * lbr_close — 释放所有槽位的 mmap 映射并关闭 perf event fd。
 */
void lbr_close(void);

/* ── 无继承模式接口（配合 tid_monitor 使用）─────────────────────────────── */

/*
 * lbr_init_no_inherit — 与 lbr_init() 语义相同（均使用 inherit=0），
 *   但名称明确表示调用方计划手动管理每个 TID，配合 tid_monitor 使用。
 * 成功返回 1，失败返回 0。
 */
int  lbr_init_no_inherit(pid_t pid);

/*
 * lbr_add_tid — 为指定 TID 分配独立的 LBR perf_event 和 mmap ring buffer。
 *   仅在 no-inherit 模式下有效（lbr_init_no_inherit() 已成功调用后）。
 * 成功返回 1，无可用槽位或打开失败返回 0。
 */
int  lbr_add_tid(pid_t tid);

/*
 * lbr_remove_tid — 释放指定 TID 的 LBR perf_event 资源。
 *   若 TID 不在槽位中则静默忽略。
 */
void lbr_remove_tid(pid_t tid);

#endif /* LBR_H */
