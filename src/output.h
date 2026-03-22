/*
 * output.h — 输出格式化与指标计算接口
 *
 * 负责将采集到的原始 PMU / LBR 数据转换为派生指标，
 * 并以 CSV 和终端两种形式输出。
 */

#ifndef OUTPUT_H
#define OUTPUT_H

#include <stdio.h>
#include <stdint.h>

#include "pmu_counters.h"
#include "lbr.h"

/* ── LBR 派生指标 ────────────────────────────────────────────────────────── */

typedef struct {
    uint64_t sample_count;  /* 本周期触发的采样次数                       */
    double   avg_span;      /* 平均跳转跨度（字节）                       */
    double   log1p_span;    /* log(1 + avg_span)，用于归一化              */
} lbr_metrics_t;

/* ── 接口函数 ─────────────────────────────────────────────────────────────── */

/*
 * output_compute_lbr — 由原始统计量计算派生指标。
 */
void output_compute_lbr(const lbr_stats_t *stats, lbr_metrics_t *out);

/*
 * output_set_print_time_fields — 控制 CSV 中是否输出每个计数器的
 *   <name>_time_enabled 与 <name>_time_running 列。
 *   enable=1 输出，enable=0（默认）不输出。
 */
void output_set_print_time_fields(int enable);

/*
 * output_csv_header — 向文件写入 CSV 表头（含换行）。
 */
void output_csv_header(FILE *f);

/*
 * output_csv_row — 向文件写入一行 CSV 数据。
 *   elapsed_ms : 自启动以来经过的毫秒数
 *   wall_str   : 挂钟时间字符串（"YYYY-MM-DD HH:MM:SS.mmm"）
 *   lbr_ok     : LBR 是否启用
 *   lbr        : LBR 派生指标（lbr_ok=0 时可为 NULL）
 */
void output_csv_row(FILE *f, uint64_t elapsed_ms, const char *wall_str,
                    int lbr_ok, const lbr_metrics_t *lbr);

/*
 * output_terminal_row — 在终端打印本采样周期的摘要。
 */
void output_terminal_row(uint64_t elapsed_ms, const char *wall_str,
                         int lbr_ok, const lbr_metrics_t *lbr);

#endif /* OUTPUT_H */
