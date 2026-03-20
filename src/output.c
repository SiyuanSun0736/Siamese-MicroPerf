/*
 * output.c — 输出格式化与指标计算实现
 */

#include <stdio.h>
#include <math.h>

#include "output.h"
#include "pmu_counters.h"
#include "lbr.h"

/* ── 指标计算 ────────────────────────────────────────────────────────────── */

void output_compute_lbr(const lbr_stats_t *stats, lbr_metrics_t *out)
{
    out->sample_count = stats->sample_count;
    out->avg_span     = stats->entry_count > 0
                        ? (double)stats->total_span / stats->entry_count
                        : 0.0;
    out->log1p_span   = log(1.0 + out->avg_span);
}

/* ── CSV 输出 ────────────────────────────────────────────────────────────── */

void output_csv_header(FILE *f)
{
    fprintf(f, "elapsed_ms,timestamp");
    for (int i = 0; i < PMU_NUM_COUNTERS; i++)
        fprintf(f, ",%s,%s_time_enabled,%s_time_running",
                pmu_counters[i].name,
                pmu_counters[i].name,
                pmu_counters[i].name);
    fprintf(f, ",lbr_samples,lbr_avg_span,lbr_log1p_span\n");
    fflush(f);
}

void output_csv_row(FILE *f, uint64_t elapsed_ms, const char *wall_str,
                    int lbr_ok, const lbr_metrics_t *lbr)
{
    fprintf(f, "%llu,%s",
            (unsigned long long)elapsed_ms, wall_str);

    for (int i = 0; i < PMU_NUM_COUNTERS; i++) {
        if (pmu_counters[i].enabled)
            fprintf(f, ",%llu,%llu,%llu",
                    (unsigned long long)pmu_counters[i].count,
                    (unsigned long long)pmu_counters[i].time_enabled,
                    (unsigned long long)pmu_counters[i].time_running);
        else
            fprintf(f, ",N/A,N/A,N/A");
    }

    if (lbr_ok && lbr)
        fprintf(f, ",%llu,%.2f,%.6f",
                (unsigned long long)lbr->sample_count,
                lbr->avg_span,
                lbr->log1p_span);
    else
        fprintf(f, ",N/A,N/A,N/A");

    fprintf(f, "\n");
    fflush(f);
}

/* ── 终端输出 ─────────────────────────────────────────────────────────────── */

void output_terminal_row(uint64_t elapsed_ms, const char *wall_str,
                         int lbr_ok, const lbr_metrics_t *lbr)
{
    printf("t=%llums  %s\n",
           (unsigned long long)elapsed_ms, wall_str);

    for (int i = 0; i < PMU_NUM_COUNTERS; i++) {
        if (pmu_counters[i].enabled)
            printf("  %-28s %llu\n",
                   pmu_counters[i].name,
                   (unsigned long long)pmu_counters[i].count);
        else
            printf("  %-28s N/A\n", pmu_counters[i].name);
    }

    if (lbr_ok && lbr)
        printf("  %-28s samples=%-6llu avg_span=%-12.1f log1p=%.6f\n",
               "LBR branch stack",
               (unsigned long long)lbr->sample_count,
               lbr->avg_span,
               lbr->log1p_span);
    else
        printf("  %-28s N/A (hardware not supported)\n", "LBR branch stack");

    printf("\n");
    fflush(stdout);
}
