/*
 * main.c — PMU 时间序列采样器入口
 *
 * 负责：参数解析、日志文件管理、信号处理、timerfd 采样循环。
 * 数据采集由 pmu_counters / lbr 模块完成；
 * 输出格式化（指标计算 + CSV / 终端）由 output 模块完成。
 *
 * 用法：
 *   sudo ./pmu_monitor [PID] [-i <interval_ms>]
 *
 *   PID         – 目标进程（-1 = 全系统，默认）
 *   -i <ms>     – 采样间隔毫秒（默认 1000）
 *
 * 示例：
 *   sudo ./pmu_monitor                   # 全系统，1 s
 *   sudo ./pmu_monitor 12345             # 监控 pid 12345，1 s
 *   sudo ./pmu_monitor 12345 -i 200      # 监控 pid 12345，200 ms
 */

#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <signal.h>
#include <errno.h>
#include <time.h>
#include <sys/stat.h>
#include <sys/timerfd.h>
#include <poll.h>
#include <stdint.h>

#include "pmu_counters.h"
#include "lbr.h"
#include "output.h"

/* ── 全局状态 ────────────────────────────────────────────────────────────── */

static FILE *log_file    = NULL;
static int   timer_fd    = -1;
static int   lbr_enabled = 0;

/* ── 工具函数 ────────────────────────────────────────────────────────────── */

/* 返回 CLOCK_MONOTONIC 毫秒时间戳 */
static uint64_t now_ms(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000ULL + (uint64_t)(ts.tv_nsec / 1000000LL);
}

/* ── 信号处理 ────────────────────────────────────────────────────────────── */

static void cleanup(int sig __attribute__((unused)))
{
    pmu_close();
    if (lbr_enabled) lbr_close();
    if (timer_fd >= 0) close(timer_fd);
    if (log_file) {
        fclose(log_file);
        printf("\nLog file closed.\n");
    }
    printf("Exiting.\n");
    exit(0);
}

/* ── main ────────────────────────────────────────────────────────────────── */

int main(int argc, char **argv)
{
    pid_t target_pid = -1;
    long  interval_ms = 1000;

    /* ---- 解析命令行参数 ---- */
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-i") == 0 && i + 1 < argc) {
            interval_ms = atol(argv[++i]);
            if (interval_ms <= 0) {
                fprintf(stderr, "Invalid interval: %ld\n", interval_ms);
                return 1;
            }
        } else {
            char *end;
            long v = strtol(argv[i], &end, 10);
            if (*end == '\0')
                target_pid = (pid_t)v;
            else {
                fprintf(stderr, "Usage: %s [PID] [-i <interval_ms>]\n", argv[0]);
                return 1;
            }
        }
    }

    signal(SIGINT,  cleanup);
    signal(SIGTERM, cleanup);

    /* ---- 创建日志目录与文件 ---- */
    mkdir("log", 0755);

    time_t     now = time(NULL);
    struct tm *tm  = localtime(&now);
    char ts_filename[256];
    snprintf(ts_filename, sizeof(ts_filename),
             "log/pmu_monitor_%04d%02d%02d_%02d%02d%02d.csv",
             tm->tm_year + 1900, tm->tm_mon + 1, tm->tm_mday,
             tm->tm_hour, tm->tm_min, tm->tm_sec);

    log_file = fopen(ts_filename, "w");
    if (!log_file) {
        fprintf(stderr, "Failed to open log file: %s\n", strerror(errno));
        return 1;
    }

    /* 软链接 log/pmu_monitor.csv → 最新文件 */
    const char *link = "log/pmu_monitor.csv";
    unlink(link);
    char rel_target[256];
    snprintf(rel_target, sizeof(rel_target),
             "pmu_monitor_%04d%02d%02d_%02d%02d%02d.csv",
             tm->tm_year + 1900, tm->tm_mon + 1, tm->tm_mday,
             tm->tm_hour, tm->tm_min, tm->tm_sec);
    symlink(rel_target, link);

    setbuf(stdout, NULL);
    setbuf(stderr, NULL);

    /* ---- 初始化 PMU 计数器 ---- */
    int cpu = (target_pid == -1) ? 0 : -1;
    int active_counters = pmu_init(target_pid, cpu);
    if (active_counters == 0) {
        fprintf(stderr, "Error: could not open any performance counters.\n");
        return 1;
    }

    /* ---- 初始化 LBR ---- */
    lbr_enabled = lbr_init(target_pid);

    /* ---- 写入 CSV 表头 ---- */
    output_csv_header(log_file);

    /* ---- 打印启动信息 ---- */
    printf("=== PMU Monitor ===\n");
    printf("Target PID : %d%s\n", target_pid,
           target_pid == -1 ? " (system-wide)" : "");
    printf("Interval   : %ld ms\n", interval_ms);
    printf("Log file   : %s\n", ts_filename);
    printf("Symlink    : %s\n\n", link);
    printf("Opened %d/%d counters:\n", active_counters, PMU_NUM_COUNTERS);
    for (int i = 0; i < PMU_NUM_COUNTERS; i++)
        printf("  %c %s\n",
               pmu_counters[i].enabled ? '+' : ' ',
               pmu_counters[i].name);
    printf("  %c LBR branch stack (lbr_avg_span, lbr_log1p_span)\n",
           lbr_enabled ? '+' : ' ');
    printf("\nMonitoring… (Ctrl+C to stop)\n\n");

    /* ---- 创建 timerfd（CLOCK_MONOTONIC，无漂移）---- */
    timer_fd = timerfd_create(CLOCK_MONOTONIC, TFD_CLOEXEC);
    if (timer_fd < 0) { perror("timerfd_create"); return 1; }

    struct itimerspec its = {
        .it_interval = { .tv_sec  =  interval_ms / 1000,
                         .tv_nsec = (interval_ms % 1000) * 1000000L },
        .it_value    = { .tv_sec  =  interval_ms / 1000,
                         .tv_nsec = (interval_ms % 1000) * 1000000L },
    };
    if (timerfd_settime(timer_fd, 0, &its, NULL) < 0) {
        perror("timerfd_settime"); return 1;
    }

    uint64_t start_ms = now_ms();

    /* ── 采样循环 ──────────────────────────────────────────────────────── */
    lbr_stats_t lbr_stats = {0, 0, 0};

    while (1) {
        struct pollfd pfd = { .fd = timer_fd, .events = POLLIN };
        if (poll(&pfd, 1, -1) < 0) {
            if (errno == EINTR) continue;
            perror("poll"); break;
        }

        uint64_t expirations = 0;
        read(timer_fd, &expirations, sizeof(expirations));
        if (expirations > 1)
            fprintf(stderr, "Warning: timer overrun — %llu missed tick(s)\n",
                    (unsigned long long)(expirations - 1));

        uint64_t elapsed = now_ms() - start_ms;

        /* 挂钟时间戳（人类可读）*/
        struct timespec wall;
        clock_gettime(CLOCK_REALTIME, &wall);
        char wall_str[32];
        strftime(wall_str, sizeof(wall_str), "%Y-%m-%d %H:%M:%S",
                 localtime(&wall.tv_sec));
        snprintf(wall_str + strlen(wall_str),
                 sizeof(wall_str) - strlen(wall_str),
                 ".%03ld", wall.tv_nsec / 1000000L);

        /* 读取 PMU 计数器（含多路复用缩放）*/
        pmu_read();
        pmu_reset();

        /* 排空 LBR ring buffer，计算本周期派生指标 */
        lbr_drain(&lbr_stats);
        lbr_metrics_t lbr_out;
        output_compute_lbr(&lbr_stats, &lbr_out);
        lbr_stats_reset(&lbr_stats);

        /* 输出至 CSV 与终端 */
        output_csv_row(log_file, elapsed, wall_str, lbr_enabled, &lbr_out);
        output_terminal_row(elapsed, wall_str, lbr_enabled, &lbr_out);
    }

    return 0;
}
