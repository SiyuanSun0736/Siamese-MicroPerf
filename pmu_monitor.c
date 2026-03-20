/*
 * pmu_monitor.c — PMU 时间序列采样器（精简版）
 *
 * 以固定间隔采集下列硬件性能计数器并输出 CSV 日志：
 *   inst_retired.any, L1-icache-load-misses,
 *   iTLB-loads, iTLB-load-misses,
 *   branch-instructions, branch-misses
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
#include <sys/ioctl.h>
#include <linux/perf_event.h>
#include <asm/unistd.h>
#include <signal.h>
#include <stdint.h>
#include <errno.h>
#include <time.h>
#include <sys/stat.h>
#include <sys/timerfd.h>
#include <poll.h>
#include <sys/mman.h>
#include <math.h>

/* ── 计数器描述符 ─────────────────────────────────────────────────────────── */

typedef struct {
    int         fd;
    const char *name;
    uint64_t    count;         /* 本采样周期的计数值                        */
    int         enabled;       /* fd 成功打开则为 1                         */
    uint64_t    time_enabled;  /* 上次读取的 time_enabled                   */
    uint64_t    time_running;  /* 上次读取的 time_running                   */
} perf_counter_t;

static perf_counter_t counters[] = {
    {-1, "inst_retired.any",        0, 0, 0, 0},
    {-1, "L1-icache-load-misses",   0, 0, 0, 0},
    {-1, "iTLB-loads",              0, 0, 0, 0},
    {-1, "iTLB-load-misses",        0, 0, 0, 0},
    {-1, "branch-instructions",     0, 0, 0, 0},
    {-1, "branch-misses",           0, 0, 0, 0},
};
#define NUM_COUNTERS (sizeof(counters) / sizeof(counters[0]))

/* ── 全局变量 ────────────────────────────────────────────────────────────── */

static FILE *log_file    = NULL;
static int   timer_fd    = -1;
static long  interval_ms = 1000;
static int   perf_cpu    = -1;  /* -1=跟随线程，>=0=指定 CPU              */

/* ── LBR 全局变量 ────────────────────────────────────────────────────────── */

#define LBR_MMAP_PAGES  16      /* 数据页数，必须为 2 的幂                 */
#define MAX_LBR_ENTRIES 32      /* 硬件最多 32 条跳转记录                  */

static int    lbr_fd         = -1;
static struct perf_event_mmap_page *lbr_meta = NULL;
static void  *lbr_data_buf   = NULL;
static size_t lbr_page_size  = 0;
static size_t lbr_data_size  = 0;

/* 当前采样周期的 LBR 累计统计 */
static uint64_t lbr_total_span   = 0;  /* 所有 LBR 跳转跨度之和            */
static uint64_t lbr_entry_count  = 0;  /* LBR 跳转条目总数                 */
static uint64_t lbr_sample_count = 0;  /* 触发的采样次数                   */

/* ── 工具函数 ────────────────────────────────────────────────────────────── */

static int perf_event_open(struct perf_event_attr *hw, pid_t pid,
                           int cpu, int group_fd, unsigned long flags)
{
    return (int)syscall(__NR_perf_event_open, hw, pid, cpu, group_fd, flags);
}

/* 独立打开（不分组），让内核通过时间分片复用所有事件；
 * 每个 fd 携带独立的 time_enabled/time_running 用于缩放。 */
static int init_counter(int idx, struct perf_event_attr *pe, pid_t pid)
{
    pe->read_format = PERF_FORMAT_TOTAL_TIME_ENABLED |
                      PERF_FORMAT_TOTAL_TIME_RUNNING;
    int fd = perf_event_open(pe, pid, perf_cpu, -1, 0);
    if (fd < 0) return 0;
    counters[idx].fd      = fd;
    counters[idx].enabled = 1;
    ioctl(fd, PERF_EVENT_IOC_ENABLE, 0);
    return 1;
}

/* ── LBR 函数 ─────────────────────────────────────────────────────────────── */

/*
 * 从 ring buffer 中安全读取 len 字节，正确处理环形回绕。
 */
static void ring_read(uint64_t tail, const char *data, size_t mask,
                      void *dst, size_t len)
{
    char *out = (char *)dst;
    for (size_t i = 0; i < len; i++)
        out[i] = data[(tail + i) & mask];
}

/*
 * init_lbr — 打开 LBR perf event 并建立 mmap ring buffer。
 *
 * 使用 PERF_SAMPLE_BRANCH_STACK 让内核在每次采样时将硬件 LBR
 * 寄存器内容写入 ring buffer；每秒约 1000 次采样提供充足的统计量。
 */
static int init_lbr(pid_t pid)
{
    lbr_page_size = (size_t)sysconf(_SC_PAGESIZE);
    lbr_data_size = (size_t)LBR_MMAP_PAGES * lbr_page_size;

    struct perf_event_attr pe;
    memset(&pe, 0, sizeof(pe));
    pe.size        = sizeof(pe);
    pe.type        = PERF_TYPE_HARDWARE;
    pe.config      = PERF_COUNT_HW_CPU_CYCLES;
    pe.sample_freq = 1000;              /* 约 1000 次采样 / 秒              */
    pe.freq        = 1;                 /* 使用频率模式而非固定周期         */
    pe.sample_type = PERF_SAMPLE_IP | PERF_SAMPLE_BRANCH_STACK;
    pe.branch_sample_type =
        PERF_SAMPLE_BRANCH_USER | PERF_SAMPLE_BRANCH_ANY;
    pe.disabled      = 1;
    pe.exclude_kernel = 1;
    pe.exclude_hv    = 1;

    int cpu_lbr = (pid == -1) ? 0 : -1;
    lbr_fd = perf_event_open(&pe, pid, cpu_lbr, -1, 0);
    if (lbr_fd < 0) {
        fprintf(stderr, "LBR perf_event_open failed: %s "
                "(需要硬件 LBR 支持，Intel Haswell+ 或 AMD Zen3+)\n",
                strerror(errno));
        return 0;
    }

    /* 映射 1 个 meta 页 + LBR_MMAP_PAGES 个数据页 */
    void *base = mmap(NULL, lbr_page_size + lbr_data_size,
                      PROT_READ | PROT_WRITE, MAP_SHARED, lbr_fd, 0);
    if (base == MAP_FAILED) {
        fprintf(stderr, "LBR mmap failed: %s\n", strerror(errno));
        close(lbr_fd);
        lbr_fd = -1;
        return 0;
    }

    lbr_meta     = (struct perf_event_mmap_page *)base;
    lbr_data_buf = (char *)base + lbr_page_size;

    ioctl(lbr_fd, PERF_EVENT_IOC_RESET, 0);
    ioctl(lbr_fd, PERF_EVENT_IOC_ENABLE, 0);
    return 1;
}

/*
 * drain_lbr — 排空 ring buffer，将 LBR 跳转数据累加到当前周期统计中。
 *
 * 每条 PERF_RECORD_SAMPLE 包含：
 *   [u64 ip] [u64 nr] [struct perf_branch_entry × nr]
 * 跳转跨度 = |from - to|，用于计算均值 lbr_avg_span。
 */
static void drain_lbr(void)
{
    if (lbr_fd < 0 || !lbr_meta) return;

    const char *data = (const char *)lbr_data_buf;
    size_t       mask = lbr_data_size - 1;

    uint64_t head = lbr_meta->data_head;
    __sync_synchronize(); /* rmb：确保在读取数据前看到最新的 head */
    uint64_t tail = lbr_meta->data_tail;

    while ((int64_t)(head - tail) > 0) {
        struct perf_event_header hdr;
        ring_read(tail, data, mask, &hdr, sizeof(hdr));

        /* 防止 hdr.size=0 导致死循环 */
        if (hdr.size < sizeof(hdr)) {
            tail = head;
            break;
        }

        if (hdr.type == PERF_RECORD_SAMPLE) {
            size_t off = sizeof(hdr);

            /* PERF_SAMPLE_IP: u64 ip */
            uint64_t ip;
            ring_read(tail + off, data, mask, &ip, sizeof(ip));
            off += sizeof(ip);

            /* PERF_SAMPLE_BRANCH_STACK: u64 nr + entries */
            uint64_t nr;
            ring_read(tail + off, data, mask, &nr, sizeof(nr));
            off += sizeof(nr);

            if (nr > 0 && nr <= MAX_LBR_ENTRIES) {
                lbr_sample_count++;
                for (uint64_t i = 0; i < nr; i++) {
                    struct perf_branch_entry entry;
                    ring_read(tail + off, data, mask,
                              &entry, sizeof(entry));
                    off += sizeof(entry);

                    /* 跳转跨度 = |from - to|（字节绝对距离）*/
                    uint64_t span = (entry.from > entry.to)
                                    ? (entry.from - entry.to)
                                    : (entry.to   - entry.from);
                    lbr_total_span  += span;
                    lbr_entry_count++;
                }
            }
        }

        tail += hdr.size;
    }

    __sync_synchronize(); /* mb：确保 data_tail 的写操作在数据读取后生效 */
    lbr_meta->data_tail = tail;
}

/*
 * close_lbr — 释放 LBR mmap 与 fd 资源。
 */
static void close_lbr(void)
{
    if (lbr_meta) {
        munmap(lbr_meta, lbr_page_size + lbr_data_size);
        lbr_meta     = NULL;
        lbr_data_buf = NULL;
    }
    if (lbr_fd >= 0) {
        close(lbr_fd);
        lbr_fd = -1;
    }
}

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
    for (size_t i = 0; i < NUM_COUNTERS; i++)
        if (counters[i].fd >= 0)
            close(counters[i].fd);
    close_lbr();
    if (timer_fd >= 0)
        close(timer_fd);
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
    pid_t target_pid      = -1;
    int   active_counters = 0;
    int   idx             = 0;

    /* ---- 解析参数 ---- */
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

    /* ---- 创建日志文件 ---- */
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

    /* ---- CSV 表头 ---- */
    fprintf(log_file, "elapsed_ms,timestamp");
    for (size_t i = 0; i < NUM_COUNTERS; i++)
        fprintf(log_file, ",%s,%s_time_enabled,%s_time_running",
                counters[i].name, counters[i].name, counters[i].name);
    fprintf(log_file, ",lbr_samples,lbr_avg_span,lbr_log1p_span\n");
    fflush(log_file);

    /* 全系统模式须指定具体 CPU；按进程模式设 cpu=-1 跟随线程 */
    perf_cpu = (target_pid == -1) ? 0 : -1;

    /* ---- 初始化 LBR perf event ---- */
    int lbr_enabled = init_lbr(target_pid);

    /* ---- 初始化 perf 计数器 ---- */
    struct perf_event_attr pe;
#define RESET_PE() do { memset(&pe, 0, sizeof(pe)); \
                        pe.size = sizeof(pe); pe.disabled = 1; } while (0)

    /* inst_retired.any — RAW 0x00C0 */
    RESET_PE();
    pe.type   = PERF_TYPE_RAW;
    pe.config = 0x00C0;
    active_counters += init_counter(idx++, &pe, target_pid);

    /* L1-icache-load-misses — HW_CACHE */
    RESET_PE();
    pe.type   = PERF_TYPE_HW_CACHE;
    pe.config = PERF_COUNT_HW_CACHE_L1I
                | (PERF_COUNT_HW_CACHE_OP_READ  << 8)
                | (PERF_COUNT_HW_CACHE_RESULT_MISS << 16);
    active_counters += init_counter(idx++, &pe, target_pid);

    /* iTLB-loads — HW_CACHE */
    pe.config = PERF_COUNT_HW_CACHE_ITLB
                | (PERF_COUNT_HW_CACHE_OP_READ  << 8)
                | (PERF_COUNT_HW_CACHE_RESULT_ACCESS << 16);
    active_counters += init_counter(idx++, &pe, target_pid);

    /* iTLB-load-misses — HW_CACHE */
    pe.config = PERF_COUNT_HW_CACHE_ITLB
                | (PERF_COUNT_HW_CACHE_OP_READ  << 8)
                | (PERF_COUNT_HW_CACHE_RESULT_MISS << 16);
    active_counters += init_counter(idx++, &pe, target_pid);

    /* branch-instructions — HARDWARE */
    RESET_PE();
    pe.type   = PERF_TYPE_HARDWARE;
    pe.config = PERF_COUNT_HW_BRANCH_INSTRUCTIONS;
    active_counters += init_counter(idx++, &pe, target_pid);

    /* branch-misses — HARDWARE */
    pe.config = PERF_COUNT_HW_BRANCH_MISSES;
    active_counters += init_counter(idx++, &pe, target_pid);

    if (active_counters == 0) {
        fprintf(stderr, "Error: could not open any performance counters.\n");
        return 1;
    }

    printf("=== PMU Monitor ===\n");
    printf("Target PID : %d%s\n", target_pid,
           target_pid == -1 ? " (system-wide)" : "");
    printf("Interval   : %ld ms\n", interval_ms);
    printf("Log file   : %s\n", ts_filename);
    printf("Symlink    : %s\n\n", link);
    printf("Opened %d/%zu counters:\n", active_counters, NUM_COUNTERS);
    for (size_t i = 0; i < NUM_COUNTERS; i++)
        printf("  %c %s\n", counters[i].enabled ? '+' : ' ', counters[i].name);
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

        /* 读取计数器，各自独立缩放 */
        struct {
            uint64_t value;
            uint64_t time_enabled;
            uint64_t time_running;
        } cnt;

        for (size_t i = 0; i < NUM_COUNTERS; i++) {
            if (!counters[i].enabled) {
                counters[i].count        = 0;
                counters[i].time_enabled = 0;
                counters[i].time_running = 0;
                continue;
            }
            if (read(counters[i].fd, &cnt, sizeof(cnt)) != sizeof(cnt)) {
                fprintf(stderr, "read counter[%zu] failed: %s\n",
                        i, strerror(errno));
                counters[i].count = 0;
                continue;
            }
            counters[i].time_enabled = cnt.time_enabled;
            counters[i].time_running = cnt.time_running;
            if (cnt.time_running > 0 && cnt.time_running < cnt.time_enabled) {
                double scale = (double)cnt.time_enabled / cnt.time_running;
                counters[i].count = (uint64_t)(cnt.value * scale);
            } else {
                counters[i].count = cnt.value;
            }
        }

        /* 重置所有计数器（为下一周期归零）*/
        for (size_t i = 0; i < NUM_COUNTERS; i++)
            if (counters[i].enabled)
                ioctl(counters[i].fd, PERF_EVENT_IOC_RESET, 0);

        /* ---- 排空 LBR ring buffer，计算本周期统计量 ---- */
        drain_lbr();
        double lbr_avg_span   = lbr_entry_count > 0
                                 ? (double)lbr_total_span / lbr_entry_count
                                 : 0.0;
        double lbr_log1p_span = log(1.0 + lbr_avg_span);
        uint64_t cur_lbr_samples = lbr_sample_count;
        /* 重置 LBR 统计，为下一周期归零 */
        lbr_total_span = lbr_entry_count = lbr_sample_count = 0;

        /* ---- 写入 CSV 行 ---- */
        fprintf(log_file, "%llu,%s",
                (unsigned long long)elapsed, wall_str);
        for (size_t i = 0; i < NUM_COUNTERS; i++) {
            if (counters[i].enabled)
                fprintf(log_file, ",%llu,%llu,%llu",
                        (unsigned long long)counters[i].count,
                        (unsigned long long)counters[i].time_enabled,
                        (unsigned long long)counters[i].time_running);
            else
                fprintf(log_file, ",N/A,N/A,N/A");
        }
        if (lbr_enabled)
            fprintf(log_file, ",%llu,%.2f,%.6f",
                    (unsigned long long)cur_lbr_samples,
                    lbr_avg_span, lbr_log1p_span);
        else
            fprintf(log_file, ",N/A,N/A,N/A");
        fprintf(log_file, "\n");
        fflush(log_file);

        /* ---- 终端摘要 ---- */
        printf("t=%llums  %s\n",
               (unsigned long long)elapsed, wall_str);
        for (size_t i = 0; i < NUM_COUNTERS; i++) {
            if (counters[i].enabled)
                printf("  %-28s %llu\n",
                       counters[i].name,
                       (unsigned long long)counters[i].count);
            else
                printf("  %-28s N/A\n", counters[i].name);
        }
        if (lbr_enabled)
            printf("  %-28s samples=%-6llu avg_span=%-12.1f log1p=%.6f\n",
                   "LBR branch stack",
                   (unsigned long long)cur_lbr_samples,
                   lbr_avg_span, lbr_log1p_span);
        else
            printf("  %-28s N/A (hardware not supported)\n", "LBR branch stack");
        printf("\n");
        fflush(stdout);
    }

    return 0;
}
