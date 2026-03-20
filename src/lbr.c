/*
 * lbr.c — LBR ring buffer 采集实现
 *
 * 使用 PERF_SAMPLE_BRANCH_STACK 从硬件 LBR 寄存器采集跳转记录，
 * 通过 perf mmap ring buffer 以零拷贝方式读取，计算跳转跨度统计。
 */

#define _GNU_SOURCE
#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include <errno.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <linux/perf_event.h>
#include <asm/unistd.h>

#include "lbr.h"

/* ── 配置常量 ────────────────────────────────────────────────────────────── */

#define LBR_MMAP_PAGES  16      /* 数据页数，必须为 2 的幂                 */
#define MAX_LBR_ENTRIES 32      /* 硬件最多 32 条跳转记录                  */

/* ── 模块私有状态 ─────────────────────────────────────────────────────────── */

static int    lbr_fd         = -1;
static struct perf_event_mmap_page *lbr_meta = NULL;
static void  *lbr_data_buf   = NULL;
static size_t lbr_page_size  = 0;
static size_t lbr_data_size  = 0;

/* ── 内部工具 ────────────────────────────────────────────────────────────── */

static int perf_event_open(struct perf_event_attr *hw, pid_t pid,
                           int cpu, int group_fd, unsigned long flags)
{
    return (int)syscall(__NR_perf_event_open, hw, pid, cpu, group_fd, flags);
}

/*
 * ring_read — 从环形缓冲区中安全读取 len 字节，自动处理回绕。
 */
static void ring_read(uint64_t tail, const char *data, size_t mask,
                      void *dst, size_t len)
{
    char *out = (char *)dst;
    for (size_t i = 0; i < len; i++)
        out[i] = data[(tail + i) & mask];
}

/* ── 公共接口 ────────────────────────────────────────────────────────────── */

int lbr_init(pid_t pid)
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
    pe.disabled       = 1;
    pe.exclude_kernel = 1;
    pe.exclude_hv     = 1;

    /* 全系统模式必须绑定至具体 CPU；按进程模式设 cpu=-1 跟随线程 */
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

    ioctl(lbr_fd, PERF_EVENT_IOC_RESET,  0);
    ioctl(lbr_fd, PERF_EVENT_IOC_ENABLE, 0);
    return 1;
}

void lbr_drain(lbr_stats_t *stats)
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
                stats->sample_count++;
                for (uint64_t i = 0; i < nr; i++) {
                    struct perf_branch_entry entry;
                    ring_read(tail + off, data, mask, &entry, sizeof(entry));
                    off += sizeof(entry);

                    /* 跳转跨度 = |from - to|（字节绝对距离）*/
                    uint64_t span = (entry.from > entry.to)
                                    ? (entry.from - entry.to)
                                    : (entry.to   - entry.from);
                    stats->total_span  += span;
                    stats->entry_count++;
                }
            }
        }

        tail += hdr.size;
    }

    __sync_synchronize(); /* mb：确保写 data_tail 在读数据之后 */
    lbr_meta->data_tail = tail;
}

void lbr_stats_reset(lbr_stats_t *stats)
{
    stats->total_span   = 0;
    stats->entry_count  = 0;
    stats->sample_count = 0;
}

void lbr_close(void)
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
