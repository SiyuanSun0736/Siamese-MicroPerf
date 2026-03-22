/*
 * lbr.c — LBR ring buffer 采集实现
 *
 * 使用 PERF_SAMPLE_BRANCH_STACK 从硬件 LBR 寄存器采集跳转记录，
 * 通过 perf mmap ring buffer 以零拷贝方式读取，计算跳转跨度统计。
 *
 * 内部以"槽位数组"管理所有活跃的 perf_event fd：
 *   · 标准模式（lbr_init）：occupy slot[0]，inherit=0（LBR 不支持继承）。
 *   · 无继承模式（lbr_init_no_inherit）：语义同上，名称更明确；
 *     后续通过 lbr_add_tid() / lbr_remove_tid() 动态增删槽位。
 *   · lbr_drain() 统一遍历所有活跃槽位，汇总到调用方的 lbr_stats_t。
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
/*
 * 跨模块跳转过滤阈值（字节）：主程序文本（0x55...）到 JIT stub / vDSO
 * 页（0x7f...）的地址差可达 ~90 TB，会严重拉高 lbr_avg_span 均值。
 * 64 KB 足以覆盖所有合法用户态循环体，同时过滤跨段伪影。
 */
#define LBR_SPAN_MAX    (64ULL * 1024)

/*
 * 无继承模式下可同时跟踪的最大 TID 数量（含根 TID）。
 * 槽位 0 始终预留给根 PID。
 */
#define MAX_LBR_SLOTS   1024

/* ── 每槽位状态 ──────────────────────────────────────────────────────────── */

typedef struct {
    pid_t  tid;          /* 对应的 TID（-1 表示全系统槽）               */
    int    fd;           /* perf_event fd                               */
    struct perf_event_mmap_page *meta;
    void  *data_buf;
} lbr_slot_t;

/* ── 模块私有状态 ─────────────────────────────────────────────────────────── */

static lbr_slot_t lbr_slots[MAX_LBR_SLOTS];
static int        lbr_slot_count  = 0;   /* 当前活跃槽位数量              */
static size_t     lbr_page_size   = 0;
static size_t     lbr_data_size   = 0;

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

/*
 * drain_slot — 排空单个槽位的 ring buffer，累加到 stats。
 */
static void drain_slot(lbr_slot_t *s, lbr_stats_t *stats)
{
    const char *data = (const char *)s->data_buf;
    size_t       mask = lbr_data_size - 1;

    uint64_t head = s->meta->data_head;
    __sync_synchronize(); /* rmb：确保在读取数据前看到最新的 head */
    uint64_t tail = s->meta->data_tail;

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

                    /* 跳转跨度 = |from - to|（字节绝对距离）
                     * 过滤超出 LBR_SPAN_MAX 的跨模块跳转（如主程序→JIT stub
                     * 页或 vDSO 调用），其地址差可达数十 TB，会污染均值。 */
                    uint64_t span = (entry.from > entry.to)
                                    ? (entry.from - entry.to)
                                    : (entry.to   - entry.from);
                    if (span <= LBR_SPAN_MAX) {
                        stats->total_span  += span;
                        stats->entry_count++;
                    }
                }
            }
        }

        tail += hdr.size;
    }

    __sync_synchronize(); /* mb：确保写 data_tail 在读数据之后 */
    s->meta->data_tail = tail;
}

/*
 * open_lbr_slot — 为 pid/tid 申请一个 perf_event 并填入 idx 槽。
 * 内核不允许 PERF_SAMPLE_BRANCH_STACK 与 inherit=1 同时使用（EINVAL），
 * 因此始终以 inherit=0 打开，子线程须由外部监控层手动添加。
 * 成功返回 1，失败返回 0（槽位内容未修改）。
 */
static int open_lbr_slot(int idx, pid_t tid)
{
    struct perf_event_attr pe;
    memset(&pe, 0, sizeof(pe));
    pe.size        = sizeof(pe);
    pe.type        = PERF_TYPE_HARDWARE;
    pe.config      = PERF_COUNT_HW_CPU_CYCLES;
    pe.sample_freq = 1000;
    pe.freq        = 1;
    pe.sample_type = PERF_SAMPLE_IP | PERF_SAMPLE_BRANCH_STACK;
    pe.branch_sample_type =
        PERF_SAMPLE_BRANCH_USER | PERF_SAMPLE_BRANCH_ANY;
    pe.disabled       = 1;
    pe.exclude_kernel = 1;
    pe.exclude_hv     = 1;
    /* inherit 保持 0：LBR 不支持继承，子线程须通过 lbr_add_tid() 手动挂载 */

    /* 全系统模式绑定 CPU 0；按进程 / 线程模式随线程迁移 */
    int cpu = (tid == -1) ? 0 : -1;

    int fd = perf_event_open(&pe, tid, cpu, -1, 0);
    if (fd < 0) {
        fprintf(stderr, "LBR perf_event_open(tid=%d) failed: %s "
                "(需要硬件 LBR 支持，Intel Haswell+ 或 AMD Zen3+)\n",
                (int)tid, strerror(errno));
        return 0;
    }

    void *base = mmap(NULL, lbr_page_size + lbr_data_size,
                      PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if (base == MAP_FAILED) {
        fprintf(stderr, "LBR mmap(tid=%d) failed: %s\n",
                (int)tid, strerror(errno));
        close(fd);
        return 0;
    }

    lbr_slots[idx].tid      = tid;
    lbr_slots[idx].fd       = fd;
    lbr_slots[idx].meta     = (struct perf_event_mmap_page *)base;
    lbr_slots[idx].data_buf = (char *)base + lbr_page_size;

    ioctl(fd, PERF_EVENT_IOC_RESET,  0);
    ioctl(fd, PERF_EVENT_IOC_ENABLE, 0);
    return 1;
}

/*
 * close_slot — 释放单个槽位的所有资源，并将其标记为未使用（fd=-1）。
 */
static void close_slot(lbr_slot_t *s)
{
    if (s->meta) {
        munmap(s->meta, lbr_page_size + lbr_data_size);
        s->meta     = NULL;
        s->data_buf = NULL;
    }
    if (s->fd >= 0) {
        close(s->fd);
        s->fd = -1;
    }
    s->tid = 0;
}

/* ── 公共接口：标准模式 ──────────────────────────────────────────────────── */

int lbr_init(pid_t pid)
{
    lbr_page_size = (size_t)sysconf(_SC_PAGESIZE);
    lbr_data_size = (size_t)LBR_MMAP_PAGES * lbr_page_size;

    memset(lbr_slots, 0, sizeof(lbr_slots));
    for (int i = 0; i < MAX_LBR_SLOTS; i++)
        lbr_slots[i].fd = -1;
    lbr_slot_count = 0;

    if (!open_lbr_slot(0, pid))
        return 0;

    lbr_slot_count = 1;
    return 1;
}

void lbr_drain(lbr_stats_t *stats)
{
    for (int i = 0; i < lbr_slot_count; i++) {
        if (lbr_slots[i].fd >= 0 && lbr_slots[i].meta)
            drain_slot(&lbr_slots[i], stats);
    }
}

void lbr_stats_reset(lbr_stats_t *stats)
{
    stats->total_span   = 0;
    stats->entry_count  = 0;
    stats->sample_count = 0;
}

void lbr_close(void)
{
    for (int i = 0; i < lbr_slot_count; i++)
        close_slot(&lbr_slots[i]);
    lbr_slot_count = 0;
}

/* ── 公共接口：无继承模式 ────────────────────────────────────────────────── */

int lbr_init_no_inherit(pid_t pid)
{
    lbr_page_size = (size_t)sysconf(_SC_PAGESIZE);
    lbr_data_size = (size_t)LBR_MMAP_PAGES * lbr_page_size;

    memset(lbr_slots, 0, sizeof(lbr_slots));
    for (int i = 0; i < MAX_LBR_SLOTS; i++)
        lbr_slots[i].fd = -1;
    lbr_slot_count = 0;

    if (!open_lbr_slot(0, pid))
        return 0;

    lbr_slot_count = 1;
    return 1;
}

int lbr_add_tid(pid_t tid)
{
    if (lbr_slot_count >= MAX_LBR_SLOTS) {
        fprintf(stderr, "lbr_add_tid: slot array full (limit=%d)\n",
                MAX_LBR_SLOTS);
        return 0;
    }

    /* 防止重复添加同一 TID */
    for (int i = 0; i < lbr_slot_count; i++) {
        if (lbr_slots[i].fd >= 0 && lbr_slots[i].tid == tid)
            return 1;   /* 已存在，幂等 */
    }

    int idx = lbr_slot_count;
    if (!open_lbr_slot(idx, tid))
        return 0;

    lbr_slot_count++;
    return 1;
}

void lbr_remove_tid(pid_t tid)
{
    for (int i = 0; i < lbr_slot_count; i++) {
        if (lbr_slots[i].tid == tid && lbr_slots[i].fd >= 0) {
            close_slot(&lbr_slots[i]);
            /* 用最后一个槽位填洞，保持数组紧凑 */
            if (i < lbr_slot_count - 1)
                lbr_slots[i] = lbr_slots[lbr_slot_count - 1];
            lbr_slot_count--;
            return;
        }
    }
}
