/*
 * test_workload2.c — 多阶段综合 PMU 压力测试（第二版）
 *
 * 相比 test_pmu_workload.c 的改进：
 *   ① 8 个专项阶段，全面覆盖所有 PMU 计数器组（含 l1d_pend_miss / iTLB）
 *   ② 用 clock_gettime(CLOCK_MONOTONIC) 精确控制阶段时长，各阶段均等
 *   ③ 每阶段打印 ISO 毫秒级时间戳，便于与 PMU 采样点对齐分析
 *   ④ 函数指针表驱动，方便增删阶段
 *
 * 阶段规划（每阶段 duration/5 秒，循环直到总超时）：
 *
 *  P1 L1D-HOT    16KB 顺序读写（4路展开）   → L1-dcache-loads/stores, mem_inst_retired.*
 *  P2 L2-THRASH  256KB 随机读写             → L1-dcache-load-misses, l1d.replacement
 *  P3 LLC-HIT    8MB 随机读写               → L1D/L2 miss + LLC hit, cache-references
 *  P4 LLC-MISS   256MB 随机读写             → LLC-load-misses, cache-misses, dTLB-load-misses
 *  P5 DTLB-STORE 256MB 按页随机写           → dTLB-store-misses 饱和
 *  P6 PEND-MISS  8MB 4路独立并发随机读      → l1d_pend_miss.pending, l1d_pend_miss.fb_full
 *  P7 PTR-CHASE  8MB 指针追逐链（串行化）   → LLC-load-misses（串行），内存访问延迟
 *  P8 ITLB-MISS  512个代码页随机调用        → iTLB-load-misses, L1-icache-load-misses
 *
 * P6 vs P4/P7 的区别：
 *   P4 单流随机读   → CPU 串行等待，miss 不叠加，pend 计数低
 *   P7 指针追逐链   → 完全串行，pend=1，测延迟而非带宽
 *   P6 四流独立随机 → CPU 同时发出 4 个 miss 请求，Fill Buffer 持续满载，
 *                      pend_miss.pending/fb_full 计数最高
 *
 * 用法：
 *   ./test_workload2 [duration_sec]
 *   默认 64 秒（每阶段 5 秒）。
 */

#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>
#include <signal.h>
#include <unistd.h>
#include <sys/mman.h>

/* ── 工作集大小 ─────────────────────────────────────────────────────── */
#define L1D_BYTES       (16UL  * 1024)           /* 16 KB  — L1D 热     */
#define L2_BYTES        (256UL * 1024)           /* 256 KB — L1D Miss   */
#define LLC_HIT_BYTES   (8UL   * 1024 * 1024)   /* 8 MB   — LLC Hit    */
#define DRAM_BYTES      (256UL * 1024 * 1024)   /* 256 MB — DRAM/dTLB  */
#define PAGE_BYTES       4096UL
#define NUM_CODE_PAGES   512     /* 512×4KB 代码页，远超 iTLB/STLB 容量 */

/* ── 每次函数调用内的内层迭代次数 ──────────────────────────────────── */
#define ITER_HOT        4000000  /* P1: L1D hot（每次约 8 ms @ 2GHz）   */
#define ITER_L2          500000  /* P2: L2 thrash                       */
#define ITER_LLC         200000  /* P3: LLC hit                         */
#define ITER_DRAM        100000  /* P4: LLC miss / DRAM                 */
#define ITER_DTLB_ST     500000  /* P5: dTLB store miss                 */
#define ITER_PEND        200000  /* P6: 4路并发 pend_miss               */
#define ITER_CHASE       200000  /* P7: 指针追逐                        */
#define ITER_ITLB       4000000  /* P8: iTLB 随机跳转                   */

static int phase_sec = 5;   /* 每阶段持续秒数，默认 5 秒 */
/* ── 全局 ── */
static volatile uint64_t g_sink    = 0;
static volatile int      g_running = 1;
static void handle_sig(int s __attribute__((unused))) { g_running = 0; }

static inline uint64_t xorshift64(uint64_t *s)
{
    *s ^= *s << 13; *s ^= *s >> 7; *s ^= *s << 17; return *s;
}

/* ── JIT 函数类型 ── */
typedef int (*jit_func_t)(int);

/* ── 上下文结构体 ── */
typedef struct {
    uint64_t   *l1d_buf;     /* L1D_BYTES / 8 个元素（2 的幂）         */
    uint64_t   *l2_buf;      /* L2_BYTES  / 8 个元素（2 的幂）         */
    uint64_t   *llc_buf;     /* LLC_HIT_BYTES / 8 个元素（2 的幂）     */
    uint8_t    *dram_buf;    /* DRAM_BYTES 字节                        */
    uintptr_t  *chase;       /* LLC_HIT_BYTES 大小的指针追逐链         */
    size_t      chase_n;
    jit_func_t *funcs;       /* NUM_CODE_PAGES 个 JIT 函数指针         */
    int        *call_order;  /* 打乱后的调用顺序索引                   */
    int         n_funcs;     /* 实际生成成功的代码页数量               */
} ctx_t;

/* ── 工具：获取单调时间（秒，双精度）── */
static inline double now_sec(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

/* ── 工具：打印挂钟时间戳（毫秒精度）── */
static void print_ts(const char *tag)
{
    struct timespec ts;
    clock_gettime(CLOCK_REALTIME, &ts);
    struct tm tm_info;
    localtime_r(&ts.tv_sec, &tm_info);
    char buf[32];
    strftime(buf, sizeof(buf), "%Y-%m-%d %H:%M:%S", &tm_info);
    printf("[%s.%03ld] %s\n", buf, ts.tv_nsec / 1000000L, tag);
    fflush(stdout);
}

/* ════════════════════════════════════════════════════════════════════
 * P1: L1D-HOT — 16KB 顺序读写，工作集完全在 L1D 内
 *
 * 设计：4路展开 read-modify-write，利用超标量乱序执行
 *       最大化每周期退休的内存指令数（mem_inst_retired.*）
 *
 * 目标计数器：
 *   L1-dcache-loads / L1-dcache-stores / mem_inst_retired.all_{loads,stores}
 * ════════════════════════════════════════════════════════════════════ */
__attribute__((noinline, optimize("O2")))
static void phase_l1d_hot(ctx_t *c)
{
    const size_t n = L1D_BYTES / sizeof(uint64_t);   /* 2048，2 的幂 */
    uint64_t *b = c->l1d_buf;
    uint64_t s = g_sink;
    for (int i = 0; i < ITER_HOT; i += 4) {
        size_t k = (size_t)i & (n - 1);
        s ^= b[k];               b[k]               = s + (uint64_t)i;
        s ^= b[(k+1) & (n-1)];   b[(k+1) & (n-1)]   = s + (uint64_t)(i+1);
        s ^= b[(k+2) & (n-1)];   b[(k+2) & (n-1)]   = s + (uint64_t)(i+2);
        s ^= b[(k+3) & (n-1)];   b[(k+3) & (n-1)]   = s + (uint64_t)(i+3);
    }
    g_sink = s;
}

/* ════════════════════════════════════════════════════════════════════
 * P2: L2-THRASH — 随机读写 256KB，超出 L1D（32KB）但在 L2（通常 256KB-1MB）内
 *
 * 每次随机访问 → L1D Miss → L2 Hit → L1D 被置换（l1d.replacement++）
 *
 * 目标计数器：
 *   L1-dcache-load-misses / l1d.replacement
 * ════════════════════════════════════════════════════════════════════ */
__attribute__((noinline))
static void phase_l2_thrash(ctx_t *c)
{
    const size_t n = L2_BYTES / sizeof(uint64_t);   /* 32768，2 的幂 */
    uint64_t *b = c->l2_buf;
    uint64_t rng = 0xabcdef1234567890ULL ^ g_sink;
    uint64_t s = 0;
    for (int i = 0; i < ITER_L2; i++) {
        xorshift64(&rng);
        size_t k = rng & (n - 1);
        s    += b[k];
        b[k] ^= s;
    }
    g_sink ^= s;
}

/* ════════════════════════════════════════════════════════════════════
 * P3: LLC-HIT — 随机读写 8MB，超出 L1D/L2 但在 LLC（通常 ≥ 8MB）内
 *
 * 目标计数器：
 *   LLC-loads / LLC-stores / cache-references
 * ════════════════════════════════════════════════════════════════════ */
__attribute__((noinline))
static void phase_llc_hit(ctx_t *c)
{
    const size_t n = LLC_HIT_BYTES / sizeof(uint64_t);  /* 1048576，2 的幂 */
    uint64_t *b = c->llc_buf;
    uint64_t rng = 0xdeadbeefcafe1234ULL ^ g_sink;
    uint64_t s = 0;
    for (int i = 0; i < ITER_LLC; i++) {
        xorshift64(&rng);
        size_t k = rng & (n - 1);
        s    += b[k];
        b[k] ^= rng;
    }
    g_sink ^= s;
}

/* ════════════════════════════════════════════════════════════════════
 * P4: LLC-MISS — 随机读写 256MB，远超 LLC 容量 → 访问 DRAM
 *
 * 同时触发大量 dTLB Load Miss（256MB 跨度，随机页）
 *
 * 目标计数器：
 *   LLC-load-misses / cache-misses / dTLB-load-misses
 * ════════════════════════════════════════════════════════════════════ */
__attribute__((noinline))
static void phase_llc_miss(ctx_t *c)
{
    const size_t n = DRAM_BYTES / sizeof(uint64_t);   /* 33554432，2 的幂 */
    uint64_t *b = (uint64_t *)(void *)c->dram_buf;
    uint64_t rng = 0x8765432112345678ULL ^ g_sink;
    uint64_t s = 0;
    for (int i = 0; i < ITER_DRAM; i++) {
        xorshift64(&rng);
        size_t k = rng & (n - 1);
        s    += b[k];
        b[k] ^= rng;
    }
    g_sink ^= s;
}

/* ════════════════════════════════════════════════════════════════════
 * P5: DTLB-STORE — 以 PAGE_BYTES 为步长随机写 256MB
 *
 * 每次写操作指向不同的 4KB 物理页（65536 页，远超 STLB 容量）
 * → dTLB Store Miss 接近 100%
 *
 * 目标计数器：
 *   dTLB-stores / dTLB-store-misses
 * ════════════════════════════════════════════════════════════════════ */
__attribute__((noinline))
static void phase_dtlb_store(ctx_t *c)
{
    const size_t n_pages = DRAM_BYTES / PAGE_BYTES;   /* 65536 页 */
    uint8_t *b = c->dram_buf;
    uint64_t rng = 0xfeedcafe87654321ULL ^ g_sink;
    for (int i = 0; i < ITER_DTLB_ST; i++) {
        xorshift64(&rng);
        b[(rng % n_pages) * PAGE_BYTES] = (uint8_t)(rng & 0xff);
    }
    g_sink ^= b[0];
}

/* ════════════════════════════════════════════════════════════════════
 * P6: PEND-MISS — 4 路完全独立的随机流同时读 8MB（非串行化）
 *
 * 关键原理：
 *   单流（P4/P7）：CPU 等前一个 miss 返回才能知道下一地址 → 串行，Fill Buffer 同时只有 1 个 miss。
 *   四流独立（P6）：四条 xorshift 状态互不依赖 → CPU 乱序引擎可同时发出 4 个 miss 请求
 *                   → L1D Miss Fill Buffer 持续满载 → l1d_pend_miss.fb_full / .pending 最高
 *
 * 目标计数器：
 *   l1d_pend_miss.pending / l1d_pend_miss.fb_full / l1d.replacement
 * ════════════════════════════════════════════════════════════════════ */
__attribute__((noinline, optimize("O2")))
static void phase_pend_miss(ctx_t *c)
{
    const size_t n = LLC_HIT_BYTES / sizeof(uint64_t);   /* 1048576，2 的幂 */
    const uint64_t *b = c->llc_buf;
    uint64_t r0 = 0xa1b2c3d4e5f60708ULL ^ g_sink;
    uint64_t r1 = 0x1234567890abcdefULL;
    uint64_t r2 = 0xfedcba9876543210ULL;
    uint64_t r3 = 0x0807060504030201ULL;
    uint64_t s0 = 0, s1 = 0, s2 = 0, s3 = 0;
    for (int i = 0; i < ITER_PEND; i++) {
        /* 四条流完全独立，可同时在乱序窗口中挂起 */
        xorshift64(&r0); s0 += b[r0 & (n - 1)];
        xorshift64(&r1); s1 += b[r1 & (n - 1)];
        xorshift64(&r2); s2 += b[r2 & (n - 1)];
        xorshift64(&r3); s3 += b[r3 & (n - 1)];
    }
    g_sink ^= s0 ^ s1 ^ s2 ^ s3;
}

/* ════════════════════════════════════════════════════════════════════
 * P7: PTR-CHASE — 8MB 随机指针追逐链（完全串行化 LLC Miss）
 *
 * 每次访问的地址依赖上一次访问的结果 → CPU 无法预测/预取
 * → 每次 LLC Miss 发出前都必须等前一次完成 → 高延迟，低带宽
 * → 与 P6 形成对比：P7 pend≈1，P6 pend≈4
 *
 * 目标计数器：
 *   LLC-load-misses（串行）/ 内存访问延迟
 * ════════════════════════════════════════════════════════════════════ */
__attribute__((noinline))
static void phase_ptr_chase(ctx_t *c)
{
    const uintptr_t *arr = c->chase;
    const size_t n = c->chase_n;
    size_t idx = (size_t)(g_sink % n);
    for (int i = 0; i < ITER_CHASE; i++)
        idx = arr[idx];
    g_sink ^= (uint64_t)idx;
}

/* ════════════════════════════════════════════════════════════════════
 * P8: ITLB-MISS — 512 个独立代码页，乱序调用
 *
 * 每次调用跳向不同的 4KB 代码页 → 完全超出 L1 iTLB（128条目）
 * 和 STLB（~1536条目），iTLB Miss 接近饱和。
 *
 * 代码页布局（x86-64）：
 *   lea rax, [rdi + idx]  ; 返回 arg0 + idx（用于防止 DCE）
 *   ret
 *   int3 × 填充
 *
 * 目标计数器：
 *   iTLB-loads / iTLB-load-misses / L1-icache-load-misses
 * ════════════════════════════════════════════════════════════════════ */
__attribute__((noinline))
static void phase_itlb_miss(ctx_t *c)
{
    if (c->n_funcs <= 0) return;   /* mmap EXEC 失败时静默跳过 */
    int n = c->n_funcs;
    int *order = c->call_order;
    int sum = 0;
    for (int i = 0; i < ITER_ITLB; i++)
        sum += c->funcs[order[i % n]]((int)((uint64_t)g_sink & 0xff));
    g_sink ^= (uint64_t)(unsigned int)sum;
}

/* ── 写入 x86-64 函数存根 ── */
static void write_stub(uint8_t *page, int idx)
{
    /* lea rax, [rdi + idx32] ; REX.W 0x48, opcode 0x8D, ModRM 0x87 */
    page[0] = 0x48;
    page[1] = 0x8D;
    page[2] = 0x87;
    page[3] = (uint8_t)( idx        & 0xFF);
    page[4] = (uint8_t)((idx >>  8) & 0xFF);
    page[5] = (uint8_t)((idx >> 16) & 0xFF);
    page[6] = (uint8_t)((idx >> 24) & 0xFF);
    page[7] = 0xC3;   /* ret */
    memset(page + 8, 0xCC, PAGE_BYTES - 8);   /* int3 填充 */
}

/* ── Fisher-Yates 打乱整数数组 ── */
static void shuffle_ints(int *arr, int n)
{
    uint64_t rng = 0x1234567890abcdefULL ^ (uint64_t)(uintptr_t)arr;
    for (int i = n - 1; i > 0; i--) {
        xorshift64(&rng);
        int j = (int)(rng % (uint64_t)(unsigned int)(i + 1));
        int t = arr[i]; arr[i] = arr[j]; arr[j] = t;
    }
}

/* ── 构建随机指针追逐链（Fisher-Yates + LCG）── */
static void build_chase(uintptr_t *arr, size_t n)
{
    size_t *perm = malloc(n * sizeof(size_t));
    if (!perm) return;
    for (size_t i = 0; i < n; i++) perm[i] = i;
    uint64_t rng = 0x123456789abcdef0ULL;
    for (size_t i = n - 1; i > 0; i--) {
        rng = rng * 6364136223846793005ULL + 1442695040888963407ULL;
        size_t j = rng % (i + 1);
        size_t t = perm[i]; perm[i] = perm[j]; perm[j] = t;
    }
    for (size_t i = 0; i < n; i++)
        arr[perm[i]] = perm[(i + 1) % n];
    free(perm);
}

/* ── 阶段定义表 ── */
typedef struct {
    const char *name;      /* 短名称，用于日志对齐 */
    const char *counters;  /* 目标计数器描述        */
    void      (*fn)(ctx_t *);
} phase_def_t;

static const phase_def_t PHASES[] = {
    { "P1:L1D-HOT",    "L1-dcache-loads/stores, mem_inst_retired.*",       phase_l1d_hot    },
    { "P2:L2-THRASH",  "L1-dcache-load-misses, l1d.replacement",           phase_l2_thrash  },
    { "P3:LLC-HIT",    "LLC-loads/stores, cache-references",                phase_llc_hit    },
    { "P4:LLC-MISS",   "LLC-load-misses, cache-misses, dTLB-load-misses",   phase_llc_miss   },
    { "P5:DTLB-STORE", "dTLB-stores, dTLB-store-misses",                   phase_dtlb_store },
    { "P6:PEND-MISS",  "l1d_pend_miss.pending, l1d_pend_miss.fb_full",      phase_pend_miss  },
    { "P7:PTR-CHASE",  "LLC-load-misses (serial), mem latency",             phase_ptr_chase  },
    { "P8:ITLB-MISS",  "iTLB-loads, iTLB-load-misses, L1-icache-load-miss", phase_itlb_miss  },
};
#define NUM_PHASES  ((int)(sizeof(PHASES) / sizeof(PHASES[0])))

/* ████ main ████ */
int main(int argc, char **argv)
{
     /* 运行直到收到停止信号（SIGINT/SIGTERM）。
         可选参数：每阶段持续秒数，默认 5 秒。 */
     if (argc > 1) phase_sec = atoi(argv[1]);
     if (phase_sec < 1) phase_sec = 1;

    signal(SIGINT,  handle_sig);
    signal(SIGTERM, handle_sig);

    printf("=== 多阶段综合 PMU 压力测试（第二版）===\n");
    printf("PID           : %d\n", getpid());
    printf("总时长        : 直到收到停止信号（无限运行）\n");
    printf("每阶段时长    : %d 秒\n", phase_sec);
    printf("阶段数        : %d\n", NUM_PHASES);
    printf("内存需求      : ~%lu MB\n\n",
           (unsigned long)((L1D_BYTES + L2_BYTES + LLC_HIT_BYTES + DRAM_BYTES
                            + LLC_HIT_BYTES /*chase*/) >> 20));
    fflush(stdout);

    /* ── 分配工作集内存 ── */
    ctx_t ctx;
    memset(&ctx, 0, sizeof(ctx));

    ctx.l1d_buf  = calloc(L1D_BYTES     / sizeof(uint64_t), sizeof(uint64_t));
    ctx.l2_buf   = calloc(L2_BYTES      / sizeof(uint64_t), sizeof(uint64_t));
    ctx.llc_buf  = calloc(LLC_HIT_BYTES / sizeof(uint64_t), sizeof(uint64_t));
    ctx.dram_buf = malloc(DRAM_BYTES);
    ctx.chase_n  = LLC_HIT_BYTES / sizeof(uintptr_t);
    ctx.chase    = malloc(ctx.chase_n * sizeof(uintptr_t));

    if (!ctx.l1d_buf || !ctx.l2_buf || !ctx.llc_buf || !ctx.dram_buf || !ctx.chase) {
        fprintf(stderr, "内存分配失败\n");
        return 1;
    }

    /* 预热：强制物理页分配，避免 page-fault 引入测量噪声 */
    memset(ctx.dram_buf, 0x55, DRAM_BYTES);
    for (size_t i = 0; i < LLC_HIT_BYTES / sizeof(uint64_t); i++)
        ctx.llc_buf[i] = (uint64_t)i;
    g_sink = ctx.llc_buf[0];
    build_chase(ctx.chase, ctx.chase_n);

    printf("内存初始化完成。\n");

    /* ── 初始化 JIT 代码页（P8 iTLB）── */
    ctx.funcs      = calloc(NUM_CODE_PAGES, sizeof(jit_func_t));
    ctx.call_order = malloc(NUM_CODE_PAGES * sizeof(int));
    ctx.n_funcs    = 0;

    if (ctx.funcs && ctx.call_order) {
        for (int i = 0; i < NUM_CODE_PAGES; i++) {
            /* W^X：先可写，写完后切 READ|EXEC */
            uint8_t *page = mmap(NULL, PAGE_BYTES,
                                 PROT_READ | PROT_WRITE,
                                 MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
            if (page == MAP_FAILED) break;
            write_stub(page, i);
            if (mprotect(page, PAGE_BYTES, PROT_READ | PROT_EXEC) != 0) {
                munmap(page, PAGE_BYTES);
                break;
            }
            ctx.funcs[ctx.n_funcs] = (jit_func_t)(void *)page;
            ctx.call_order[ctx.n_funcs] = ctx.n_funcs;
            ctx.n_funcs++;
        }
        if (ctx.n_funcs > 0) {
            shuffle_ints(ctx.call_order, ctx.n_funcs);
            printf("iTLB 代码页: %d 页 (%lu KB)\n",
                   ctx.n_funcs, (unsigned long)(ctx.n_funcs * PAGE_BYTES >> 10));
        } else {
            printf("警告: iTLB 代码页 mmap 失败（P8 将跳过）。\n");
        }
    }
    printf("\n");
    fflush(stdout);

    /* ── 主循环 ── */
    int round = 0;

    while (g_running) {
        round++;
        printf("── Round %d ──────────────────────────────────────────────\n", round);

        for (int p = 0; p < NUM_PHASES && g_running; p++) {
            char tag[160];

            /* 阶段开始 */
            snprintf(tag, sizeof(tag), "%-16s %-52s BEGIN",
                     PHASES[p].name, PHASES[p].counters);
            print_ts(tag);

            double ph_end = now_sec() + (double)phase_sec;
            long calls = 0;
            while (g_running && now_sec() < ph_end) {
                PHASES[p].fn(&ctx);
                calls++;
            }

            /* 阶段结束，打印实际调用次数 */
            snprintf(tag, sizeof(tag), "%-16s END   calls=%-6ld  sink=0x%016llx",
                     PHASES[p].name, calls, (unsigned long long)g_sink);
            print_ts(tag);
        }
    }

    /* ── 清理 ── */
    for (int i = 0; i < ctx.n_funcs; i++)
        if (ctx.funcs[i])
            munmap((void *)(uintptr_t)(void *)ctx.funcs[i], PAGE_BYTES);
    free(ctx.funcs);
    free(ctx.call_order);
    free(ctx.l1d_buf);
    free(ctx.l2_buf);
    free(ctx.llc_buf);
    free(ctx.dram_buf);
    free(ctx.chase);

    printf("\n完成。共 %d 轮，g_sink=0x%016llx\n",
           round, (unsigned long long)g_sink);
    return 0;
}
