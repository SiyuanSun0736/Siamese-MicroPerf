/*
 * test_workload.c — PMU 采集正确性验证（6 计数器 + LBR 专项）
 *
 * 专为验证 pmu_monitor 采集的以下 7 项指标而设计，每个阶段精准驱动
 * 一个目标计数器明显升高，其余保持相对低位，形成可观测对比：
 *
 *  P1 INST-PEAK   纯算术 4 路展开，无内存压力，无随机分支
 *                 → inst_retired.any 最高（其他阶段的 3-5 倍）
 *
 *  P2 BRANCH-PRED 密集可预测分支（8 条/迭代，计数器驱动，预测命中率 >99%）
 *                 → branch-instructions 最高；branch-misses 极低
 *
 *  P3 BRANCH-RAND 随机分支（4 条/迭代，由 xorshift64 LSB 驱动）
 *                 → branch-misses / branch-instructions ≈ 40-50%
 *
 *  P4 ITLB-WARM   随机调用 N_WARM_PAGES(32) 个 JIT 代码页
 *                 → iTLB-loads 高；iTLB-load-misses ≈ 0（32页 < L1 iTLB 128条目）
 *                 → L1-icache-load-misses 中等（32×4KB=128KB > L1I 32KB）
 *
 *  P5 ITLB-COLD   随机调用 N_COLD_PAGES(512) 个 JIT 代码页
 *                 → iTLB-load-misses 最高（512页 >> L1 iTLB 128条目）
 *                 → L1-icache-load-misses 最高（512×4KB=2MB >> L1I 32KB）
 *
 *  P6 LBR-WIDE    16 路手动展开 L1D 热循环（工作集 16KB，循环体 ≈ 415 字节）
 *                 → lbr_avg_span 最高（后向跳跃跨度约等于循环体字节数）
 *
 * ─────────────────────────────────────────────────────────────────────
 * 验证方法：对齐 pmu_monitor 的采样时间戳，观察各阶段计数器数值：
 *
 *  inst_retired.any    : P1 ≫ 其他（纯算术退休率最高）
 *  L1-icache-load-misses: P5 > P4 ≫ P1/P2/P3（执行大量不同代码页）
 *  iTLB-loads          : P4 ≈ P5 ≫ P1/P2/P3（频繁访问代码页）
 *  iTLB-load-misses    : P5 ≫ P4 ≈ 0（P4 的 32 页常驻 L1 iTLB）
 *  branch-instructions : P2 ≫ 其他（8 条可预测分支/迭代）
 *  branch-misses       : P3 的 miss 率 ≈ 50%；P2 的 miss 率 < 1%
 *  lbr_avg_span        : P6 ≫ 其他（~512 字节循环体后向跳跃）
 *
 * 用法：
 *   ./test_workload [phase_sec]
 *   默认每阶段 5 秒，无限轮次直到收到 SIGINT/SIGTERM。
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
#define L1D_BYTES        (16UL * 1024)   /* 16 KB — P1/P6 工作集，常驻 L1D  */
#define PAGE_BYTES        4096UL
#define N_COLD_PAGES      512            /* P5 ITLB-COLD：超出 L1 iTLB 容量 */
#define N_WARM_PAGES       32            /* P4 ITLB-WARM：常驻 L1 iTLB      */

/* ── 每次阶段函数调用的内层迭代次数 ────────────────────────────────── */
#define ITER_COMPUTE    50000000  /* P1: 4路算术，~50ms/call @ 3GHz IPC=3  */
#define ITER_BPRED      50000000  /* P2: 8条可预测分支/迭代                 */
#define ITER_BRAND      20000000  /* P3: 4条随机分支/迭代                   */
#define ITER_JIT         3000000  /* P4/P5: 每次调用 JIT 函数总次数         */
#define ITER_LBR         2000000  /* P6: 16路展开 L1D 循环                  */

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
    uint64_t   *l1d_buf;         /* L1D_BYTES/8 个元素（P1/P6 工作集）    */
    jit_func_t *funcs;           /* N_COLD_PAGES 个独立 JIT 代码页         */
    int        *call_order_cold; /* 512 页随机顺序（P5 ITLB-COLD）         */
    int        *call_order_warm; /* 32  页随机顺序（P4 ITLB-WARM）         */
    int         n_funcs;         /* 实际分配成功的代码页数量               */
    int         n_funcs_warm;    /* min(n_funcs, N_WARM_PAGES)             */
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
 * P1: INST-PEAK — 4 路独立算术链，无内存压力，无随机分支
 *
 * 设计：四条互不依赖的 xorshift64 / LCG 链并行执行，最大化 ILP，
 *       使 CPU 以接近峰值 IPC 退休指令。
 *
 * 目标计数器：  inst_retired.any 最高（其他阶段的 3-5 倍）
 * 对比参考：    P5 ITLB-COLD 受 TLB/缓存延迟拖累，inst_retired 最低
 * ════════════════════════════════════════════════════════════════════ */
__attribute__((noinline, optimize("O2")))
static void phase_inst_peak(ctx_t *c)
{
    (void)c;
    uint64_t a = g_sink ^ 0x1111111111111111ULL;
    uint64_t b = 0x2222222222222222ULL;
    uint64_t cc = 0x3333333333333333ULL;
    uint64_t d = 0x4444444444444444ULL;
    for (int i = 0; i < ITER_COMPUTE; i++) {
        /* 四路完全独立：乱序引擎可同时 issue 所有操作 */
        a  = a  * 6364136223846793005ULL + 1442695040888963407ULL;
        b ^= b << 13; b ^= b >> 7; b ^= b << 17;
        cc = cc * 3935559000370003845ULL + 2691343689449507681ULL;
        d ^= d << 11; d ^= d >> 5; d ^= d << 23;
    }
    g_sink = a ^ b ^ cc ^ d;
}

/* ════════════════════════════════════════════════════════════════════
 * P2: BRANCH-PRED — 密集可预测分支（8 条/迭代）
 *
 * 设计：每迭代含 8 条由循环计数器驱动的条件跳转，分支方向固定可预测，
 *       CPU 分支预测器命中率接近 100%。
 *
 * 目标计数器：  branch-instructions 最高
 * 对比参考：    branch-misses 极低（< 0.1%）；与 P3 对比验证 miss 率
 * ════════════════════════════════════════════════════════════════════ */
__attribute__((noinline, optimize("O2")))
static void phase_branch_pred(ctx_t *c)
{
    (void)c;
    uint64_t s = g_sink;
    for (int i = 0; i < ITER_BPRED; i++) {
        /* 8 条可预测分支：计数器模运算，方向固定 */
        if (i & 1)         s += (uint64_t)i;          else s ^= (uint64_t)i;
        if (i & 2)         s ^= (uint64_t)(i >> 1);   else s += (uint64_t)(i | 1);
        if (i % 4  == 0)   s += 1;
        if (i % 8  == 0)   s ^= 0xAAAAAAAAAAAAAAAAULL;
        if (i % 16 == 0)   s += 0x1234567890ABCDEFULL;
        if (i % 32 == 0)   s ^= s >> 17;
        if (i % 64 == 0)   s += s >> 13;
        if (i % 128 == 0)  s ^= 0xDEADBEEFCAFEBABEULL;
    }
    g_sink = s;
}

/* ════════════════════════════════════════════════════════════════════
 * P3: BRANCH-RAND — 随机不可预测分支（4 条/迭代）
 *
 * 设计：每迭代用 xorshift64 输出的不同位驱动 4 条独立分支，
 *       各位随机均匀 → 每条分支预测成功率 ≈ 50%。
 *
 * 目标计数器：  branch-misses / branch-instructions ≈ 40-50%
 * 对比参考：    P2 同等分支密度但 miss 率 < 0.1%，形成鲜明对比
 * ════════════════════════════════════════════════════════════════════ */
/* optimize("no-if-conversion")：禁止 GCC if-conversion pass 将 if/else 转为
 * CMOV（条件移动）指令；若编译为 CMOV 则不产生真实分支，branch-misses 为 0。 */
__attribute__((noinline, optimize("O2", "no-if-conversion")))
static void phase_branch_rand(ctx_t *c)
{
    (void)c;
    uint64_t rng = g_sink ^ 0xDEADBEEFCAFE1234ULL;
    uint64_t s = 0;
    for (int i = 0; i < ITER_BRAND; i++) {
        xorshift64(&rng);
        /* 4 条分支各由独立随机位驱动：无法被预测 */
        if (rng & 0x01ULL)  s += rng;          else s ^= rng;
        if (rng & 0x02ULL)  s ^= (rng >> 1);   else s += (rng >> 3);
        if (rng & 0x04ULL)  s *=  3;           else s += 7;
        if (rng & 0x08ULL)  s ^= 0xABULL;      else s += 0x12ULL;
    }
    g_sink ^= s;
}

/* ════════════════════════════════════════════════════════════════════
 * P4: ITLB-WARM — 循环调用 N_WARM_PAGES(32) 个 JIT 代码页
 *
 * 设计：32 页 × 4KB = 128KB；L1 iTLB 通常有 128-256 个 4KB 条目，
 *       预热后 32 页全部常驻 iTLB，后续访问全部命中。
 *       32 页 > L1I 缓存（32KB），故仍有 L1-icache-load-misses。
 *
 * 目标计数器：  iTLB-loads 高；iTLB-load-misses ≈ 0（预热后）
 * 对比参考：    与 P5 对比：iTLB-load-misses 数值悬殊
 * ════════════════════════════════════════════════════════════════════ */
__attribute__((noinline))
static void phase_itlb_warm(ctx_t *c)
{
    if (c->n_funcs_warm <= 0) return;
    int   n     = c->n_funcs_warm;
    int  *order = c->call_order_warm;
    int   sum   = 0;
    for (int i = 0; i < ITER_JIT; i++)
        sum += c->funcs[order[i % n]]((int)((uint64_t)g_sink & 0xff));
    g_sink ^= (uint64_t)(unsigned int)sum;
}

/* ════════════════════════════════════════════════════════════════════
 * P5: ITLB-COLD — 随机调用 N_COLD_PAGES(512) 个 JIT 代码页
 *
 * 设计：512 页 >> L1 iTLB 容量（128 条目），随机乱序访问确保
 *       每次都从不同页取指令，L1 iTLB 持续冷触 miss。
 *       512 页 × 4KB = 2MB >> L1I（32KB）→ icache miss 同样最高。
 *
 * 目标计数器：  iTLB-load-misses 最高；L1-icache-load-misses 最高
 * 对比参考：    P4 的 iTLB-load-misses 接近 0；P1 的 icache miss 最低
 * ════════════════════════════════════════════════════════════════════ */
__attribute__((noinline))
static void phase_itlb_cold(ctx_t *c)
{
    if (c->n_funcs <= 0) return;
    int   n     = c->n_funcs;
    int  *order = c->call_order_cold;
    int   sum   = 0;
    for (int i = 0; i < ITER_JIT; i++)
        sum += c->funcs[order[i % n]]((int)((uint64_t)g_sink & 0xff));
    g_sink ^= (uint64_t)(unsigned int)sum;
}

/* ════════════════════════════════════════════════════════════════════
 * P6: LBR-WIDE — 16 路手动展开的 L1D 热循环（验证 lbr_avg_span）
 *
 * 设计：16 路展开使循环体字节数 ≈ 512 字节，循环末尾的后向跳跃
 *       跨度即循环体大小，直接决定 lbr_avg_span（每条 LBR 记录的
 *       主要来源是该后向跳跃）。工作集 16KB 常驻 L1D，icache/iTLB
 *       均无压力，保证 lbr_avg_span 反映循环体大小而非其他因素。
 *
 * 目标计数器：  lbr_avg_span 最高（≈ 循环体字节数，其他阶段通常 70-110）
 * 对比参考：    P2 BRANCH-PRED 循环体紧凑（约 70 字节），span 较低
 * ════════════════════════════════════════════════════════════════════ */
__attribute__((noinline, optimize("O2")))
static void phase_lbr_wide(ctx_t *c)
{
    const size_t n = L1D_BYTES / sizeof(uint64_t);   /* 2048，2 的幂 */
    uint64_t *b = c->l1d_buf;
    uint64_t s = g_sink;
    /* 16 路展开：每组 2 条指令（load-XOR + store），16 组/迭代
     * 编译后循环体 ≈ 415 字节（实测均值）
     * → lbr_avg_span 明显高于其他阶段的紧凑循环                              */
    for (int i = 0; i < ITER_LBR; i++) {
        size_t k = (size_t)i & (n - 1);
#define STEP(OFF) \
        s ^= b[(k+(OFF)) & (n-1)]; \
        b[(k+(OFF)) & (n-1)] = s + (uint64_t)(OFF);
        STEP( 0) STEP( 1) STEP( 2) STEP( 3)
        STEP( 4) STEP( 5) STEP( 6) STEP( 7)
        STEP( 8) STEP( 9) STEP(10) STEP(11)
        STEP(12) STEP(13) STEP(14) STEP(15)
#undef STEP
    }
    g_sink = s;
}

/* ── 写入 x86-64 JIT 函数存根（W^X）── */
static void write_stub(uint8_t *page, int idx)
{
    /* lea rax, [rdi + idx32]  ; REX.W=0x48, LEA=0x8D, ModRM=0x87 */
    page[0] = 0x48;
    page[1] = 0x8D;
    page[2] = 0x87;
    page[3] = (uint8_t)( idx        & 0xFF);
    page[4] = (uint8_t)((idx >>  8) & 0xFF);
    page[5] = (uint8_t)((idx >> 16) & 0xFF);
    page[6] = (uint8_t)((idx >> 24) & 0xFF);
    page[7] = 0xC3;              /* ret */
    memset(page + 8, 0xCC, PAGE_BYTES - 8);   /* int3 填充 */
}

/* ── Fisher-Yates 随机打乱整数数组 ── */
static void shuffle_ints(int *arr, int n)
{
    uint64_t rng = 0x1234567890ABCDEFULL ^ (uint64_t)(uintptr_t)arr;
    for (int i = n - 1; i > 0; i--) {
        xorshift64(&rng);
        int j = (int)(rng % (uint64_t)(unsigned int)(i + 1));
        int t = arr[i]; arr[i] = arr[j]; arr[j] = t;
    }
}

/* ── 阶段定义表 ── */
typedef struct {
    const char *name;      /* 短名，用于日志对齐      */
    const char *target;    /* 目标计数器（预期最高值） */
    void      (*fn)(ctx_t *);
} phase_def_t;

static const phase_def_t PHASES[] = {
    { "P1:INST-PEAK",   "inst_retired.any HIGH",                          phase_inst_peak   },
    { "P2:BRANCH-PRED", "branch-instructions HIGH / branch-misses LOW",   phase_branch_pred },
    { "P3:BRANCH-RAND", "branch-misses/branch-inst ~50%",                 phase_branch_rand },
    { "P4:ITLB-WARM",   "iTLB-loads HIGH / iTLB-load-misses ~0",         phase_itlb_warm   },
    { "P5:ITLB-COLD",   "iTLB-load-misses HIGH / L1-icache-miss HIGH",    phase_itlb_cold   },
    { "P6:LBR-WIDE",    "lbr_avg_span HIGH (wide loop body)",             phase_lbr_wide    },
};
#define NUM_PHASES  ((int)(sizeof(PHASES) / sizeof(PHASES[0])))

/* ████ main ████ */
int main(int argc, char **argv)
{
    /* 可选参数：每阶段持续秒数，默认 5 秒 */
    if (argc > 1) phase_sec = atoi(argv[1]);
    if (phase_sec < 1) phase_sec = 1;

    signal(SIGINT,  handle_sig);
    signal(SIGTERM, handle_sig);

    printf("=== PMU 采集正确性验证测试 ===\n");
    printf("PID         : %d\n", getpid());
    printf("每阶段时长  : %d 秒\n", phase_sec);
    printf("阶段数      : %d\n", NUM_PHASES);
    printf("\n");
    printf("验证预期（与 pmu_monitor 采样数据对比）：\n");
    printf("  P1 INST-PEAK   inst_retired.any            >> 其他阶段\n");
    printf("  P2 BRANCH-PRED branch-instructions HIGH    / branch-misses < 0.1%%\n");
    printf("  P3 BRANCH-RAND branch-misses/branch-inst   ~~ 40-50%%\n");
    printf("  P4 ITLB-WARM   iTLB-loads HIGH             / iTLB-load-misses ~~ 0\n");
    printf("  P5 ITLB-COLD   iTLB-load-misses HIGH       / L1-icache-load-misses HIGH\n");
    printf("  P6 LBR-WIDE    lbr_avg_span                >> 其他阶段\n");
    printf("\n");
    fflush(stdout);

    /* ── 分配工作集内存 ── */
    ctx_t ctx;
    memset(&ctx, 0, sizeof(ctx));

    ctx.l1d_buf = calloc(L1D_BYTES / sizeof(uint64_t), sizeof(uint64_t));
    if (!ctx.l1d_buf) {
        fprintf(stderr, "内存分配失败\n");
        return 1;
    }

    /* ── 初始化 JIT 代码页（P4/P5）── */
    ctx.funcs           = calloc(N_COLD_PAGES, sizeof(jit_func_t));
    ctx.call_order_cold = malloc(N_COLD_PAGES * sizeof(int));
    ctx.call_order_warm = malloc(N_WARM_PAGES * sizeof(int));
    ctx.n_funcs         = 0;

    if (ctx.funcs && ctx.call_order_cold && ctx.call_order_warm) {
        for (int i = 0; i < N_COLD_PAGES; i++) {
            /* W^X：先 WRITE 写存根，再切换为 READ|EXEC */
            uint8_t *page = mmap(NULL, PAGE_BYTES,
                                 PROT_READ | PROT_WRITE,
                                 MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
            if (page == MAP_FAILED) break;
            write_stub(page, i);
            if (mprotect(page, PAGE_BYTES, PROT_READ | PROT_EXEC) != 0) {
                munmap(page, PAGE_BYTES);
                break;
            }
            ctx.funcs[ctx.n_funcs]           = (jit_func_t)(void *)page;
            ctx.call_order_cold[ctx.n_funcs] = ctx.n_funcs;
            ctx.n_funcs++;
        }

        /* 冷序列：全部 N_COLD_PAGES 页，随机打乱 */
        shuffle_ints(ctx.call_order_cold, ctx.n_funcs);

        /* 热序列：仅使用前 N_WARM_PAGES 页（或实际分配数），随机打乱 */
        ctx.n_funcs_warm = ctx.n_funcs < N_WARM_PAGES ? ctx.n_funcs : N_WARM_PAGES;
        for (int i = 0; i < ctx.n_funcs_warm; i++)
            ctx.call_order_warm[i] = i;
        shuffle_ints(ctx.call_order_warm, ctx.n_funcs_warm);

        if (ctx.n_funcs > 0)
            printf("JIT 代码页: 冷集=%d页(%luKB)  热集=%d页(%luKB)\n",
                   ctx.n_funcs,      (unsigned long)(ctx.n_funcs      * PAGE_BYTES >> 10),
                   ctx.n_funcs_warm, (unsigned long)(ctx.n_funcs_warm * PAGE_BYTES >> 10));
        else
            printf("警告: JIT 代码页 mmap 失败（P4/P5 将跳过）。\n");
    }
    printf("\n");
    fflush(stdout);

    /* ── 主循环 ── */
    int round = 0;
    while (g_running) {
        round++;
        printf("── Round %d ────────────────────────────────────────────\n", round);

        for (int p = 0; p < NUM_PHASES && g_running; p++) {
            char tag[160];

            /* 阶段开始 */
            snprintf(tag, sizeof(tag), "%-17s %-50s BEGIN",
                     PHASES[p].name, PHASES[p].target);
            print_ts(tag);

            double ph_end = now_sec() + (double)phase_sec;
            long   calls  = 0;
            while (g_running && now_sec() < ph_end) {
                PHASES[p].fn(&ctx);
                calls++;
            }

            /* 阶段结束 */
            snprintf(tag, sizeof(tag), "%-17s END  calls=%-6ld sink=0x%016llx",
                     PHASES[p].name, calls, (unsigned long long)g_sink);
            print_ts(tag);
        }
    }

    /* ── 清理 ── */
    for (int i = 0; i < ctx.n_funcs; i++)
        if (ctx.funcs[i])
            munmap((void *)(uintptr_t)(void *)ctx.funcs[i], PAGE_BYTES);
    free(ctx.funcs);
    free(ctx.call_order_cold);
    free(ctx.call_order_warm);
    free(ctx.l1d_buf);

    printf("\n完成。共 %d 轮，g_sink=0x%016llx\n",
           round, (unsigned long long)g_sink);
    return 0;
}
