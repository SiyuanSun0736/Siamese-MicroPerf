/*
 * tid_monitor.c — proc connector 监控层实现
 *
 * 内核通过 NETLINK_CONNECTOR / CN_IDX_PROC 向用户态发送进程生命周期事件：
 *   PROC_EVENT_FORK  — 进程 / 线程通过 clone() 创建新子代
 *   PROC_EVENT_EXIT  — 线程或进程退出
 *
 * 本模块维护一个"被跟踪 TID 集合"，以根 PID 为起点，将每次 FORK 事件的
 * 子 TID 加入集合（仅当父 TGID 或父 TID 已在集合中时），并在 EXIT 事件
 * 时将其移除。每次加入 / 移除时分别触发用户注册的回调。
 *
 * 注意：
 *   · proc connector 报告的是内核 TID（= gettid()），对于多线程进程，
 *     同一 TGID 下每条线程拥有独立的 TID。
 *   · 需要 CAP_NET_ADMIN 权限（通常以 root 运行）。
 *   · socket 使用 SOCK_NONBLOCK，dispatch() 内部以 MSG_DONTWAIT 排空队列。
 */

#define _GNU_SOURCE
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <stdint.h>
#include <unistd.h>
#include <errno.h>
#include <sys/socket.h>
#include <linux/netlink.h>
#include <linux/connector.h>
#include <linux/cn_proc.h>

#include "tid_monitor.h"

/* ── 跟踪集合配置 ────────────────────────────────────────────────────────── */

/*
 * 最大并发跟踪 TID 数量。线性扫描；4096 足以覆盖大多数应用场景，
 * 若需更大规模可改为哈希表。
 */
#define MAX_TRACKED_TIDS  4096

/* ── 模块私有状态 ─────────────────────────────────────────────────────────── */

static int            nl_fd       = -1;
static tid_born_cb_t  cb_born     = NULL;
static tid_dead_cb_t  cb_dead     = NULL;
static void          *cb_userdata = NULL;

/* 被跟踪的 TID 集合（含根进程及其所有派生线程 / 子进程）*/
static pid_t tracked[MAX_TRACKED_TIDS];
static int   tracked_count = 0;

/* ── 跟踪集合操作（线性扫描，O(n)，n 通常 < 数百）─────────────────────────── */

static int tracked_contains(pid_t tid)
{
    for (int i = 0; i < tracked_count; i++)
        if (tracked[i] == tid) return 1;
    return 0;
}

static void tracked_add(pid_t tid)
{
    if (tracked_contains(tid)) return;
    if (tracked_count >= MAX_TRACKED_TIDS) {
        fprintf(stderr, "tid_monitor: tracked set full (limit=%d)\n",
                MAX_TRACKED_TIDS);
        return;
    }
    tracked[tracked_count++] = tid;
}

static void tracked_remove(pid_t tid)
{
    for (int i = 0; i < tracked_count; i++) {
        if (tracked[i] == tid) {
            /* 用最后一个元素填洞，保持紧凑 */
            tracked[i] = tracked[--tracked_count];
            return;
        }
    }
}

/* ── proc connector 订阅 / 取消订阅 ─────────────────────────────────────── */

static int send_connector_op(enum proc_cn_mcast_op op)
{
    /* 构造一个对齐的复合消息：nlmsghdr + cn_msg + 操作码 */
    struct {
        struct nlmsghdr          nl;
        struct cn_msg            cn;
        enum proc_cn_mcast_op    data;
    } __attribute__((packed)) msg;

    memset(&msg, 0, sizeof(msg));

    msg.nl.nlmsg_len   = sizeof(msg);
    msg.nl.nlmsg_type  = NLMSG_DONE;
    msg.nl.nlmsg_flags = 0;
    msg.nl.nlmsg_seq   = 0;
    msg.nl.nlmsg_pid   = (uint32_t)getpid();

    msg.cn.id.idx = CN_IDX_PROC;
    msg.cn.id.val = CN_VAL_PROC;
    msg.cn.len    = sizeof(enum proc_cn_mcast_op);

    msg.data = op;

    if (send(nl_fd, &msg, sizeof(msg), 0) < 0) {
        perror("tid_monitor: send connector op");
        return -1;
    }
    return 0;
}

/* ── 公共接口 ────────────────────────────────────────────────────────────── */

int tid_monitor_init(pid_t root_pid,
                     tid_born_cb_t on_born,
                     tid_dead_cb_t on_dead,
                     void *userdata)
{
    nl_fd = socket(PF_NETLINK,
                   SOCK_DGRAM | SOCK_NONBLOCK | SOCK_CLOEXEC,
                   NETLINK_CONNECTOR);
    if (nl_fd < 0) {
        perror("tid_monitor: socket(NETLINK_CONNECTOR)");
        return -1;
    }

    struct sockaddr_nl sa;
    memset(&sa, 0, sizeof(sa));
    sa.nl_family = AF_NETLINK;
    sa.nl_groups = CN_IDX_PROC;       /* 订阅 proc event 多播组 */
    sa.nl_pid    = (uint32_t)getpid();

    if (bind(nl_fd, (struct sockaddr *)&sa, sizeof(sa)) < 0) {
        perror("tid_monitor: bind");
        close(nl_fd);
        nl_fd = -1;
        return -1;
    }

    /* 向内核注册"开始监听"意愿 */
    if (send_connector_op(PROC_CN_MCAST_LISTEN) < 0) {
        close(nl_fd);
        nl_fd = -1;
        return -1;
    }

    cb_born     = on_born;
    cb_dead     = on_dead;
    cb_userdata = userdata;

    /* 将根进程加入初始跟踪集 */
    tracked_count = 0;
    tracked_add(root_pid);

    return 0;
}

int tid_monitor_fd(void)
{
    return nl_fd;
}

void tid_monitor_dispatch(void)
{
    /*
     * 使用足够大的缓冲区容纳多个 netlink 消息（内核可能批量发送）。
     * 4096 字节覆盖约 20+ 条紧凑的 proc_event 记录。
     */
    char buf[4096];
    ssize_t n;

    while ((n = recv(nl_fd, buf, sizeof(buf), MSG_DONTWAIT)) > 0) {
        struct nlmsghdr *nlh = (struct nlmsghdr *)buf;

        for (; NLMSG_OK(nlh, (unsigned int)n); nlh = NLMSG_NEXT(nlh, n)) {
            if (nlh->nlmsg_type == NLMSG_NOOP  ||
                nlh->nlmsg_type == NLMSG_ERROR ||
                nlh->nlmsg_type == NLMSG_OVERRUN)
                continue;

            struct cn_msg *cn = (struct cn_msg *)NLMSG_DATA(nlh);

            /* 忽略非 proc connector 的消息 */
            if (cn->id.idx != CN_IDX_PROC || cn->id.val != CN_VAL_PROC)
                continue;

            struct proc_event *ev = (struct proc_event *)cn->data;

            switch (ev->what) {
            case PROC_EVENT_FORK: {
                /*
                 * parent_pid  = 父 TID（线程级）
                 * parent_tgid = 父 TGID（进程级）
                 * child_pid   = 新 TID
                 * child_tgid  = 新 TGID（进程）
                 *
                 * 只要父亲的"任意一层身份"在跟踪集中，就接纳子代。
                 */
                pid_t ptid  = (pid_t)ev->event_data.fork.parent_pid;
                pid_t ptgid = (pid_t)ev->event_data.fork.parent_tgid;
                pid_t ctid  = (pid_t)ev->event_data.fork.child_pid;
                pid_t ctgid = (pid_t)ev->event_data.fork.child_tgid;

                if (tracked_contains(ptid) || tracked_contains(ptgid)) {
                    /* 将新 TGID（若不同于 ctid 则代表新进程）一并加入 */
                    if (ctgid != ctid)
                        tracked_add(ctgid);
                    tracked_add(ctid);

                    if (cb_born)
                        cb_born(ctid, cb_userdata);
                }
                break;
            }

            case PROC_EVENT_EXIT: {
                /*
                 * process_pid  = 退出的 TID
                 * process_tgid = 退出的 TGID
                 * 当最后一条线程退出时 process_pid == process_tgid。
                 */
                pid_t tid = (pid_t)ev->event_data.exit.process_pid;

                if (tracked_contains(tid)) {
                    tracked_remove(tid);
                    if (cb_dead)
                        cb_dead(tid, cb_userdata);
                }
                break;
            }

            default:
                break;
            }
        }
    }

    /* EAGAIN / EWOULDBLOCK 表示队列已空，属正常情况 */
    if (n < 0 && errno != EAGAIN && errno != EWOULDBLOCK)
        perror("tid_monitor: recv");
}

void tid_monitor_close(void)
{
    if (nl_fd >= 0) {
        send_connector_op(PROC_CN_MCAST_IGNORE);
        close(nl_fd);
        nl_fd = -1;
    }
    tracked_count = 0;
}
