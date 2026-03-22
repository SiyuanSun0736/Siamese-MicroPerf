/*
 * tid_monitor.h — 通过内核 proc connector 监听新 TID 的创建与退出
 *
 * 使用 NETLINK_CONNECTOR / CN_IDX_PROC 订阅内核进程事件（fork / clone /
 * exit），在不借助 ptrace 的情况下实时获得子线程 / 子进程通知，并允许
 * 调用方立即为每个新 TID 挂载独立的 perf_event。
 *
 * 内核通知粒度为"线程"（TID），因此线程与进程均可被检测到。
 * 需要 CAP_NET_ADMIN（通常即 root）权限。
 */

#ifndef TID_MONITOR_H
#define TID_MONITOR_H

#include <sys/types.h>

/* ── 回调类型 ─────────────────────────────────────────────────────────────── */

/*
 * tid_born_cb_t — 监控到新 TID 出现时被调用。
 *   tid      : 新线程 / 进程的 TID（pthread 场景等于 gettid()）
 *   userdata : 透传指针
 *
 * 注意：回调在 tid_monitor_dispatch() 调用栈中同步执行，
 *       不允许在回调内递归调用 tid_monitor_*。
 */
typedef void (*tid_born_cb_t)(pid_t tid, void *userdata);

/*
 * tid_dead_cb_t — 监控到 TID 退出时被调用。
 */
typedef void (*tid_dead_cb_t)(pid_t tid, void *userdata);

/* ── 接口函数 ─────────────────────────────────────────────────────────────── */

/*
 * tid_monitor_init — 建立 proc connector netlink 订阅。
 *
 *   root_pid  : 要跟踪的进程树根 PID；其 fork / clone 派生的所有子代
 *               均会被自动纳入跟踪集，并触发 on_born 回调。
 *   on_born   : 新 TID 出现时的回调（可为 NULL）。
 *   on_dead   : TID 退出时的回调（可为 NULL）。
 *   userdata  : 透传给回调的任意指针。
 *
 * 成功返回 0，失败（权限不足 / 内核不支持）返回 -1。
 */
int  tid_monitor_init(pid_t root_pid,
                      tid_born_cb_t on_born,
                      tid_dead_cb_t on_dead,
                      void *userdata);

/*
 * tid_monitor_fd — 返回内部 netlink socket fd，供外部 poll() / epoll 使用。
 * 在 tid_monitor_init() 成功返回后才有效。
 */
int  tid_monitor_fd(void);

/*
 * tid_monitor_dispatch — 从 netlink socket 读取并处理所有当前待处理事件。
 * 应在 poll() 报告 fd 可读之后立即调用；内部使用 MSG_DONTWAIT，
 * 一次调用会排空队列直到 EAGAIN。
 */
void tid_monitor_dispatch(void);

/*
 * tid_monitor_close — 向内核发送 PROC_CN_MCAST_IGNORE 并关闭 socket。
 */
void tid_monitor_close(void);

#endif /* TID_MONITOR_H */
