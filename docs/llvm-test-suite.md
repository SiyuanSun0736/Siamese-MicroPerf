
```bash
cmake -DCMAKE_C_COMPILER=/usr/bin/clang \
      -DCMAKE_CXX_COMPILER=/usr/bin/clang++ \
      -C ../cmake/caches/O3.cmake \
      ..
```
还存在问题：
```bash
cmake -DTEST_SUITE_INLINE_LOOP=ON \
      -DCMAKE_C_COMPILER=/usr/bin/clang \
      -DCMAKE_CXX_COMPILER=/usr/bin/clang++ \
      -C ../cmake/caches/O1-g.cmake ..
```
或者考虑
时间换机制（动态附加模式）：放弃自动继承。在外部引入监控层（如监听内核的 clone/fork / exec 动作），在 hexxagon 创建新子线程的瞬间，立刻由监控层手动为这个新 TID 分配并挂载独立的 LBR 采集事件。