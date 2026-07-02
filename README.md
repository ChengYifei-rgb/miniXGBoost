# MiniXGBoost

一个从零实现的现代 C++17 梯度提升树项目，用于学习 XGBoost 的核心算法，而不是官方库的包装器。

## 已实现功能

- 二阶梯度提升
- 平方误差回归与二元逻辑分类
- 精确贪心分裂搜索
- L1/L2 正则化、`gamma` 与最小子节点权重
- 最大深度和最小叶子样本数
- 行采样、列采样
- 缺失值默认方向学习
- 并行特征分裂搜索（`std::async`）
- 验证集监控与早停
- 基于增益的特征重要性
- 文本模型保存和加载
- 自动化回归、分类及序列化测试

## 构建

```powershell
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release
ctest --test-dir build -C Release --output-on-failure
```

## 运行

回归演示：

```powershell
.\build\minixgb_demo.exe
```

分类演示：

```powershell
.\build\minixgb_demo.exe classification
```

## 阅读顺序

1. `include/minixgb.hpp`：数据结构和公开接口。
2. `src/minixgb.cpp` 中的 `leafWeight` 和 `splitGain`：数学核心。
3. `Tree::findBestSplitForFeature`：精确贪心分裂。
4. `Tree::buildNode`：递归建树。
5. `Booster::fit`：逐轮添加树并更新梯度。

## 工程边界

本项目完整覆盖单机内存版 XGBoost 的核心训练链路，但没有实现官方项目的直方图近似、分布式训练、GPU、稀疏页、外存训练和各语言绑定。那些属于工业扩展，不影响通过本项目学习二阶提升树的算法本质。

