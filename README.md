# MiniXGBoost

[![CMake](https://github.com/ChengYifei-rgb/miniXGBoost/actions/workflows/cmake.yml/badge.svg)](https://github.com/ChengYifei-rgb/miniXGBoost/actions/workflows/cmake.yml)
[![C++17](https://img.shields.io/badge/C%2B%2B-17-00599C.svg)](https://isocpp.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

MiniXGBoost is a dependency-free C++17 implementation of gradient-boosted
decision trees. It implements the core ideas behind XGBoost instead of wrapping
the official library, making the training process small enough to study while
remaining useful as a tested CMake library.

## Highlights

- Second-order gradient boosting for regression and binary classification
- Exact greedy split search with learned default directions for missing values
- L1/L2 regularization, gamma pruning, and minimum child weight
- Row and column sampling with deterministic random seeds
- Bounded parallel feature search controlled by `n_jobs`
- Validation monitoring and early stopping
- Gain-based feature importance
- Portable text model serialization
- Installable CMake target: `MiniXGBoost::minixgb`
- Debug and Release CI on Linux, macOS, and Windows

## How It Works

Each boosting round computes first- and second-order derivatives of the loss,
fits a regression tree to those statistics, and adds the scaled tree output to
the current prediction. A candidate split is accepted only when its regularized
gain is positive and both children satisfy the configured constraints.

See [docs/algorithm.md](docs/algorithm.md) for the objective, leaf-weight, split
gain, missing-value, and early-stopping details.

## Build And Test

Requirements: CMake 3.16+ and a C++17 compiler.

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release --parallel
ctest --test-dir build -C Release --output-on-failure
```

Disable optional targets when embedding the library:

```bash
cmake -S . -B build \
  -DMINIXGB_BUILD_DEMO=OFF \
  -DMINIXGB_BUILD_TESTS=OFF
```

## Run The Demo

```bash
# Regression
./build/minixgb_demo

# Binary classification with missing feature values
./build/minixgb_demo classification
```

On a multi-config Windows generator, the executable is usually under
`build/Release/`.

## Library Example

```cpp
#include <minixgb.hpp>

int main() {
    minixgb::Dataset train{
        {{0.0, 1.0}, {1.0, 1.5}, {2.0, 3.0}, {3.0, 5.0}},
        {0.0, 0.0, 1.0, 1.0}
    };

    minixgb::Params params;
    params.objective = minixgb::Objective::BinaryLogistic;
    params.n_estimators = 60;
    params.max_depth = 3;
    params.learning_rate = 0.1;
    params.n_jobs = 4;
    params.verbose = false;

    minixgb::Booster model(params);
    model.fit(train);

    const double probability = model.predictOne({2.5, 4.0});
    model.saveModel("classifier.mxgb");
}
```

## Use From Another CMake Project

Install MiniXGBoost:

```bash
cmake --install build --prefix ./install
```

Then consume the exported target:

```cmake
find_package(MiniXGBoost CONFIG REQUIRED)
target_link_libraries(my_app PRIVATE MiniXGBoost::minixgb)
```

## Repository Layout

```text
include/minixgb.hpp       Public API
src/minixgb.cpp           Training, prediction, and serialization
src/main.cpp              Regression and classification demo
tests/test_minixgb.cpp    Deterministic unit and integration tests
docs/algorithm.md         Mathematical and implementation notes
```

## Scope

MiniXGBoost focuses on the in-memory, single-machine learning algorithm. It does
not implement histogram or approximate split finding, distributed training,
GPU acceleration, sparse pages, external-memory training, or language bindings.
For production workloads, use the official [XGBoost](https://github.com/dmlc/xgboost)
project.

## License

MIT
