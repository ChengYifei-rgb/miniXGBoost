#include "minixgb.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <iostream>
#include <stdexcept>

namespace {

void require(bool condition, const char* message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

void testMath() {
    require(std::abs(minixgb::leafWeight(-4.0, 3.0, 1.0, 0.0) - 1.0) < 1e-12,
            "Leaf weight formula failed");
    const double gain = minixgb::splitGain(-2.0, 2.0, -10.0, 1.0, 1.0, 0.0, 0.0);
    require(gain > 7.0, "Split gain formula failed");
    require(std::abs(minixgb::sigmoid(0.0) - 0.5) < 1e-12,
            "Sigmoid failed");
}

void testRegressionAndSerialization() {
    minixgb::Dataset data;
    for (int i = 0; i < 120; ++i) {
        const double x = -3.0 + 6.0 * i / 119.0;
        data.x.push_back({x, x * x});
        data.y.push_back(std::sin(x) + 0.2 * x * x);
    }

    minixgb::Params params;
    params.n_estimators = 100;
    params.max_depth = 4;
    params.learning_rate = 0.1;
    params.verbose = false;
    minixgb::Booster model(params);
    model.fit(data);

    const auto prediction = model.predict(data.x);
    double mse = 0.0;
    for (std::size_t i = 0; i < data.rows(); ++i) {
        const double error = prediction[i] - data.y[i];
        mse += error * error;
    }
    mse /= static_cast<double>(data.rows());
    require(std::sqrt(mse) < 0.08, "Regression did not fit the training data");

    const char* model_path = "minixgb_test_model.tmp";
    model.saveModel(model_path);
    minixgb::Booster restored;
    restored.loadModel(model_path);
    const auto restored_prediction = restored.predict(data.x);
    std::remove(model_path);
    for (std::size_t i = 0; i < prediction.size(); ++i) {
        require(std::abs(prediction[i] - restored_prediction[i]) < 1e-12,
                "Serialization changed predictions");
    }
}

void testClassification() {
    minixgb::Dataset data;
    for (int i = -10; i <= 10; ++i) {
        for (int j = -10; j <= 10; ++j) {
            const double x1 = i / 5.0;
            const double x2 = j / 5.0;
            data.x.push_back({x1, x2});
            data.y.push_back(x1 * x1 + x2 * x2 > 1.5 ? 1.0 : 0.0);
        }
    }

    minixgb::Params params;
    params.objective = minixgb::Objective::BinaryLogistic;
    params.n_estimators = 80;
    params.max_depth = 4;
    params.learning_rate = 0.12;
    params.verbose = false;
    minixgb::Booster model(params);
    model.fit(data);
    const auto classes = model.predictClass(data.x);
    int correct = 0;
    for (std::size_t i = 0; i < classes.size(); ++i) {
        correct += classes[i] == static_cast<int>(data.y[i]);
    }
    const double accuracy = static_cast<double>(correct) / classes.size();
    require(accuracy > 0.97, "Classification accuracy is too low");
}

} // namespace

int main() {
    try {
        testMath();
        testRegressionAndSerialization();
        testClassification();
        std::cout << "All MiniXGBoost tests passed.\n";
    } catch (const std::exception& error) {
        std::cerr << "Test failure: " << error.what() << '\n';
        return 1;
    }
    return 0;
}

