#include "minixgb.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

namespace {

void require(bool condition, const char* message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

template <typename Exception, typename Function>
void requireThrows(Function&& function, const char* message) {
    try {
        function();
    } catch (const Exception&) {
        return;
    }
    throw std::runtime_error(message);
}

void testMath() {
    require(std::abs(minixgb::leafWeight(-4.0, 3.0, 1.0, 0.0) - 1.0) < 1e-12,
            "Leaf weight formula failed");
    const double gain = minixgb::splitGain(-2.0, 2.0, -10.0, 1.0, 1.0, 0.0, 0.0);
    require(gain > 7.0, "Split gain formula failed");
    require(std::abs(minixgb::sigmoid(0.0) - 0.5) < 1e-12,
            "Sigmoid failed");
    require(minixgb::leafWeight(2.0, 0.0, 0.0, 0.0) == 0.0,
            "Leaf weight must handle a zero denominator");
}

void testValidation() {
    minixgb::Dataset ragged{{{1.0}, {2.0, 3.0}}, {0.0, 1.0}};
    requireThrows<std::invalid_argument>([&] { ragged.validate(); },
                                         "Ragged rows were accepted");

    minixgb::Dataset infinite_feature{
        {{std::numeric_limits<double>::infinity()}}, {0.0}};
    requireThrows<std::invalid_argument>([&] { infinite_feature.validate(); },
                                         "Infinite feature was accepted");

    minixgb::Dataset invalid_label{
        {{1.0}}, {std::numeric_limits<double>::quiet_NaN()}};
    requireThrows<std::invalid_argument>([&] { invalid_label.validate(); },
                                         "Non-finite label was accepted");
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

    requireThrows<std::invalid_argument>(
        [&] { model.predictClass(data.x, 1.1); },
        "Out-of-range classification threshold was accepted");
    requireThrows<std::invalid_argument>(
        [&] { model.predict({{1.0}}); },
        "Prediction with the wrong feature count was accepted");
}

void testBinaryValidationDataset() {
    minixgb::Dataset train{{{0.0}, {1.0}}, {0.0, 1.0}};
    minixgb::Dataset validation{{{0.5}}, {0.5}};
    minixgb::Params params;
    params.objective = minixgb::Objective::BinaryLogistic;
    params.verbose = false;
    minixgb::Booster model(params);
    requireThrows<std::invalid_argument>(
        [&] { model.fit(train, &validation); },
        "Invalid binary label in validation data was accepted");
}

void testParallelDeterminismAndMissingValues() {
    minixgb::Dataset data;
    for (int i = 0; i < 240; ++i) {
        std::vector<double> row;
        double target = 0.0;
        for (int feature = 0; feature < 8; ++feature) {
            const double value = std::sin(0.13 * i + feature) + 0.01 * i;
            if (feature == 1) {
                target += 0.8 * value;
            } else if (feature == 3) {
                target -= 0.3 * value;
            } else if (feature == 5) {
                target += std::sin(value);
            }
            row.push_back((i + feature) % 31 == 0
                              ? std::numeric_limits<double>::quiet_NaN()
                              : value);
        }
        data.y.push_back(target);
        data.x.push_back(std::move(row));
    }

    minixgb::Params serial_params;
    serial_params.n_estimators = 30;
    serial_params.max_depth = 3;
    serial_params.verbose = false;
    serial_params.n_jobs = 1;

    minixgb::Params parallel_params = serial_params;
    parallel_params.n_jobs = 4;

    minixgb::Booster serial(serial_params);
    minixgb::Booster parallel(parallel_params);
    serial.fit(data);
    parallel.fit(data);

    const auto serial_prediction = serial.predict(data.x);
    const auto parallel_prediction = parallel.predict(data.x);
    for (std::size_t i = 0; i < serial_prediction.size(); ++i) {
        require(std::abs(serial_prediction[i] - parallel_prediction[i]) < 1e-12,
                "Parallel training changed deterministic predictions");
    }
}

void testEarlyStoppingState() {
    minixgb::Dataset data;
    for (int i = 0; i < 20; ++i) {
        data.x.push_back({static_cast<double>(i)});
        data.y.push_back(4.0);
    }

    minixgb::Params params;
    params.n_estimators = 20;
    params.early_stopping_rounds = 3;
    params.verbose = false;
    minixgb::Booster model(params);
    model.fit(data, &data);

    require(model.treeCount() == 1, "Early stopping did not restore the best tree count");
    require(model.trainMetricHistory().size() == model.treeCount(),
            "Training history does not match the restored model");
    require(model.validationMetricHistory().size() == model.treeCount(),
            "Validation history does not match the restored model");
}

} // namespace

int main() {
    try {
        testMath();
        testValidation();
        testRegressionAndSerialization();
        testClassification();
        testBinaryValidationDataset();
        testParallelDeterminismAndMissingValues();
        testEarlyStoppingState();
        std::cout << "All MiniXGBoost tests passed.\n";
    } catch (const std::exception& error) {
        std::cerr << "Test failure: " << error.what() << '\n';
        return 1;
    }
    return 0;
}

