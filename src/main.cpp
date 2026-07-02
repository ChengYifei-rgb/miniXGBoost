#include "minixgb.hpp"

#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <string>

namespace {

minixgb::Dataset makeRegressionData(std::size_t count, std::uint64_t seed) {
    minixgb::Dataset data;
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<double> distribution(-3.0, 3.0);
    std::normal_distribution<double> noise(0.0, 0.12);

    for (std::size_t i = 0; i < count; ++i) {
        const double x1 = distribution(rng);
        const double x2 = distribution(rng);
        data.x.push_back({x1, x2});
        data.y.push_back(std::sin(x1) + 0.35 * x2 * x2 + noise(rng));
    }
    return data;
}

minixgb::Dataset makeClassificationData(std::size_t count, std::uint64_t seed) {
    minixgb::Dataset data;
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<double> distribution(-2.0, 2.0);

    for (std::size_t i = 0; i < count; ++i) {
        double x1 = distribution(rng);
        const double x2 = distribution(rng);
        if (i % 17 == 0) {
            x1 = std::numeric_limits<double>::quiet_NaN();
        }
        const double clean_x1 = std::isnan(x1) ? 0.0 : x1;
        const double radius = clean_x1 * clean_x1 + x2 * x2;
        data.x.push_back({x1, x2});
        data.y.push_back(radius > 1.6 ? 1.0 : 0.0);
    }
    return data;
}

void printImportance(const minixgb::Booster& model) {
    const auto importance = model.featureImportance();
    std::cout << "Feature importance (normalized gain):\n";
    for (std::size_t i = 0; i < importance.size(); ++i) {
        std::cout << "  f" << i << ": " << importance[i] << '\n';
    }
}

} // namespace

int main(int argc, char** argv) {
    try {
        const bool classification = argc > 1 && std::string(argv[1]) == "classification";
        minixgb::Params params;
        params.objective = classification
            ? minixgb::Objective::BinaryLogistic
            : minixgb::Objective::SquaredError;
        params.n_estimators = 80;
        params.max_depth = 4;
        params.learning_rate = 0.08;
        params.lambda = 1.0;
        params.alpha = 0.02;
        params.gamma = 0.001;
        params.subsample = 0.9;
        params.colsample_bytree = 1.0;
        params.n_jobs = 4;
        params.early_stopping_rounds = 10;
        params.verbose = true;

        const minixgb::Dataset train = classification
            ? makeClassificationData(600, 7)
            : makeRegressionData(600, 7);
        const minixgb::Dataset validation = classification
            ? makeClassificationData(200, 99)
            : makeRegressionData(200, 99);

        minixgb::Booster model(params);
        model.fit(train, &validation);

        std::cout << "\nTrained trees: " << model.treeCount() << '\n';
        printImportance(model);

        const std::string model_path = classification
            ? "classification.mxgb"
            : "regression.mxgb";
        model.saveModel(model_path);
        std::cout << "Saved model: " << model_path << '\n';

        minixgb::Booster restored;
        restored.loadModel(model_path);
        std::cout << std::fixed << std::setprecision(6)
                  << "Reloaded model prediction for first row: "
                  << restored.predictOne(validation.x.front()) << '\n';
    } catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << '\n';
        return 1;
    }
    return 0;
}

