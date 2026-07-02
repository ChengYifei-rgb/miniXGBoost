#include "minixgb.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <future>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <utility>

namespace minixgb {
namespace {

constexpr double kEpsilon = 1e-12;

double nodeScore(double gradient_sum, double hessian_sum,
                 double lambda, double alpha) {
    if (hessian_sum <= 0.0) {
        return 0.0;
    }
    const double shrunk = softThreshold(gradient_sum, alpha);
    return shrunk * shrunk / (hessian_sum + lambda);
}

double clampProbability(double value) {
    return std::clamp(value, 1e-15, 1.0 - 1e-15);
}

std::string objectiveName(Objective objective) {
    return objective == Objective::SquaredError ? "squared_error" : "binary_logistic";
}

Objective parseObjective(const std::string& value) {
    if (value == "squared_error") {
        return Objective::SquaredError;
    }
    if (value == "binary_logistic") {
        return Objective::BinaryLogistic;
    }
    throw std::runtime_error("Unknown objective in model file: " + value);
}

} // namespace

void Dataset::validate() const {
    if (x.empty()) {
        throw std::invalid_argument("Dataset must contain at least one row");
    }
    if (x.size() != y.size()) {
        throw std::invalid_argument("Feature rows and labels must have equal length");
    }
    const std::size_t width = x.front().size();
    if (width == 0) {
        throw std::invalid_argument("Dataset must contain at least one feature");
    }
    for (const auto& row : x) {
        if (row.size() != width) {
            throw std::invalid_argument("Every feature row must have equal width");
        }
    }
}

void Params::validate() const {
    if (n_estimators <= 0 || max_depth < 0 || min_samples_leaf <= 0) {
        throw std::invalid_argument("Invalid tree count, depth, or leaf size");
    }
    if (!(learning_rate > 0.0 && learning_rate <= 1.0)) {
        throw std::invalid_argument("learning_rate must be in (0, 1]");
    }
    if (lambda < 0.0 || alpha < 0.0 || gamma < 0.0 || min_child_weight < 0.0) {
        throw std::invalid_argument("Regularization values cannot be negative");
    }
    if (!(subsample > 0.0 && subsample <= 1.0) ||
        !(colsample_bytree > 0.0 && colsample_bytree <= 1.0)) {
        throw std::invalid_argument("Sampling ratios must be in (0, 1]");
    }
    if (early_stopping_rounds < 0 || n_jobs <= 0) {
        throw std::invalid_argument("early_stopping_rounds and n_jobs are invalid");
    }
}

double sigmoid(double value) {
    if (value >= 0.0) {
        const double z = std::exp(-value);
        return 1.0 / (1.0 + z);
    }
    const double z = std::exp(value);
    return z / (1.0 + z);
}

double softThreshold(double gradient, double alpha) {
    if (gradient > alpha) {
        return gradient - alpha;
    }
    if (gradient < -alpha) {
        return gradient + alpha;
    }
    return 0.0;
}

double leafWeight(double gradient_sum, double hessian_sum,
                  double lambda, double alpha) {
    return -softThreshold(gradient_sum, alpha) / (hessian_sum + lambda);
}

double splitGain(double left_gradient, double left_hessian,
                 double right_gradient, double right_hessian,
                 double lambda, double alpha, double gamma) {
    const double parent_gradient = left_gradient + right_gradient;
    const double parent_hessian = left_hessian + right_hessian;
    return 0.5 * (
        nodeScore(left_gradient, left_hessian, lambda, alpha) +
        nodeScore(right_gradient, right_hessian, lambda, alpha) -
        nodeScore(parent_gradient, parent_hessian, lambda, alpha)
    ) - gamma;
}

Tree::Tree(const Params& params, std::size_t feature_count)
    : params_(params), feature_count_(feature_count), feature_gains_(feature_count, 0.0) {}

void Tree::fit(const Matrix& x,
               const std::vector<double>& gradients,
               const std::vector<double>& hessians,
               const std::vector<std::size_t>& row_indices,
               std::mt19937_64& rng) {
    nodes_.clear();
    feature_gains_.assign(feature_count_, 0.0);

    std::vector<int> features(feature_count_);
    std::iota(features.begin(), features.end(), 0);
    std::shuffle(features.begin(), features.end(), rng);
    const auto selected_count = std::max<std::size_t>(
        1, static_cast<std::size_t>(std::ceil(feature_count_ * params_.colsample_bytree)));
    features.resize(std::min(selected_count, features.size()));
    std::sort(features.begin(), features.end());

    buildNode(x, gradients, hessians, row_indices, features, 0);
}

int Tree::buildNode(const Matrix& x,
                    const std::vector<double>& gradients,
                    const std::vector<double>& hessians,
                    const std::vector<std::size_t>& rows,
                    const std::vector<int>& features,
                    int depth) {
    double gradient_sum = 0.0;
    double hessian_sum = 0.0;
    for (std::size_t row : rows) {
        gradient_sum += gradients[row];
        hessian_sum += hessians[row];
    }

    const int node_id = static_cast<int>(nodes_.size());
    Node node;
    node.weight = leafWeight(gradient_sum, hessian_sum, params_.lambda, params_.alpha);
    nodes_.push_back(node);

    if (depth >= params_.max_depth ||
        rows.size() < static_cast<std::size_t>(2 * params_.min_samples_leaf) ||
        hessian_sum < 2.0 * params_.min_child_weight) {
        return node_id;
    }

    const Split best = findBestSplit(x, gradients, hessians, rows, features);
    if (!best.valid || best.gain <= kEpsilon) {
        return node_id;
    }

    std::vector<std::size_t> left_rows;
    std::vector<std::size_t> right_rows;
    left_rows.reserve(rows.size());
    right_rows.reserve(rows.size());

    for (std::size_t row : rows) {
        const double value = x[row][static_cast<std::size_t>(best.feature)];
        const bool goes_left = std::isnan(value) ? best.default_left : value <= best.threshold;
        (goes_left ? left_rows : right_rows).push_back(row);
    }

    if (left_rows.size() < static_cast<std::size_t>(params_.min_samples_leaf) ||
        right_rows.size() < static_cast<std::size_t>(params_.min_samples_leaf)) {
        return node_id;
    }

    nodes_[node_id].is_leaf = false;
    nodes_[node_id].feature = best.feature;
    nodes_[node_id].threshold = best.threshold;
    nodes_[node_id].gain = best.gain;
    nodes_[node_id].default_left = best.default_left;
    feature_gains_[static_cast<std::size_t>(best.feature)] += best.gain;

    const int left_id = buildNode(x, gradients, hessians, left_rows, features, depth + 1);
    const int right_id = buildNode(x, gradients, hessians, right_rows, features, depth + 1);
    nodes_[node_id].left = left_id;
    nodes_[node_id].right = right_id;
    return node_id;
}

Tree::Split Tree::findBestSplit(const Matrix& x,
                                const std::vector<double>& gradients,
                                const std::vector<double>& hessians,
                                const std::vector<std::size_t>& rows,
                                const std::vector<int>& features) const {
    Split best;

    if (params_.n_jobs > 1 && features.size() > 1) {
        std::vector<std::future<Split>> jobs;
        jobs.reserve(features.size());
        for (int feature : features) {
            jobs.push_back(std::async(std::launch::async, [&, feature] {
                return findBestSplitForFeature(x, gradients, hessians, rows, feature);
            }));
        }
        for (auto& job : jobs) {
            Split candidate = job.get();
            if (candidate.valid && (!best.valid || candidate.gain > best.gain)) {
                best = candidate;
            }
        }
    } else {
        for (int feature : features) {
            Split candidate = findBestSplitForFeature(x, gradients, hessians, rows, feature);
            if (candidate.valid && (!best.valid || candidate.gain > best.gain)) {
                best = candidate;
            }
        }
    }
    return best;
}

Tree::Split Tree::findBestSplitForFeature(
    const Matrix& x,
    const std::vector<double>& gradients,
    const std::vector<double>& hessians,
    const std::vector<std::size_t>& rows,
    int feature) const {
    struct Entry {
        double value;
        double gradient;
        double hessian;
    };

    std::vector<Entry> entries;
    entries.reserve(rows.size());
    double missing_gradient = 0.0;
    double missing_hessian = 0.0;
    std::size_t missing_count = 0;

    for (std::size_t row : rows) {
        const double value = x[row][static_cast<std::size_t>(feature)];
        if (std::isnan(value)) {
            missing_gradient += gradients[row];
            missing_hessian += hessians[row];
            ++missing_count;
        } else {
            entries.push_back({value, gradients[row], hessians[row]});
        }
    }

    if (entries.size() < 2) {
        return {};
    }

    std::sort(entries.begin(), entries.end(),
              [](const Entry& a, const Entry& b) { return a.value < b.value; });

    double total_gradient = 0.0;
    double total_hessian = 0.0;
    for (const Entry& entry : entries) {
        total_gradient += entry.gradient;
        total_hessian += entry.hessian;
    }

    double left_gradient = 0.0;
    double left_hessian = 0.0;
    Split best;

    for (std::size_t i = 0; i + 1 < entries.size(); ++i) {
        left_gradient += entries[i].gradient;
        left_hessian += entries[i].hessian;

        if (entries[i].value == entries[i + 1].value) {
            continue;
        }

        const double right_gradient = total_gradient - left_gradient;
        const double right_hessian = total_hessian - left_hessian;
        const std::size_t left_count = i + 1;
        const std::size_t right_count = entries.size() - left_count;
        const double threshold = entries[i].value +
                                 (entries[i + 1].value - entries[i].value) * 0.5;

        auto consider = [&](bool default_left) {
            const double gl = left_gradient + (default_left ? missing_gradient : 0.0);
            const double hl = left_hessian + (default_left ? missing_hessian : 0.0);
            const double gr = right_gradient + (default_left ? 0.0 : missing_gradient);
            const double hr = right_hessian + (default_left ? 0.0 : missing_hessian);
            const std::size_t lc = left_count + (default_left ? missing_count : 0);
            const std::size_t rc = right_count + (default_left ? 0 : missing_count);

            if (lc < static_cast<std::size_t>(params_.min_samples_leaf) ||
                rc < static_cast<std::size_t>(params_.min_samples_leaf) ||
                hl < params_.min_child_weight || hr < params_.min_child_weight) {
                return;
            }

            const double gain = splitGain(gl, hl, gr, hr,
                                          params_.lambda, params_.alpha, params_.gamma);
            if (!best.valid || gain > best.gain) {
                best = {true, feature, threshold, gain, default_left};
            }
        };

        consider(true);
        consider(false);
    }
    return best;
}

double Tree::predictRaw(const std::vector<double>& row) const {
    if (nodes_.empty()) {
        throw std::logic_error("Cannot predict with an empty tree");
    }
    int node_id = 0;
    while (!nodes_[static_cast<std::size_t>(node_id)].is_leaf) {
        const Node& node = nodes_[static_cast<std::size_t>(node_id)];
        const double value = row[static_cast<std::size_t>(node.feature)];
        const bool goes_left = std::isnan(value) ? node.default_left : value <= node.threshold;
        node_id = goes_left ? node.left : node.right;
    }
    return nodes_[static_cast<std::size_t>(node_id)].weight;
}

void Tree::save(std::ostream& output) const {
    output << feature_count_ << ' ' << nodes_.size() << '\n';
    output << std::setprecision(17);
    for (const Node& node : nodes_) {
        output << node.is_leaf << ' ' << node.feature << ' ' << node.threshold << ' '
               << node.weight << ' ' << node.gain << ' ' << node.default_left << ' '
               << node.left << ' ' << node.right << '\n';
    }
}

void Tree::load(std::istream& input) {
    std::size_t node_count = 0;
    input >> feature_count_ >> node_count;
    feature_gains_.assign(feature_count_, 0.0);
    nodes_.assign(node_count, {});
    for (Node& node : nodes_) {
        input >> node.is_leaf >> node.feature >> node.threshold >> node.weight
              >> node.gain >> node.default_left >> node.left >> node.right;
        if (!node.is_leaf && node.feature >= 0) {
            feature_gains_[static_cast<std::size_t>(node.feature)] += node.gain;
        }
    }
    if (!input) {
        throw std::runtime_error("Corrupt tree data in model file");
    }
}

Booster::Booster(Params params) : params_(std::move(params)) {
    params_.validate();
}

void Booster::fit(const Dataset& train, const Dataset* validation) {
    params_.validate();
    train.validate();
    if (validation != nullptr) {
        validation->validate();
        if (validation->cols() != train.cols()) {
            throw std::invalid_argument("Train and validation feature counts differ");
        }
    }

    if (params_.objective == Objective::BinaryLogistic) {
        for (double label : train.y) {
            if (label != 0.0 && label != 1.0) {
                throw std::invalid_argument("Binary labels must be exactly 0 or 1");
            }
        }
    }

    feature_count_ = train.cols();
    trees_.clear();
    train_history_.clear();
    validation_history_.clear();

    const double label_mean = std::accumulate(train.y.begin(), train.y.end(), 0.0) /
                              static_cast<double>(train.y.size());
    base_score_ = params_.objective == Objective::SquaredError
        ? label_mean
        : std::log(clampProbability(label_mean) / (1.0 - clampProbability(label_mean)));

    std::vector<double> train_raw(train.rows(), base_score_);
    std::vector<double> validation_raw;
    if (validation != nullptr) {
        validation_raw.assign(validation->rows(), base_score_);
    }

    std::vector<double> gradients(train.rows());
    std::vector<double> hessians(train.rows());
    std::mt19937_64 rng(params_.seed);
    double best_metric = std::numeric_limits<double>::infinity();
    std::size_t best_tree_count = 0;
    int rounds_without_improvement = 0;

    for (int round = 0; round < params_.n_estimators; ++round) {
        calculateGradients(train_raw, train.y, gradients, hessians);
        const auto rows = sampleRows(train.rows(), rng);

        Tree tree(params_, feature_count_);
        tree.fit(train.x, gradients, hessians, rows, rng);

        for (std::size_t i = 0; i < train.rows(); ++i) {
            train_raw[i] += params_.learning_rate * tree.predictRaw(train.x[i]);
        }
        if (validation != nullptr) {
            for (std::size_t i = 0; i < validation->rows(); ++i) {
                validation_raw[i] += params_.learning_rate * tree.predictRaw(validation->x[i]);
            }
        }

        trees_.push_back(std::move(tree));
        const double train_metric = metric(train_raw, train.y);
        train_history_.push_back(train_metric);
        const double watched_metric = validation != nullptr
            ? metric(validation_raw, validation->y)
            : train_metric;
        if (validation != nullptr) {
            validation_history_.push_back(watched_metric);
        }

        if (params_.verbose) {
            std::cout << "[" << round + 1 << "] train=" << train_metric;
            if (validation != nullptr) {
                std::cout << " validation=" << watched_metric;
            }
            std::cout << '\n';
        }

        if (watched_metric + kEpsilon < best_metric) {
            best_metric = watched_metric;
            best_tree_count = trees_.size();
            rounds_without_improvement = 0;
        } else {
            ++rounds_without_improvement;
        }

        if (params_.early_stopping_rounds > 0 &&
            rounds_without_improvement >= params_.early_stopping_rounds) {
            trees_.resize(best_tree_count);
            if (params_.verbose) {
                std::cout << "Early stopping at " << best_tree_count << " trees\n";
            }
            break;
        }
    }
}

void Booster::calculateGradients(const std::vector<double>& raw_predictions,
                                 const std::vector<double>& labels,
                                 std::vector<double>& gradients,
                                 std::vector<double>& hessians) const {
    for (std::size_t i = 0; i < labels.size(); ++i) {
        if (params_.objective == Objective::SquaredError) {
            gradients[i] = raw_predictions[i] - labels[i];
            hessians[i] = 1.0;
        } else {
            const double probability = sigmoid(raw_predictions[i]);
            gradients[i] = probability - labels[i];
            hessians[i] = std::max(probability * (1.0 - probability), 1e-16);
        }
    }
}

double Booster::metric(const std::vector<double>& raw_predictions,
                       const std::vector<double>& labels) const {
    double total = 0.0;
    if (params_.objective == Objective::SquaredError) {
        for (std::size_t i = 0; i < labels.size(); ++i) {
            const double error = raw_predictions[i] - labels[i];
            total += error * error;
        }
        return std::sqrt(total / static_cast<double>(labels.size()));
    }
    for (std::size_t i = 0; i < labels.size(); ++i) {
        const double probability = clampProbability(sigmoid(raw_predictions[i]));
        total -= labels[i] * std::log(probability) +
                 (1.0 - labels[i]) * std::log(1.0 - probability);
    }
    return total / static_cast<double>(labels.size());
}

std::vector<std::size_t> Booster::sampleRows(std::size_t row_count,
                                             std::mt19937_64& rng) const {
    std::vector<std::size_t> rows(row_count);
    std::iota(rows.begin(), rows.end(), 0);
    if (params_.subsample < 1.0) {
        std::shuffle(rows.begin(), rows.end(), rng);
        const auto count = std::max<std::size_t>(
            1, static_cast<std::size_t>(std::ceil(row_count * params_.subsample)));
        rows.resize(count);
    }
    return rows;
}

std::vector<double> Booster::predictRaw(const Matrix& x) const {
    std::vector<double> result(x.size(), base_score_);
    for (std::size_t i = 0; i < x.size(); ++i) {
        if (x[i].size() != feature_count_) {
            throw std::invalid_argument("Prediction row has the wrong feature count");
        }
        for (const Tree& tree : trees_) {
            result[i] += params_.learning_rate * tree.predictRaw(x[i]);
        }
    }
    return result;
}

double Booster::predictOne(const std::vector<double>& row) const {
    const double raw = predictRaw(Matrix{row}).front();
    return params_.objective == Objective::BinaryLogistic ? sigmoid(raw) : raw;
}

std::vector<double> Booster::predict(const Matrix& x) const {
    std::vector<double> result = predictRaw(x);
    if (params_.objective == Objective::BinaryLogistic) {
        std::transform(result.begin(), result.end(), result.begin(), sigmoid);
    }
    return result;
}

std::vector<int> Booster::predictClass(const Matrix& x, double threshold) const {
    if (params_.objective != Objective::BinaryLogistic) {
        throw std::logic_error("predictClass requires BinaryLogistic objective");
    }
    const auto probabilities = predict(x);
    std::vector<int> classes(probabilities.size());
    std::transform(probabilities.begin(), probabilities.end(), classes.begin(),
                   [threshold](double p) { return p >= threshold ? 1 : 0; });
    return classes;
}

std::vector<double> Booster::featureImportance() const {
    std::vector<double> importance(feature_count_, 0.0);
    for (const Tree& tree : trees_) {
        const auto& gains = tree.featureGains();
        for (std::size_t i = 0; i < importance.size(); ++i) {
            importance[i] += gains[i];
        }
    }
    const double total = std::accumulate(importance.begin(), importance.end(), 0.0);
    if (total > 0.0) {
        for (double& value : importance) {
            value /= total;
        }
    }
    return importance;
}

void Booster::saveModel(const std::string& path) const {
    std::ofstream output(path);
    if (!output) {
        throw std::runtime_error("Cannot open model file for writing: " + path);
    }
    output << "MINIXGB 1\n";
    output << objectiveName(params_.objective) << '\n';
    output << std::setprecision(17)
           << params_.learning_rate << ' ' << params_.lambda << ' ' << params_.alpha << ' '
           << params_.gamma << ' ' << params_.min_child_weight << '\n';
    output << feature_count_ << ' ' << base_score_ << ' ' << trees_.size() << '\n';
    for (const Tree& tree : trees_) {
        tree.save(output);
    }
}

void Booster::loadModel(const std::string& path) {
    std::ifstream input(path);
    if (!input) {
        throw std::runtime_error("Cannot open model file for reading: " + path);
    }
    std::string magic;
    int version = 0;
    input >> magic >> version;
    if (magic != "MINIXGB" || version != 1) {
        throw std::runtime_error("Unsupported MiniXGBoost model file");
    }
    std::string objective;
    input >> objective;
    params_.objective = parseObjective(objective);
    input >> params_.learning_rate >> params_.lambda >> params_.alpha
          >> params_.gamma >> params_.min_child_weight;
    std::size_t tree_count = 0;
    input >> feature_count_ >> base_score_ >> tree_count;
    trees_.assign(tree_count, Tree(params_, feature_count_));
    for (Tree& tree : trees_) {
        tree.load(input);
    }
    train_history_.clear();
    validation_history_.clear();
}

} // namespace minixgb

