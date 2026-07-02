#pragma once

#include <cstddef>
#include <cstdint>
#include <iosfwd>
#include <random>
#include <string>
#include <vector>

namespace minixgb {

using Matrix = std::vector<std::vector<double>>;

enum class Objective {
    SquaredError,
    BinaryLogistic
};

struct Dataset {
    Matrix x;
    std::vector<double> y;

    void validate() const;
    std::size_t rows() const noexcept { return x.size(); }
    std::size_t cols() const noexcept { return x.empty() ? 0 : x.front().size(); }
};

struct Params {
    Objective objective = Objective::SquaredError;
    int n_estimators = 100;
    int max_depth = 4;
    int min_samples_leaf = 1;
    double learning_rate = 0.1;
    double lambda = 1.0;
    double alpha = 0.0;
    double gamma = 0.0;
    double min_child_weight = 1.0;
    double subsample = 1.0;
    double colsample_bytree = 1.0;
    int early_stopping_rounds = 0;
    int n_jobs = 1;
    std::uint64_t seed = 42;
    bool verbose = true;

    void validate() const;
};

struct Node {
    bool is_leaf = true;
    int feature = -1;
    double threshold = 0.0;
    double weight = 0.0;
    double gain = 0.0;
    bool default_left = true;
    int left = -1;
    int right = -1;
};

double sigmoid(double value);
double softThreshold(double gradient, double alpha);
double leafWeight(double gradient_sum, double hessian_sum,
                  double lambda, double alpha);
double splitGain(double left_gradient, double left_hessian,
                 double right_gradient, double right_hessian,
                 double lambda, double alpha, double gamma);

class Tree {
public:
    Tree() = default;
    Tree(const Params& params, std::size_t feature_count);

    void fit(const Matrix& x,
             const std::vector<double>& gradients,
             const std::vector<double>& hessians,
             const std::vector<std::size_t>& row_indices,
             std::mt19937_64& rng);

    double predictRaw(const std::vector<double>& row) const;
    const std::vector<Node>& nodes() const noexcept { return nodes_; }
    const std::vector<double>& featureGains() const noexcept { return feature_gains_; }

    void save(std::ostream& output) const;
    void load(std::istream& input);

private:
    struct Split {
        bool valid = false;
        int feature = -1;
        double threshold = 0.0;
        double gain = 0.0;
        bool default_left = true;
    };

    int buildNode(const Matrix& x,
                  const std::vector<double>& gradients,
                  const std::vector<double>& hessians,
                  const std::vector<std::size_t>& rows,
                  const std::vector<int>& features,
                  int depth);

    Split findBestSplit(const Matrix& x,
                        const std::vector<double>& gradients,
                        const std::vector<double>& hessians,
                        const std::vector<std::size_t>& rows,
                        const std::vector<int>& features) const;

    Split findBestSplitForFeature(const Matrix& x,
                                  const std::vector<double>& gradients,
                                  const std::vector<double>& hessians,
                                  const std::vector<std::size_t>& rows,
                                  int feature) const;

    Params params_;
    std::size_t feature_count_ = 0;
    std::vector<Node> nodes_;
    std::vector<double> feature_gains_;
};

class Booster {
public:
    explicit Booster(Params params = {});

    void fit(const Dataset& train, const Dataset* validation = nullptr);
    double predictOne(const std::vector<double>& row) const;
    std::vector<double> predict(const Matrix& x) const;
    std::vector<int> predictClass(const Matrix& x, double threshold = 0.5) const;

    std::vector<double> featureImportance() const;
    const std::vector<double>& trainMetricHistory() const noexcept { return train_history_; }
    const std::vector<double>& validationMetricHistory() const noexcept { return validation_history_; }
    std::size_t treeCount() const noexcept { return trees_.size(); }
    double baseScore() const noexcept { return base_score_; }

    void saveModel(const std::string& path) const;
    void loadModel(const std::string& path);

private:
    void calculateGradients(const std::vector<double>& raw_predictions,
                            const std::vector<double>& labels,
                            std::vector<double>& gradients,
                            std::vector<double>& hessians) const;

    double metric(const std::vector<double>& raw_predictions,
                  const std::vector<double>& labels) const;

    std::vector<double> predictRaw(const Matrix& x) const;
    std::vector<std::size_t> sampleRows(std::size_t row_count, std::mt19937_64& rng) const;

    Params params_;
    std::size_t feature_count_ = 0;
    double base_score_ = 0.0;
    std::vector<Tree> trees_;
    std::vector<double> train_history_;
    std::vector<double> validation_history_;
};

} // namespace minixgb

