//
// Created by cartercpp on 9/10/26.
//

#ifndef BITCOINNN_NEURAL_NETWORK_H
#define BITCOINNN_NEURAL_NETWORK_H

#include <initializer_list>
#include <vector>
#include <cstddef>
#include "math_vector.h"
#include "matrix.h"

class neural_network
{
public:

    // CONSTRUCTORS

    explicit neural_network(
        std::initializer_list<matrix<double>>,
        std::initializer_list<math_vector<double>>,
        std::initializer_list<std::size_t>,
        double
    );

    explicit neural_network(
        std::initializer_list<std::size_t>,
        double
    );

    // METHODS

    math_vector<double> predict(const math_vector<double>&) const;
    void fit(const math_vector<double>&, const math_vector<double>&);

    const std::vector<matrix<double>>& weights() const noexcept;
    const std::vector<math_vector<double>>& biases() const noexcept;

private:

    static math_vector<double> Relu(math_vector<double>);
    static math_vector<double> ReluDerivative(math_vector<double>);

    std::vector<math_vector<double>> Forward(const math_vector<double>&) const;

    std::vector<matrix<double>> m_weightMatrices;
    std::vector<math_vector<double>> m_biasVectors;
    std::vector<std::size_t> m_neuronsPerLayer;
    double m_learningRate;
};

#endif //BITCOINNN_NEURAL_NETWORK_H