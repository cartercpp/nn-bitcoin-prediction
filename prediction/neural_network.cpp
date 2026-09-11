//
// Created by cartercpp on 9/10/26.
//

#include "neural_network.h"
#include <vector>
#include <utility>
#include <random>
#include <cstddef>
#include <cmath>
#include "math_vector.h"
#include "matrix.h"

neural_network::neural_network(
    std::initializer_list<matrix<double>> weightMatrices,
    std::initializer_list<math_vector<double>> biasVectors,
    std::initializer_list<std::size_t> neuronsPerLayer,
    double learningRate
) : m_weightMatrices{weightMatrices}, m_biasVectors{biasVectors}, m_learningRate{learningRate},
    m_neuronsPerLayer{neuronsPerLayer}
{
}

neural_network::neural_network(
    std::initializer_list<std::size_t> neuronsPerLayer,
    double learningRate
) : m_neuronsPerLayer{neuronsPerLayer}, m_learningRate{learningRate}
{
    const std::size_t layers = neuronsPerLayer.size();
    m_weightMatrices.reserve(layers);
    m_biasVectors.reserve(layers);

    std::random_device rd;

    for (std::size_t layer = 1; layer < m_neuronsPerLayer.size(); ++layer)
    {
        const std::size_t layerSize = m_neuronsPerLayer[layer],
                          prevLayerSize = m_neuronsPerLayer[layer - 1];

        std::normal_distribution<double> dist(0, std::sqrt(2 / static_cast<double>(prevLayerSize)));

        matrix<double> weights(layerSize, prevLayerSize, 0);
        for (std::size_t i = 0; i < layerSize; ++i)
            for (std::size_t i2 = 0; i2 < prevLayerSize; ++i2)
                weights[i][i2] = dist(rd);

        m_weightMatrices.emplace_back(std::move(weights));
        m_biasVectors.emplace_back(layerSize, 0);
    }
}

std::vector<math_vector<double>> neural_network::Forward(const math_vector<double>& input) const
{
    if (input.size() != m_neuronsPerLayer[0])
        throw std::invalid_argument{"Incorrect # of inputs given"};

    std::vector<math_vector<double>> activations;
    activations.reserve(m_neuronsPerLayer.size());
    activations.push_back(input);

    for (std::size_t layer = 0; layer < m_weightMatrices.size(); ++layer)
    {
        const math_vector<double>& layerActivations{activations[layer]};
        const matrix<double>& layerWeights{m_weightMatrices[layer]};
        const math_vector<double>& layerBiases{m_biasVectors[layer]};

        if (layer + 1 < m_weightMatrices.size())
            activations.emplace_back(Relu(layerWeights * layerActivations + layerBiases));
        else
            activations.emplace_back(layerWeights * layerActivations + layerBiases);
    }

    return activations;
}

void neural_network::fit(const math_vector<double>& input, const math_vector<double>& target)
{
    const std::vector<math_vector<double>> activations{Forward(input)};
    const math_vector<double>& prediction{activations.back()};
    math_vector<double> delta{prediction - target};

    std::vector<matrix<double>> weightDeltas(m_weightMatrices.size());
    std::vector<math_vector<double>> biasDeltas(m_biasVectors.size());

    for (std::size_t iter = 0; iter < m_weightMatrices.size(); ++iter)
    {
        const math_vector<double>& layerActivations{activations[activations.size() - iter - 1]};
        const math_vector<double>& prevLayerActivations{activations[activations.size() - iter - 2]};

        if (iter > 0)
            delta = delta.multiply(ReluDerivative(layerActivations));

        weightDeltas[weightDeltas.size() - iter - 1] = outer_product(delta, prevLayerActivations);
        biasDeltas[biasDeltas.size() - iter - 1] = delta;

        delta = m_weightMatrices[m_weightMatrices.size() - iter - 1].transpose() * delta;
    }

    for (std::size_t i = 0; i < m_weightMatrices.size(); ++i)
    {
        m_weightMatrices[i] -= m_learningRate * weightDeltas[i];
        m_biasVectors[i] -= m_learningRate * biasDeltas[i];
    }
}

math_vector<double> neural_network::predict(const math_vector<double>& input) const
{
    return Forward(input).back();
}

math_vector<double> neural_network::Relu(math_vector<double> vec)
{
    for (std::size_t i = 0; i < vec.size(); ++i)
        vec[i] = (vec[i] > 0) ? vec[i] : 0;

    return vec;
}

math_vector<double> neural_network::ReluDerivative(math_vector<double> vec)
{
    for (std::size_t i = 0; i < vec.size(); ++i)
        vec[i] = vec[i] > 0;

    return vec;
}

const std::vector<matrix<double>>& neural_network::weights() const noexcept
{
    return m_weightMatrices;
}

const std::vector<math_vector<double>>& neural_network::biases() const noexcept
{
    return m_biasVectors;
}