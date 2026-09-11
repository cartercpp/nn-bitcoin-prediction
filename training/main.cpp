#include <iostream>
#include <format>
#include <print>
#include <fstream>
#include <vector>
#include <algorithm>
#include <ranges>
#include <thread>
#include <stop_token>
#include <cstddef>
#include "neural_network.h"

template <typename ValueType>
struct std::formatter<std::vector<ValueType>>
{
    constexpr auto parse(std::format_parse_context& ctx)
    {
        return ctx.begin();
    }

    auto format(const std::vector<ValueType>& vec, std::format_context& ctx) const
    {
        auto out = ctx.out();

        *out++ = '{';

        for (std::size_t i = 0; i < vec.size(); ++i)
        {
            if (i != 0)
                out = std::format_to(out, ", ");

            out = std::format_to(out, "{}", vec[i]);
        }

        *out++ = '}';

        return out;
    }
};

int main()
{
    std::vector<double> btcPriceHistory;
    btcPriceHistory.reserve(3462);

    {
        std::ifstream file{"/home/cartercpp/Documents/C++/BitcoinNN/btc.txt"};
        double price;
        while (file >> price)
            btcPriceHistory.push_back(price);
    }

    const auto [minPrice, maxPrice] = std::ranges::minmax(btcPriceHistory);
    auto normalizeBTCPrice = [minPrice, maxPrice](double price) {
        return (price - minPrice) / (maxPrice - minPrice);
    };

    constexpr std::size_t inputSize = 365;
    neural_network btcNN({inputSize, 64, 32, 1}, 0.005);

    {
        std::jthread thr{[&](std::stop_token st) {
            std::size_t epoch = 0;

            while (!st.stop_requested())
            {
                double averagePercentLoss = 0;

                for (std::size_t start = 0; start + inputSize < btcPriceHistory.size(); ++start)
                {
                    const auto normInput
                            = std::ranges::subrange(btcPriceHistory.begin() + start,
                                                    btcPriceHistory.begin() + start + inputSize)
                              | std::views::transform(normalizeBTCPrice)
                              | std::ranges::to<std::vector<double> >();

                    const double normTarget = normalizeBTCPrice(btcPriceHistory[start + inputSize]);
                    const double normPrediction = btcNN.predict(normInput)[0];
                    averagePercentLoss += std::abs(normPrediction - normTarget) / normTarget
                            * (1 / static_cast<double>(btcPriceHistory.size() - inputSize + 1));

                    btcNN.fit(math_vector<double>{normInput}, math_vector<double>(1, normTarget));
                }

                std::println("Epoch {} average training loss: {:.2f}%", epoch, averagePercentLoss * 100);
                ++epoch;
            }
        }};
        std::cin.get();
    }

    std::ofstream weightsFile{"/home/cartercpp/Documents/C++/BitcoinNN/weights.txt"};
    weightsFile << std::format("{}", btcNN.weights());

    std::ofstream biasesFile{"/home/cartercpp/Documents/C++/BitcoinNN/biases.txt"};
    biasesFile << std::format("{}", btcNN.biases());
}
