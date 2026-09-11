#include <iostream>
#include <fstream>
#include <vector>
#include <deque>
#include <algorithm>
#include <ranges>
#include <thread>
#include <chrono>
#include <cstddef>
#include "neural_network.h"

extern neural_network btcNN;

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

    auto denormalizeBTCPrice = [minPrice, maxPrice](double percent) {
        return minPrice + percent * (maxPrice - minPrice);
    };

    constexpr std::size_t inputSize = 365,
                          rows = 80,
                          columns = inputSize + 1;
    constexpr char aiChar = 'o',
                   normalChar = '-';

    char grid[rows][columns];

    std::cout << "\033[?25l";
    std::cout << "\033[2J";

    std::deque<double> normNNPredictions;

    for (std::size_t start = 0; start + inputSize < btcPriceHistory.size(); ++start)
    {
        const auto normInput =
            std::ranges::subrange(btcPriceHistory.begin() + start, btcPriceHistory.begin() + start + inputSize)
            | std::views::transform(normalizeBTCPrice) | std::ranges::to<std::vector<double>>();
        const double normPrediction = btcNN.predict(normInput)[0];

        normNNPredictions.push_back(normPrediction);
        if (normNNPredictions.size() > columns)
            normNNPredictions.pop_front();

        for (std::size_t row = 0; row < rows; ++row)
            for (std::size_t column = 0; column < columns; ++column)
                grid[row][column] = ' ';

        for (std::size_t column = 0; column < columns; ++column)
        {
            const double normPrice = normalizeBTCPrice(btcPriceHistory[start + column]);
            const std::size_t row = rows - 1 - static_cast<std::size_t>(normPrice * (rows - 1));
            grid[row][column] = normalChar;
        }

        for (std::size_t i = 0; i < normNNPredictions.size(); ++i)
        {
            const double normPrice = normNNPredictions[i];
            const std::size_t column = (columns - normNNPredictions.size()) + i,
                              row = rows - 1 - static_cast<std::size_t>(normPrice * (rows - 1));
            grid[row][column] = aiChar;
        }

        std::cout << "\033[0H";
        for (std::size_t row = 0; row < rows; ++row)
        {
            for (std::size_t column = 0; column < columns; ++column)
            {
                switch (grid[row][column])
                {
                    case aiChar:
                        std::cout << "\033[38;2;0;255;255m";
                        break;
                    case normalChar:
                        std::cout << "\033[38;2;220;220;220m";
                        break;
                }

                std::cout << grid[row][column];
            }

            std::cout << '\n';
        }
        std::cout << std::flush;
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }

    std::cout << "\033[?25h";
}
