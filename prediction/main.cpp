#include <iostream>
#include <print>
#include <fstream>
#include <vector>
#include <deque>
#include <algorithm>
#include <ranges>
#include <thread>
#include <stop_token>
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

    std::cout << "\033[2J";
    std::cout << "\033[0H";
    std::cout.imbue(std::locale{"en_US.UTF-8"});

    constexpr std::size_t inputSize = 365;

    std::deque<double> normPrevPredictions =
        std::ranges::subrange(btcPriceHistory.end() - inputSize, btcPriceHistory.end())
        | std::views::transform(normalizeBTCPrice) | std::ranges::to<std::deque<double>>();

    std::chrono::year_month_day date{
        std::chrono::year{2026},
        std::chrono::month{9},
        std::chrono::day{10}
    };
    std::size_t epoch = 0;

    std::jthread thr{[&](std::stop_token st) {
        while (!st.stop_requested())
        {
            const auto normInput = normPrevPredictions | std::ranges::to<std::vector<double>>();
            const double normPrediction = btcNN.predict(normInput)[0];

            normPrevPredictions.pop_front();
            normPrevPredictions.push_back(normPrediction);

            if (epoch % 365 == 0)
            {
                std::println(std::cout, "{:%m/%d/%Y} Prediction: ${:.2Lf}",
                            date, denormalizeBTCPrice(normPrediction));
                std::this_thread::sleep_for(std::chrono::seconds(1));
            }

            ++epoch;
            date = std::chrono::year_month_day{std::chrono::sys_days{date} + std::chrono::days{1}};
        }
    }};
    std::cin.get();
}
