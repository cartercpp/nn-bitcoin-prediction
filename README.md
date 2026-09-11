# nn-bitcoin-prediction

A from-scratch feed-forward neural network in modern C++ that trains on historical Bitcoin prices and predicts the next day's close.

No PyTorch, Eigen, or other ML libraries — just C++23 and the standard library, plus small custom `matrix` and `math_vector` types.

## Architecture

- Input: last **365** daily prices, min-max normalized to `[0, 1]`
- Hidden layers: **64** and **32** units with ReLU
- Output: **1** linear unit (normalized next-day price)
- Learning rate: `0.005`
- Weight init: He / Kaiming normal (`N(0, sqrt(2 / fan_in))`), biases start at zero
- Training: sliding-window SGD, one sample at a time

The same network class is used in all three programs. Training starts from random weights. Prediction and visualization load a trained snapshot baked into `btcNN.cpp`.

## Layout

| Directory | What it does |
| --- | --- |
| `training/` | Train the network on `btc.txt`. Prints average percent loss each epoch. On Enter, writes `weights.txt` and `biases.txt`. |
| `prediction/` | Autoregressive forecast from the last 365 prices. Prints a yearly snapshot and keeps rolling the predicted price forward. Press Enter to stop. |
| `visualization/` | Terminal chart of actual prices (`-`) vs network predictions (`o`) as the window slides through history. |

Each directory is self-contained (`main.cpp`, `neural_network.{h,cpp}`, `matrix.h`, `math_vector.h`, plus shared data files).

## Data

`btc.txt` is one closing price per line (~3462 days). The programs currently open a hardcoded path:

```text
/home/cartercpp/Documents/C++/BitcoinNN/btc.txt
```

Point that path at the `btc.txt` next to each `main.cpp` (or wherever you keep the series) before building.

## Build

Requires a C++23 compiler (`std::print`, `std::ranges::to`, `std::jthread`, chrono calendars).

```bash
# training
c++ -std=c++23 -O2 training/main.cpp training/neural_network.cpp -o train

# prediction (needs the baked-in weights in btcNN.cpp)
c++ -std=c++23 -O2 prediction/main.cpp prediction/neural_network.cpp prediction/btcNN.cpp -o predict

# visualization
c++ -std=c++23 -O2 visualization/main.cpp visualization/neural_network.cpp visualization/btcNN.cpp -o visualize
```

## Run

```bash
./train       # Enter stops training and dumps weights/biases
./predict     # Enter stops the forecast loop
./visualize   # slides through history in the terminal
```

This is a teaching demo, not trading advice. A plain MLP with a 365-day window will not reliably forecast Bitcoin.
