# Crypto Short Trading Analyzer

## Overview
This Streamlit application enables simulation and optimization of short trading strategies for cryptocurrencies using historical data. It supports RSI-based trading strategies, grid search parameter optimization, and comprehensive visualization of trading results.

## Installation
1. **Requirements**:
   - Python 3.8+
   - Required libraries: `streamlit`, `pandas`, `numpy`, `plotly`
   - Install dependencies:
     ```bash
     pip install streamlit pandas numpy plotly
     ```

2. **Running the App**:
   - Download `short_strategy_simulation_streamlit.py`.
   - Run the following command in your terminal:
     ```bash
     streamlit run short_strategy_simulation_streamlit.py
     ```
   - The app will launch in your browser at `http://localhost:8501`.
[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://short-strategy-simulation.streamlit.app/)

## Input Data Format
The app expects a CSV file with historical data in the following format:
- **Columns**: `coin;volume;open;high;low;close;datetime`
- **Delimiter**: `;`
- **Date Formats**: Supports `DD.MM.YYYY HH:MM`, `DD.MM.YYYY HH:MM:SS`, `YYYY-MM-DD HH:MM:SS`, `MM/DD/YYYY HH:MM`
- **Example**:
  ```
  coin;volume;open;high;low;close;datetime
  BTC;1000.5;45000;45200;44900;45100;13.07.2025 00:00
  ```

## User Interface Guide
### 1. **Uploading Data**
- In the sidebar, click **"📁 Data Upload"** and select your CSV file.
- Ensure the file adheres to the specified format. Errors will be displayed if the format is incorrect.

### 2. **Selecting Coins**
- In the sidebar, choose the coin selection method:
  - **Single Coin**: Select one coin from the dropdown.
  - **Multiple Coins**: Use multiselect to choose multiple coins.
  - **All Coins**: Analyze all coins available in the dataset.

### 3. **Operation Modes**
The app supports two modes:
#### a) **Strategy Testing**
- Configure parameters in the sidebar:
  - **RSI Period**: RSI calculation period (5–30).
  - **RSI Overbought**: Threshold for short entry (60–90).
  - **Position Size (%)**: Percentage of balance per position (5–50%).
  - **Stop Loss (%)**: Maximum loss for position closure (2–15%).
  - **Take Profit (%)**: Target profit for position closure (2–15%).
  - **Max Averaging**: Number of additional entries (0–5).
  - **Averaging Threshold (%)**: Price change for averaging (5–20%).
  - **Averaging Multiplier**: Multiplier for averaging position size (1.0–3.0).
- Click **"🚀 Run Test"**.
- **Results**:
  - **Single Coin**: Displays metrics (win rate, PnL, drawdown), equity curve, PnL distribution, and trade markers on the price chart.
  - **Multiple Coins**: Shows comparative tables, metric histograms, and aggregated trade details.

#### b) **Parameter Optimization**
- Specify parameter ranges (similar to testing).
- Click **"🚀 Run Optimization"**.
- The app performs a grid search across all parameter combinations.
- **Results**:
  - Summary table per coin.
  - Best configurations by win rate, PnL, and drawdown.
  - Heatmap for analyzing metric dependencies (e.g., stop-loss vs. take-profit).
  - Detailed table of all results with sorting options.

### 4. **Exporting Results**
- **Testing**: Download a CSV of trades or copy strategy parameters.
- **Optimization**: Download a CSV of all trades or review detailed results.

## Strategy Features
- **Entry Signal**: Short when RSI exceeds the overbought threshold and price reaches a local high.
- **Position Management**:
  - Stop-loss and take-profit for risk control.
  - Averaging on price increases (with a limit).
  - Breakeven stop-loss adjustment at ≥1% profit.
- **Metrics**: Win rate, total PnL, average PnL, profit factor, max drawdown.

## Limitations
- Does not account for exchange fees or slippage.
- Optimization may be computationally intensive with many parameter combinations.
- Requires high-quality historical data for accurate results.

## Tips
- Start with optimization to identify optimal parameters.
- Test strategies on out-of-sample data for real-world validation.
- Use minute-level data for precise modeling.
