Blackjack Machine Learning Analysis

This project analyzes blackjack simulation data using machine learning to predict optimal player actions based on hand composition and dealer upcard.

ML Models: Logistic Regression, Random Forest, XGBoost, Neural Networks
Ensemble Methods: Voting classifier combining top performers
Win Probability Prediction: Regression models to predict expected outcomes
Strategy Visualization: Heatmaps showing optimal actions for different hand combinations
Feature Engineering: Hand totals, soft/hard hands, pairs, split/double capabilities

The analysis uses blackjack_simulator.csv containing:
- Player hands and dealer upcards
- Actions taken (Hit/Stand/Double/Split)
- Win/loss outcomes
- Card counting metrics (excluded from ML features)

- Predict optimal action (Hit/Stand/Double/Split)
- Trained on 500k samples from simulation data  
- Performance metrics: MSE, MAE, R²

Input Formats
Player Hand: [10, 11] or K A or 10 11
Dealer Upcard: 6 or 6 or A for Ace
Face Cards: J, Q, K = 10, A = 11

For running the analysis models download the data set from here: https://www.kaggle.com/datasets/dennisho/blackjack-hands