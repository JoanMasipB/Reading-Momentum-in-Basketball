# Basketball Momentum Analysis

## Overview
This project analyzes momentum in FIBA basketball games using advanced data analysis techniques. The analysis focuses on identifying and understanding momentum periods during games, utilizing both traditional statistical approaches and machine learning models.

## Data Description
The dataset contains play-by-play data from FIBA basketball games with the following key columns:
* `game_id`: Unique identifier for each game
* `numberofplay`: Sequential play number within each game
* `playtype`: Type of basketball play (TO, ST, FTM, FTA, 2FGM, 2FGA, 3FGM, 3FGA, O, D, RV)
* `quarter`: Game quarter (1-4)
* `points_a`, `points_b`: Score progression for both teams
* `plus_minus`: Point differential
* `seconds`: Time remaining in the quarter (600 to 0)
* `possession`: Team possession (1 or 2)

## Methodology

### Momentum Definition
Momentum is identified using several criteria:
1. Score differential changes over time
2. Minimum point differential requirement (≥ 5 points)
3. Time window consideration (45-360 seconds)
4. Continuous performance tracking

### Analysis Approaches
The project implements three different analytical methods:
1. Traditional Statistical Analysis
    - Play-by-play momentum detection
    - Quarter-by-quarter analysis
    - Team performance metrics

2. Machine Learning with XGBoost
    - Sequential feature engineering
    - Binary classification model
    - Precision-focused evaluation

3. Transformer-Based Deep Learning
    - Sequence modeling
    - Attention mechanisms
    - Complex pattern recognition

## Implementation

### Prerequisites
```python
import polars as pl
import numpy as np
import tensorflow as tf
from xgboost import XGBClassifier
from sklearn.metrics import precision_score, recall_score
```

### Key Features
* Quarter-specific analysis
* Team performance tracking
* Shooting efficiency during momentum periods
* Momentum duration analysis

## Results

### Momentum Characteristics
* Average duration of momentum periods
* Distribution across quarters
* Impact on game outcomes
* Team shooting percentages during momentum

### Model Performance
* Precision and recall metrics
* Classification accuracy
* Feature importance analysis

## Future Improvements
1. Enhanced feature engineering
2. Alternative momentum definitions
3. Real-time prediction capabilities
4. Integration of team-specific statistics

## Usage
```pyt
