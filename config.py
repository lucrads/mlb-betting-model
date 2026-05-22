import os

# The Odds API key — set via env var or --odds-key CLI flag
ODDS_API_KEY = os.environ.get("ODDS_API_KEY", "")

# Simulation settings
NUM_SIMULATIONS = 1000
STARTER_INNINGS_LIMIT = 6        # Switch to bullpen after this many innings
STARTER_BATTERS_FACED_LIMIT = 27 # Or after this many TBF

# Edge thresholds
BET_EDGE_THRESHOLD = 0.05   # 5%+ edge → BET
LEAN_EDGE_THRESHOLD = 0.02  # 2%+ edge → LEAN

# Base running probabilities
BASE_RUNNING = {
    "single_runner_1b_to_3b_prob": 0.30,   # prob runner on 1B takes extra base on single
    "single_runner_2b_scores_prob": 0.75,  # prob runner on 2B scores on single
    "double_runner_1b_scores_prob": 0.30,  # prob runner on 1B scores on double
    "flyout_tag_3b_prob": 0.85,            # prob runner on 3B tags and scores on flyout (<2 outs)
    "groundout_runner_3b_scores_prob": 0.85, # prob runner on 3B scores on groundout (<2 outs)
}

# Pitch type groupings
FASTBALL_TYPES = {"FF", "SI", "FC"}
BREAKING_TYPES = {"SL", "ST", "CU", "KC"}
OFFSPEED_TYPES = {"CH", "FS"}

# Season for stats
CURRENT_SEASON = 2026

# FIP constant for the pitcher's own FIP-to-wOBA conversion
# Maps a pitcher's own computed FIP to an estimated wOBA allowed:
#   woba_allowed = FIP_WOBA_INTERCEPT + fip * FIP_WOBA_SLOPE
# At FIP 3.00 → ~0.296, FIP 4.10 → ~0.320, FIP 5.00 → ~0.340
FIP_WOBA_INTERCEPT = 0.230
FIP_WOBA_SLOPE = 0.022

# League-average calibration constants for contact-quality signals
LEAGUE_AVG_BARREL_RATE  = 0.065   # ~6.5% of BBE are barrels (EV ≥ 98 mph, LA 26–30°)
LEAGUE_AVG_MEAN_LA      = 12.0    # degrees — avg launch angle on batted balls
LEAGUE_AVG_SPRINT_SPEED = 27.0    # ft/s — Baseball Savant sprint-speed leaderboard avg

# Park HR factors relative to league average (1.0 = neutral).
# Approximate 3-year rolling values; update each off-season.
PARK_HR_FACTORS: dict[str, float] = {
    "Colorado Rockies":        1.25,   # Coors Field — altitude
    "Cincinnati Reds":         1.18,   # Great American Ball Park
    "New York Yankees":        1.15,   # short right-field porch
    "Texas Rangers":           1.12,   # Globe Life Field
    "Atlanta Braves":          1.10,   # Truist Park
    "Philadelphia Phillies":   1.08,   # Citizens Bank Park
    "Houston Astros":          1.07,   # Minute Maid Park
    "Milwaukee Brewers":       1.06,   # American Family Field
    "Chicago Cubs":            1.05,   # Wrigley Field (wind-dependent; avg)
    "Baltimore Orioles":       1.04,   # Camden Yards
    "Boston Red Sox":          1.02,   # Fenway Park (wall helps 2B, neutral HR)
    "Toronto Blue Jays":       1.01,   # Rogers Centre
    "Los Angeles Dodgers":     1.00,
    "Minnesota Twins":         0.99,   # Target Field
    "Detroit Tigers":          0.98,   # Comerica Park
    "Pittsburgh Pirates":      0.97,   # PNC Park
    "Kansas City Royals":      0.97,   # Kauffman Stadium
    "Cleveland Guardians":     0.96,   # Progressive Field
    "New York Mets":           0.96,   # Citi Field
    "Seattle Mariners":        0.95,   # T-Mobile Park
    "Chicago White Sox":       0.95,   # Guaranteed Rate Field
    "San Diego Padres":        0.94,   # Petco Park
    "St. Louis Cardinals":     0.93,   # Busch Stadium
    "Washington Nationals":    0.92,   # Nationals Park
    "Arizona Diamondbacks":    0.91,   # Chase Field
    "Miami Marlins":           0.90,   # loanDepot park
    "Los Angeles Angels":      0.90,   # Angel Stadium
    "Athletics":               0.89,   # current nomadic/new venue
    "Oakland Athletics":       0.89,
    "Tampa Bay Rays":          0.88,   # Tropicana Field (dome, neutral air)
    "San Francisco Giants":    0.86,   # Oracle Park — marine layer suppresses HR
}
