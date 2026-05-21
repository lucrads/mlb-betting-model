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
