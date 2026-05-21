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

# League average wOBA by pitch type (baseline estimates)
LEAGUE_AVG_WOBA_BY_PITCH = {
    "FF": 0.330,  # 4-seam fastball
    "SI": 0.320,  # sinker
    "FC": 0.315,  # cutter
    "SL": 0.295,  # slider
    "ST": 0.290,  # sweeper
    "CU": 0.285,  # curveball
    "KC": 0.280,  # knuckle-curve
    "CH": 0.300,  # changeup
    "FS": 0.295,  # splitter
    "OTHER": 0.310,
}

# Pitch type groupings for normalization
FASTBALL_TYPES = {"FF", "SI", "FC"}
BREAKING_TYPES = {"SL", "ST", "CU", "KC"}
OFFSPEED_TYPES = {"CH", "FS"}

# Season for stats
CURRENT_SEASON = 2026
