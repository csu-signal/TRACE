# ========== Grid Configuration Defaults ==========

GRID_SIZE = 3  # this should always be 3 for the demo, essentially it will create a GRID_SIZE ** 2 grid
DEFAULT_REGION_FRAC = 0.35  # Defines the overall size of the grid, the grid will take up (min(frame_height, frame_width) * region_frac) ** 2 space in the center of the frame
REGION_FRAC_INCREMENT = (
    0.01  # How much to increment/decrement the region frac by when pressing W or S
)

# ========== Color Detection ==========

# Hue ranges - these are [min, max) (besides red!!), i.e. min value is inclusive and max value is exclusive.
# NOTE: Red is weird because it's at the top of the hue/color wheel. Instead its range is (min, max], and it checks to see if the mean color value is above max and below min.
RED_MIN_HUE = 10
RED_MAX_HUE = 160

ORANGE_MIN_HUE = 10
ORANGE_MAX_HUE = 20

YELLOW_MIN_HUE = 20
YELLOW_MAX_HUE = 35

GREEN_MIN_HUE = 35
GREEN_MAX_HUE = 95

BLUE_MIN_HUE = 95
BLUE_MAX_HUE = 160

WHITE_BASEBOARD_SATURATION_THRESH = 75

# ========== Shape Detection ==========

# Shape ratios - these are based on the width and height of the bounding boxes for the segmentations. The ratio is calculated as max(width, height) / min(width, height)
MIN_SQUARE_RATIO = 0.8
MAX_SQUARE_RATIO = 1.2

MIN_RECTANGLE_RATIO = 1.8
MAX_RECTANGLE_RATIO = 2.2

# ========== Mask Filtering ==========

# Fraction for determining the threshold that will ignore masks that aren't  X% of a grid cell's area
MASK_SIZE_THRESH_FRAC = 0.6

# Fraction for determining the threshold that ignore masks that don't take up X% of a particular grid cell's area. This is different, albeit nuanced, than MASK_SIZE_THRESH_FRAC, which just makes sure that a mask's size is greater than a percentage of a grid cell. This ensures that a mask actually covers X% of a particular grid cell.
CELL_AREA_INTERSECTION_THRESH_FRAC = 0.4

# ========== Depth Stuff ==========

# Cell depths
# NOTE: Use the inspect_grid_depth_ranges.py script to calibrate these values (higher values means closer to the camera, lower depth values mean further away, this is true for Depth Anything V2)
# NOTE: Depth Anything V2 does not use metric values by default to measure depth, so depth values can be negative. Even if you swtich it to use metric, higher values are closer and lower values are further away.
CELL_MIN_DEPTH = 19.5
CELL_MAX_DEPTH = 15

# Config files for calibrating depth
DEPTH_MIN_CONFIG = "grid_cell_min_depths.json"
DEPTH_MAX_CONFIG = "grid_cell_max_depths.json"

# ========== SAM2 Configuration ==========

# The number of rows and columns to place in the point prompt grid
POINT_PROMPTS_PER_AXIS = 3
# How centered to make the point prompts within the central crop of the image
DEFAULT_POINT_PROMPT_GRID_REGION_FRAC = 0.55


# Automatic Mask Generator Parameters
SAM2_PREDICTED_IOU_THRESH = 0.5
SAM2_STABILITY_SCORE_THRESH = 0.75

# Postprocess Small Regions Parameters
POSTPROCESS_MIN_AREA = 1000
POSTPROCESS_NMS_THRESH = 0.9

SAM2_CHECKPOINT_PATH = "C:\\Users\\jack\\Code\\sam2\\checkpoints\\sam2.1_hiera_large.pt"
SAM2_MODEL_CONFIG_PATH = "configs/sam2.1/sam2.1_hiera_l.yaml"
