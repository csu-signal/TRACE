# Grid defaults
GRID_SIZE = 3  # this should always be 3 for the demo, essentially it will create a GRID_SIZE ** 2 grid
DEFAULT_REGION_FRAC = 0.35  # Defines the overall size of the grid, the grid will take up (min(frame_height, frame_width) * region_frac) ** 2 space in the center of the frame
REGION_FRAC_INCREMENT = (
    0.01  # How much to increment/decrement the region frac by when pressing W or S
)

# Hue ranges - these are [min, max) (besides red!!), i.e. min value is inclusive and max value is exclusive.
# NOTE: Red is weird because it's at the top of the hue/color wheel. Instead its range is (min, max], and it checks to see if the mean color value is above max and below min.
RED_MIN_HUE = 10
RED_MAX_HUE = 160

ORANGE_MIN_HUE = 10
ORANGE_MAX_HUE = 20

YELLOW_MIN_HUE = 20
YELLOW_MAX_HUE = 35

GREEN_MIN_HUE = 35
GREEN_MAX_HUE = 85

BLUE_MIN_HUE = 85
BLUE_MAX_HUE = 160

# Shape ratios - these are based on the width and height of the bounding boxes for the segmentations. The ratio is calculated as width / height
MIN_SQUARE_RATIO = 0.8
MAX_SQUARE_RATIO = 1.2

# Rectangles need a lower range and an upper range. If height is bigger, then ratio should be close to 0.5. If width is bigger, then ratio should be close to 2
LOWER_MIN_RECTANGLE_RATIO = 0.3
LOWER_MAX_RECTANGLE_RATIO = 0.7
UPPER_MIN_RECTANGLE_RATIO = 1.8
UPPER_MAX_RECTANGLE_RATIO = 2.2

# Cell depths
# NOTE: Use the inspect_grid_depth_ranges.py script to calibrate these values (higher values means closer to the camera, lower depth values mean further away, this is true for Depth Anything V2)
# NOTE: Depth Anything V2 does not use metric values by default to measure depth, so depth values can be negative. Even if you swtich it to use metric, higher values are closer and lower values are further away.
CELL_MIN_DEPTH = 19.5
CELL_MAX_DEPTH = 15

# Config files for calibrating depth
DEPTH_MIN_CONFIG = "grid_cell_min_depths.json"
DEPTH_MAX_CONFIG = "grid_cell_max_depths.json"

# SAM2 Configuration
POINT_PROMPTS_PER_AXIS = 3  # the sqrt of the number of point prompts in the grid that SAM2 automatic mask generator uses
DEFAULT_POINT_PROMPT_GRID_REGION_FRAC = (
    0.7  # How centered to make the point prompts within the central crop of the image
)

# SAM2 Automatic Mask Generator Parameters
SAM2_PREDICTED_IOU_THRESH = 0.5
SAM2_STABILITY_SCORE_THRESH = 0.75

# SAM2 Postprocess Small Regions Parameters
POSTPROCESS_MIN_AREA = 1000
POSTPROCESS_NMS_THRESH = 0.8
