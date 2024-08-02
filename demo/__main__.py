import os

from demo.config import K4A_DIR, WTD_END_TIMES

os.add_dll_directory(K4A_DIR)
import cv2 as cv

from demo.base_profile import BaseProfile
from demo.profiles import (BradyLaptopProfile, LiveProfile, RecordedProfile,
                           create_recorded_profile, create_wtd_eval_profiles)

if __name__ == "__main__":
    # live_prof = LiveProfile([
    #     ("Videep", 2),
    #     ("Austin", 6),
    #     ("Mariah", 15)
    #     ])

    # live_prof = LiveProfile([
    #     ("Group", 6),
    #     ])

    groups = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    profiles: list[BaseProfile] = []
    for group in groups:
        profiles += create_wtd_eval_profiles(
            group,
            "wtd_inputs",
            "wtd_outputs",
            end_time=WTD_END_TIMES[group],
            configs=("no_gt",),
        )

    for prof in profiles:
        prof.run()
