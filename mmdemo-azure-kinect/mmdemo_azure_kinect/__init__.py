import os

import azure_kinect_config as config

os.add_dll_directory(str(config.K4A_DLL_DIR))
