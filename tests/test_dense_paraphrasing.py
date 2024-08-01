import os
from pathlib import Path
import shutil
import pandas as pd
from testing_profiles import TestDenseParaphrasingProfile
from demo.featureModules import DenseParaphrasingFeature

def test_dense_paraphrasing():
    # assert not os.path.exists("demo_outputs/dense_paraphrasing"), "Output will overwrite existing files"
    output_dir = Path(__file__).parent / "dense_paraphrasing_out"

    TestDenseParaphrasingProfile(output_dir).run()

    df = pd.read_csv(output_dir / DenseParaphrasingFeature.LOG_FILE)

