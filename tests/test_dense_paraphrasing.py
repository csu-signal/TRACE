from testing_profiles import TestDenseParaphrasingProfile
from demo.featureModules.move.move_classifier import rec_common_ground

def test_dense_paraphrasing():
    TestDenseParaphrasingProfile("demo_outputs/dense_paraphrasing").run()
