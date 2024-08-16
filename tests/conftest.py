import wave
from pathlib import Path

import pytest


@pytest.fixture
def test_data_dir():
    return Path(__file__).parent / "data"


class Utils:
    @staticmethod
    def get_length(wav_path):
        """
        Helper function to get length of wav file
        """
        wf = wave.open(str(wav_path), "rb")
        length = wf.getnframes() / wf.getframerate()
        wf.close()
        return length

    @staticmethod
    def levenshtein(s1, s2):
        """
        Edit distance using DP
        https://en.wikibooks.org/wiki/Algorithm_Implementation/Strings/Levenshtein_distance#Python
        """
        if len(s1) < len(s2):
            return Utils.levenshtein(s2, s1)

        # len(s1) >= len(s2)
        if len(s2) == 0:
            return len(s1)

        previous_row = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = (
                    previous_row[j + 1] + 1
                )  # j+1 instead of j since previous_row and current_row are one character longer
                deletions = current_row[j] + 1  # than s2
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row

        return previous_row[-1]


@pytest.fixture
def test_utils():
    """
    Utility functions for testing
    """
    return Utils
