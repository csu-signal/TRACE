import pytest

from mmdemo.features.utterance.vad_builder_feature import VADUtteranceBuilder
from mmdemo.interfaces import AudioFileInterface
from tests.utils.audio import get_length


@pytest.fixture
def activity_path(test_data_dir):
    """
    Path to wav file with activity
    """
    return test_data_dir / "activity.wav"


@pytest.fixture
def no_activity_path(test_data_dir):
    """
    Path to wav file with no activity
    """
    return test_data_dir / "no_activity.wav"


@pytest.fixture
def chunk_length(activity_path, no_activity_path):
    """
    Length of activity/no activity wav file
    """
    activity_length = get_length(activity_path)
    no_activity_length = get_length(no_activity_path)
    assert (
        activity_length == no_activity_length
    ), "activity and no activity files must be the same length"
    return activity_length


@pytest.fixture
def vad_builder(chunk_length):
    vad_builder = VADUtteranceBuilder(
        delete_input_files=False, max_utterance_time=chunk_length * 5
    )
    vad_builder.initialize()
    yield vad_builder
    vad_builder.finalize()


single_speaker = [
    (["01101100110"], [[4, 4, 4]]),
    (["11110100"], [[5, 3]]),
    (["00001000001110"], [[3, 5]]),
]
multiple_speakers = [
    (["0110", "110110", "000"], [[4], [3, 4], []]),
    (["010110", "0111010", "01010101"], [[3, 4], [5, 3], [3, 3, 3]]),
    (["1010"] * 10, [[2, 3]] * 10),
]
max_length = [
    (["1111110"], [[5, 3]]),
    (
        ["011111111110", "11111110", "0011110010"],
        [[5, 5, 4], [5, 4], [5, 3]],
    ),
    (["001111", "0111100"], [[5], [5]]),
]


@pytest.mark.parametrize(
    "inputs,expected", single_speaker + multiple_speakers + max_length
)
def test_vad_segmentation(
    vad_builder,
    activity_path,
    no_activity_path,
    chunk_length,
    inputs: list[str],
    expected: list[list[int]],
):
    """
    Test that the segmentation works correctly for both single and multiple
    speakers.

    `input` -- a list of string where 0 represents an audio clip with no voice
               activity and 1 represents an audio clip with voice activity.
               The different strings in the list correspond to different
               speakers who speak simultaneously.
    `output_lengths` -- a list of lists of expected lengths of audio clips
                        contained inside the segmented utterances. Different
                        lists correspond to different speakers.
    """
    feature_outputs: list[AudioFileInterface] = []

    for chunk_index in range(max(len(i) for i in inputs)):
        input_interfaces: list[AudioFileInterface] = []
        for speaker_index in range(len(inputs)):
            # skip if there are no more chunks for that speaker
            if len(inputs[speaker_index]) <= chunk_index:
                continue

            kwargs = {
                "speaker_id": f"{speaker_index}",
                "start_time": chunk_index * chunk_length,
                "end_time": chunk_index * chunk_length + chunk_length,
            }
            activity_interface = AudioFileInterface(**kwargs, path=activity_path)
            no_activity_interface = AudioFileInterface(**kwargs, path=no_activity_path)

            if inputs[speaker_index][chunk_index] == "0":
                input_interfaces.append(no_activity_interface)
            else:
                input_interfaces.append(activity_interface)

        # get all current output from the feature
        output = vad_builder.get_output(*input_interfaces)
        while output is not None:
            feature_outputs.append(output)
            output = vad_builder.get_output()

    output_chunk_counts = [[] for _ in range(len(inputs))]
    for output_audio in feature_outputs:
        # check that the file length is the same as `end - start` time
        length = output_audio.end_time - output_audio.start_time
        actual_length = get_length(output_audio.path)
        assert get_length(output_audio.path) == pytest.approx(
            length
        ), f"AudioFileInterface length is {length} but the actual file length is {actual_length}"

        # save the number of chunks
        output_chunk_counts[int(output_audio.speaker_id)].append(
            round(length / chunk_length)
        )

    # make sure the number of chunks are equal
    assert (
        output_chunk_counts == expected
    ), "Ouptut utterance chunks are not equal to the expected utterance chunks"
