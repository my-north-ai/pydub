"""
SpeechSegment is an object implementing specific methods to deal with
speech audio data.
"""

from math import ceil

from numpy import ndarray, ones, zeros, dot, int64, argmin
from numpy import transpose, arange

from .utils import convolve, get_silence
from .audio_segment import AudioSegment

class SpeechSegment(AudioSegment):

    def __init__(self, data=None, *args, **kwargs):

        super().__init__(data, *args, **kwargs)

    def naive_partition(self, max_size):
        """
        Partition the audios in chuncks of max_size.
        max_size: int (in bytes)
        """

        max_frames = max_size // self.frame_width
        n_partitions = (int(self.frame_count()) // max_frames) + 1
        internal_max_length = int(self.frame_count() / n_partitions) + 1
        for i in range(n_partitions):
            i *= internal_max_length
            waveform = self._waveform[i : i + internal_max_length]
            segment = self._spawn(data = waveform)
            yield segment

    def partition(self, max_size, format=None):
        """
        Partitions audio segments in moments of silence according to a
        specific heuristic.
        max_size: int (in bytes)
        """

        if format is None:
            n_partitions = ceil(len(self.raw_data) / max_size)
        else:
            size = len(self.export(format=format).read())
            n_partitions = ceil(size / max_size)

        if n_partitions > 1:
            max_length = ceil(self.frame_count() / n_partitions)
            conv_width = 1 + self.frame_rate // 2
            conv_offset = self.frame_rate // 8
            slack = max_length // 8

            # compute the amplitudes and the convolution
            waveform = self._waveform.mean(axis=1)
            amplitudes = waveform ** 2
            convolved_amplitudes, mapping = convolve(
                amplitudes,
                conv_width,
                conv_offset
            )

            # find the mid-points
            internal_length = max_length - 2 * slack
            n_partitions = ceil(self.frame_count() / internal_length)
            mid_points = [i * internal_length for i in range(1, n_partitions)]

            # find a most silent point around the mid-points, this is where the
            # waveform will be sliced
            cut_points = [None] * (n_partitions + 1)
            cut_points[0] = 0
            cut_points[-1] = int(self.frame_count() - 1)
            for i, mid_point in enumerate(mid_points):
                cut_points[i + 1] = int(
                    get_silence(
                        convolved_amplitudes,
                        mapping,
                        mid_point - slack,
                        mid_point + slack
                    )
                )
            # partition the waveform
            for i in range(len(cut_points) - 1):
                waveform = self._waveform[cut_points[i]: cut_points[i + 1]]
                segment = self._spawn(data = waveform)
                yield segment
        else:
            yield self
