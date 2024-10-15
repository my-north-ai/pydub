"""
SpeechSegment is an object implementing specific methods to deal with 
speech audio data.
"""

from numpy import ndarray, ones, zeros, dot, int64, argmin
from numpy import transpose, arange

from uniplot import plot

from .utils import convolve, get_silence
from .audio_segment import AudioSegment

class SpeechSegment(AudioSegment):

    def __init__(self, data=None, *args, **kwargs):

        super().__init__(data, *args, **kwargs)

    def partition(self, max_size):
        """
        Partitions audio segments in moments of silence according to a 
        specific heuristic.
        max_size: int (in bytes)
        """
        
        if self.frame_count() * self.sample_width >= max_size:

            max_length = max_size / self.sample_width
            conv_width = 1 + self.frame_rate // 2
            conv_offset = self.frame_rate // 8
            slack = max_length // 4

            # compute the amplitudes and the convolution
            waveform = self._waveform.mean(axis=1)
            amplitudes = waveform ** 2
            convolved_amplitudes, mapping = convolve(
                amplitudes,
                conv_width,
                conv_offset
            )

            # find the mid-points
            internal_max_length = max_length - slack
            n_partitions = self.frame_count() // internal_max_length
            mid_points = arange(
                internal_max_length,
                (n_partitions + 1) * internal_max_length,
                internal_max_length,
                dtype=int64
            )

            # find a most silent point around the mid-points, this is where the 
            # waveform will be sliced
            cut_points = zeros(len(mid_points) + 2, dtype=int64)
            cut_points[0] = 0
            cut_points[-1] = self.frame_count() - 1
            for i in range(len(mid_points)):
                cut_points[i + 1] = get_silence(
                    convolved_amplitudes,
                    mapping,
                    mid_points[i] - slack,
                    mid_points[i] + slack
                )

            # partition the waveform
            for i in range(len(cut_points) - 1):
                waveform = self._waveform[cut_points[i]: cut_points[i + 1]]
                segment = self._spawn(
                    data = waveform,
                    overrides={"channels": 1}
                )
                yield segment
        else:
            yield self
