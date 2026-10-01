"""Per-player variable playback-speed arrangements (time warping).

A schedule divides one clip cycle into segments, each played at a relative
rate (e.g. slow the stance phase, hurry the swing phase), and can
additionally stretch the whole cycle to a requested target duration.
Schedules live on the player (not the shared clip resource) so different
characters can warp the same clip differently, and they are part of the
saved playback state.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class SpeedSegment:
    """Clip-local interval ``(previous_end, end]`` played at ``rate``."""

    end: float
    rate: float


class SpeedSchedule:
    """Piecewise-constant playback rate over one clip cycle.

    ``segments`` are ``(end, rate)`` pairs in clip-local time, sorted and
    covering ``[0, clip.duration]``; for looping clips the arrangement
    repeats every cycle. ``target_duration`` optionally rescales the whole
    arrangement so one full cycle takes exactly that many wall-clock
    seconds at playback speed 1.
    """

    def __init__(self, segments, target_duration=None):
        pairs = [
            (float(end), float(rate))
            for end, rate in (
                (s.end, s.rate) if isinstance(s, SpeedSegment) else s
                for s in segments
            )
        ]
        if not pairs:
            raise ValueError("speed schedule needs at least one segment")
        pairs.sort(key=lambda pair: pair[0])
        previous = 0.0
        warped = 0.0
        for end, rate in pairs:
            if end <= previous:
                raise ValueError("speed segments must be increasing")
            if rate <= 0.0:
                raise ValueError("segment rates must be positive")
            warped += (end - previous) / rate
            previous = end
        if target_duration is not None and target_duration <= 0.0:
            raise ValueError("target_duration must be positive")
        self.segments = tuple(
            SpeedSegment(end, rate) for end, rate in pairs
        )
        self.target_duration = target_duration
        # Wall-clock length of one cycle at speed 1 without target scaling.
        self.warped_duration = warped

    @property
    def duration(self) -> float:
        return self.segments[-1].end

    def validate_for(self, clip):
        if abs(self.segments[-1].end - clip.duration) > 1e-9:
            raise ValueError(
                f"schedule ends at {self.segments[-1].end} but clip "
                f"{clip.name!r} has duration {clip.duration}"
            )

    def time_scale(self) -> float:
        """Extra rate factor needed to hit the target cycle duration."""
        if self.target_duration is None:
            return 1.0
        return self.warped_duration / self.target_duration

    def segment_index(self, phase: float) -> int:
        for index, segment in enumerate(self.segments):
            if phase < segment.end:
                return index
        return len(self.segments) - 1

    def rate_at(self, phase: float) -> float:
        return self.segments[self.segment_index(phase)].rate

    def segment_start(self, index: int) -> float:
        return 0.0 if index == 0 else self.segments[index - 1].end

    def to_dict(self) -> dict:
        return {
            "segments": [[s.end, s.rate] for s in self.segments],
            "target_duration": self.target_duration,
        }

    @staticmethod
    def from_dict(data):
        if data is None:
            return None
        return SpeedSchedule(data["segments"], data.get("target_duration"))
