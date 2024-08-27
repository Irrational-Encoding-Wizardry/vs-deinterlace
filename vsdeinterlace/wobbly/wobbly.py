from __future__ import annotations

import json
from dataclasses import dataclass, field
from fractions import Fraction
from math import ceil
from typing import Any, Sequence

from vstools import (MISSING, CustomIndexError, CustomNotImplementedError,
                     CustomTypeError, CustomValueError, FieldBased,
                     FieldBasedT, FileNotExistsError, FuncExceptT,
                     FunctionUtil, Keyframes, SPath, SPathLike, Timecodes,
                     UnsupportedFieldBasedError, VSFunction, core, vs)

from vsdeinterlace.wobbly.info import Preset

from .base import _WobblyProcessBase
from .exceptions import InvalidCycleError, InvalidMatchError
from .info import (CustomList, FreezeFrame, InterlacedFade, OrphanField,
                   Section, VDecParams, VfmParams, WobblyMeta, WobblyVideo,
                   _HoldsFrameNum, _HoldsStartEndFrames)
from .types import CustomPostFiltering, Match, OrphanMatch

__all__: list[str] = [
    "WobblyParsed",
    "parse_wobbly"
]


@dataclass
class WobblyParsed(_WobblyProcessBase):
    """Dataclass representing the contents of a parsed wobbly file."""

    work_clip: WobblyVideo
    """Information about how to handle the work clip used in Wobbly itself."""

    meta: WobblyMeta
    """Meta information about Wobbly."""

    vfm_params: VfmParams
    """An object containing all the vivtc.VFM parameters passed through wibbly."""

    vdecimate_params: VDecParams
    """An object containing all the vivtc.VDecimate parameters passed through wibbly."""

    field_order: FieldBasedT
    """The field order represented as a FieldBased object."""

    matches: Sequence[str] = field(default_factory=list)
    """
    The field matches. See this webpage for more information:
    `<http://underpop.online.fr/f/ffmpeg/help/p_002fc_002fn_002fu_002fb-meaning.htm.gz>`_
    """

    combed_frames: set[int] = field(default_factory=set)
    """A set of combed frames. Frames with interlaced fades will be excluded."""

    orphan_frames: set[OrphanField] = field(default_factory=set)
    """A set of OrphanField objects representing an orphan."""

    decimations: set[int] = field(default_factory=set)
    """A set of frames to decimate."""

    sections: list[Section] = field(default_factory=list)
    """A set of Section objects representing the scenes of a video."""

    interlaced_fades: set[InterlacedFade] = field(default_factory=set)
    """A set of InterlacedFade objects representing frames marked as interlaced fades."""

    freeze_frames: set[FreezeFrame] = field(default_factory=set)
    """A list of FreezeFrame objects representing ranges to freeze, and which frames to replace them with."""

    presets: set[Preset] = field(default_factory=set)
    """A set of Presets used in the wobbly file."""

    custom_lists: list[CustomList] = field(default_factory=list)
    """A set of CustomLists used in the wobbly file."""

    def __init__(self, file_path: SPathLike, func_except: FuncExceptT | None = None) -> None:
        """
        Parse a wobbly file and return a WobblyParsed object.

        :param wobbly_path:     Path to the .wob file Wobbly outputs. If the path does not end
                                with a .wob extension, it will be implicitly added.
        :param func_except:     Function returned for custom error handling.
                                This should only be set by VS package developers.

        :return:                A WobblyParsed object containing the parsed data.
        """

        self._func = func_except or WobblyParsed

        wob_file = self._check_wob_path(file_path)

        with wob_file.open('r') as f:
            self._wob_data = dict[str, Any](json.load(f))

        self.work_clip = WobblyVideo(
            self._get_val("input file"),
            self._get_val("source filter"),
            self._get_val("trim", []),
            Fraction(*self._get_val("input frame rate", [30000, 1001]))
        )

        self.meta = WobblyMeta(
            self._get_val("wobbly version", -1),
            self._get_val("project format version", -1),
            self._get_val("generated with", None)
        )

        self.vfm_params = VfmParams(**dict(self._get_val("vfm parameters", {})))
        self.vdecimate_params = VDecParams(**dict(self._get_val("vdecimate parameters", {})))

        if self.vdecimate_params.cycle != 5:
            raise InvalidCycleError("Wobbly currently only supports a cycle of 5 frames!", self._func)

        self.field_order = FieldBased.from_param(self.vfm_params.order + 1, self._func)

        if not self.field_order.is_inter:
            raise UnsupportedFieldBasedError("Your source may not be PROGRESSIVE!", self._func)

        self.matches = self._get_val("matches", [])
        self.combed_frames = self._get_val("combed frames", set())
        self.decimations = self._get_val("decimated frames", set())

        if not bool(len(illegal_chars := set(self.matches) - {*Match.__args__})):  # type:ignore[attr-defined]
            raise InvalidMatchError(f"Illegal characters found in matches: {tuple(illegal_chars)}", self._func)

        self._set_sections()
        self._set_interlaced_fades()
        self._set_freeze_frames()
        self._set_orphan_frames()
        self._set_presets()
        self._set_custom_lists()

        # Further sanitizing where necessary.
        self._remove_ifades_from_combed()

    def apply(
        self,
        clip: vs.VideoNode | None = None,
        tff: FieldBasedT | None = None,
        orphan_handling: bool | OrphanMatch | Sequence[OrphanMatch] = False,
        orphan_deinterlacing_function: VSFunction = core.resize.Bob,
        func_except: FuncExceptT | None = None
    ) -> vs.VideoNode:
        """
        Apply all the Wobbly processing to a given clip.

        :param clip:                Clip to process. If None, index the `input_file` file as defined
                                    in the original Wobbly file, using the same indexer Wobbly used.
                                    Default: None.
        :param tff:                 Top-Field-First. If None, obtain this from the parsed wobbly file.
                                    If it can't obtain it from that, it will try to obtain it from the VideoNode.
                                    Default: None.
        :param orphan_handling:     Deinterlace orphan fields. This may restore motion that is otherwise lost,
                                    at the cost of heavy speed loss and potential deinterlacing artifacting.
                                    If an OrphanMatch or a sequence of OrphanMatch values is passed instead,
                                    it will only handle orphan fields that were matched to the selected matches.
                                    Default: False.
        :param func_except:         Function returned for custom error handling.
                                    This should only be set by VS package developers.

        :return:                    Clip with wobbly processing applied, and optionally orphan fields deinterlaced.
        """

        func_except = func_except or self._func

        if clip is None:
            clip = self.work_clip.source(func_except)
        else:
            clip = self.work_clip.trim(clip)
            clip = self.work_clip.set_framerate(clip)

        func = FunctionUtil(clip, func_except, None, vs.YUV)

        wclip = FieldBased.from_param(tff or self.field_order, self.apply).apply(func.work_clip)

        orphan_proc = self._get_orphans_to_process(orphan_handling, func.func)

        if self.custom_lists:
            wclip = self._apply_custom_list(wclip, self.custom_lists, CustomPostFiltering.SOURCE)

        if self.matches:
            matches_to_proc = self._force_c_match(self.matches, orphan_proc)  # type:ignore[arg-type]
            wclip = self._apply_fieldmatches(wclip, matches_to_proc, self.field_order, func.func)

        if self.custom_lists:
            wclip = self._apply_custom_list(wclip, self.custom_lists, CustomPostFiltering.FIELD_MATCH)

        if self.freeze_frames:
            wclip = self._apply_freezeframes(wclip, self.freeze_frames, func.func)

        if self.decimations:
            wclip = self._mark_framerates(wclip)

        if self.interlaced_fades:
            wclip = self._apply_interlaced_fades(wclip, self.interlaced_fades, func.func)

        if orphan_proc:
            wclip = self._deinterlace_orphans_mark(wclip, orphan_proc, orphan_deinterlacing_function, func.func)

        if self.combed_frames:
            wclip = self._apply_combed_markers(wclip, self.combed_frames)

        if self.decimations:
            wclip = wclip.std.DeleteFrames(list(self.decimations))

        wclip = FieldBased.PROGRESSIVE.apply(wclip)

        if self.custom_lists:
            wclip = self._apply_custom_list(wclip, self.custom_lists, CustomPostFiltering.DECIMATE)

        return func.return_clip(wclip)

    def remove_sections(self) -> None:
        """Remove all sections from the wobbly data except for the first one."""

        self.sections = [self.sections[0]]

    def remove_presets(self) -> None:
        """Remove all presets from the wobbly data."""

        for section in self.sections:
            section.presets = []

    def _check_wob_path(self, file_path: SPathLike) -> SPath:
        """Check the wob file path and return an SPath object."""

        wob_file = SPath(file_path)

        if wob_file.suffix != '.wob':
            wob_file = wob_file.parent / (wob_file.name + '.wob')

        if not wob_file.exists():
            raise FileNotExistsError(f"Could not find the file, \"{wob_file}\"!", self._func)

        return wob_file

    def _get_val(self, key: str, fallback: Any = MISSING) -> Any:
        """Get a value from the wobbly data dictionary."""

        if (val := self._wob_data.get(key, fallback)) is MISSING:
            raise CustomValueError(f"Could not get the value from \"{key}\" in the wobbly file!", self._func)

        return val

    def _force_c_match(
        self, matches: list[str], frames: list[int | _HoldsFrameNum | _HoldsStartEndFrames]
    ) -> list[Match]:
        """Force a 'c' match for a given list of frames."""

        for frame in frames:
            if isinstance(frame, _HoldsStartEndFrames):
                for i in range(frame.start_frame, frame.end_frame + 1):
                    matches[i] = 'c'

                continue

            if isinstance(frame, _HoldsFrameNum):
                frame = frame.framenum

            matches[frame] = 'c'

        return matches  # type:ignore[return-value]

    def _set_sections(self) -> None:
        """Set the sections attribute."""

        sections_data: list[dict[str, Any]] = self._get_val("sections", [{}])

        self.sections = [
            Section(
                section.get("start", 0),
                sections_data[i + 1].get("start", 0) - 1 if i < len(sections_data) - 1 else len(self.matches) - 1,
                section.get("presets", [])
            )
            for i, section in enumerate(sections_data)
        ]

    def _set_interlaced_fades(self) -> None:
        """Set the interlaced fades attribute."""

        ifades_data: list[dict[str, int | float]] = self._get_val("interlaced fades", [{}])

        self.interlaced_fades = {
            InterlacedFade(
                int(ifade.get("frame", -1)),
                ifade.get("field_difference", 0.0)
            )
            for ifade in ifades_data
        }

    def _set_freeze_frames(self) -> None:
        """Set the freeze frames attribute."""

        self.freeze_frames = {
            FreezeFrame(*tuple(freezes[0]))
            for freezes in zip(self._get_val("frozen frames", ()))
        }

    def _set_orphan_frames(self) -> None:
        """Set the orphan frames attribute."""

        self.orphan_frames = set()

        try:
            for section in self.sections:
                frame_num = section.start_frame

                # TODO: double-check p and u are correct
                if self.matches[frame_num] == 'n':
                    self.orphan_frames.add(OrphanField(frame_num, 'n'))
                elif self.matches[frame_num] == 'p':
                    self.orphan_frames.add(OrphanField(frame_num, 'p'))

                frame_num = section.end_frame

                if self.matches[frame_num] == 'b':
                    self.orphan_frames.add(OrphanField(frame_num, 'b'))
                elif self.matches[frame_num] == 'u':
                    self.orphan_frames.add(OrphanField(frame_num, 'u'))
        except IndexError as e:
            raise CustomIndexError(" ".join([str(e), f"(frame: {frame_num})"]), self._func)

    def _set_presets(self) -> None:
        """Set the presets attribute."""

        preset_data: list[dict[str, str]] = self._get_val("presets", [])

        self.presets = {
            Preset(
                preset.get("name", "fallback name"),
                preset.get("contents", "raise RuntimeError('No contents provided!')"),
            )
            for preset in preset_data
        }

    def _set_custom_lists(self) -> None:
        """Set the custom lists attribute."""

        custom_list_data: list[dict[str, Any]] = self._get_val("custom lists", [])

        self.custom_lists = [
            CustomList(
                custom_list.get("name", "fallback name"),
                custom_list.get("preset", ""),
                custom_list.get("position", ""),
                custom_list.get("frames", []),
                self.presets
            )
            for custom_list in custom_list_data
        ]

    def _remove_ifades_from_combed(self) -> None:
        """Remove interlaced fades from the combed frames attribute."""

        if self.interlaced_fades:
            self.combed_frames = set(self.combed_frames) - set(i.framenum for i in self.interlaced_fades)

    def _get_orphans_to_process(
        self, process: bool | OrphanMatch | Sequence[OrphanMatch] = False,
        func_except: FuncExceptT | None = None
    ) -> Sequence[OrphanField]:
        """
        Get a list of orphan fields to process.

        :param process:         If True, process all orphan fields. If an OrphanMatch or a sequence of OrphanMatch
                                values is passed instead, it will only handle orphan fields that were matched to
                                the selected matches.
                                Default: False.
        :param func_except:     Function returned for custom error handling.
                                This should only be set by VS package developers.

        :return:                A list of OrphanField objects representing the orphan fields to process.
        """

        if not process:
            return []

        func = func_except or self.apply

        orphan_matches = OrphanMatch.__args__  # type:ignore[attr-defined]

        if isinstance(process, bool):
            matches_check = orphan_matches
        elif any(m in ('b', 'n', 'p', 'u') for m in list(process)):
            matches_check = list(process)
        else:
            raise CustomTypeError(
                "Expected type (bool | OrphanMatch | Sequence[OrphanMatch]), "
                f"got {type(process).__name__}!", func,
            )

        if process is orphan_matches:
            return list(self.orphan_frames)

        return [orphan for orphan in self.orphan_frames if orphan.match in matches_check]

    def _mark_framerates(self, clip: vs.VideoNode) -> vs.VideoNode:
        """Mark the framerates per cycle."""

        framerates = [
            self.work_clip.framerate.numerator / self.vdecimate_params.cycle * i
            for i in range(self.vdecimate_params.cycle, 0, -1)
        ]

        fps_clips = [
            clip.std.AssumeFPS(None, int(fps), self.work_clip.framerate.denominator)
            .std.SetFrameProps(
                wobbly_cycle_fps=int(fps // 1000),
                _DurationNum=int(fps),
                _DurationDen=self.work_clip.framerate.denominator
            ) for fps in framerates
        ]

        split_decimations = [
            [
                j for j in range(i * self.vdecimate_params.cycle, (i + 1) * self.vdecimate_params.cycle)
                if j in self.decimations
            ] for i in range(0, ceil((max(self.decimations) + 1) / self.vdecimate_params.cycle))
        ]

        n_split_decimations = len(split_decimations)

        indices = [
            0 if (sd_idx := ceil(n / self.vdecimate_params.cycle)) >= n_split_decimations
            else len(split_decimations[sd_idx]) for n in range(clip.num_frames)
        ]

        return clip.std.FrameEval(lambda n: fps_clips[indices[n]])

    @property
    def section_keyframes(self) -> Keyframes:
        """Return a Keyframes object created using sections."""

        # TODO: Create a keyframes property that takes decimation into account.
        return Keyframes([i.start_frame for i in self.sections])

    # TODO: Calculate the timecodes from the information provided, not a clip.
    @property
    def timecodes(self) -> Timecodes:
        """The timecodes represented as a Timecode object."""

        raise CustomNotImplementedError("Timecodes have not yet been implemented!", self._func)


def parse_wobbly(
    wobbly_file: SPathLike,
    clip: vs.VideoNode | None = None,
    tff: FieldBasedT | None = None,
    orphan_deinterlacing_function: VSFunction = core.resize.Bob,
    orphan_handling: bool | OrphanMatch | Sequence[OrphanMatch] = False,
) -> vs.VideoNode:
    """
    Parse the contents of a Wobbly project file and return a processed clip.

    For more information, see the documentation for the WobblyParsed class.

    :return:    A clip with the Wobbly processing applied.
    """

    wob_file = SPath(wobbly_file)

    return WobblyParsed(wob_file, parse_wobbly).apply(
        clip, tff, orphan_handling, orphan_deinterlacing_function, parse_wobbly
    )