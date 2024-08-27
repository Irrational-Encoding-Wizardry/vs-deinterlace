from __future__ import annotations

from typing import Any, Sequence, cast

from vsexprtools import ExprVars, complexpr_available, norm_expr
from vspyplugin import PyPluginCython
from vsrgtools import BlurMatrix
from vstools import (
    ConvMode, CustomEnum, FuncExceptT, FunctionUtil, GenericVSFunction, KwargsT, PlanesT, core, depth, scale_8bit, vs
)

__all__ = [
    'fix_interlaced_fades',
    'vinverse',
    'equilines'
]


class FixInterlacedFades(CustomEnum):
    Average: FixInterlacedFades = object()  # type: ignore
    Darken: FixInterlacedFades = object()  # type: ignore
    Brighten: FixInterlacedFades = object()  # type: ignore

    def __call__(
        self, clip: vs.VideoNode, colors: float | list[float] | PlanesT = 0.0,
        planes: PlanesT = None, func: FuncExceptT | None = None
    ) -> vs.VideoNode:
        """
        Give a mathematically perfect solution to decombing fades made *after* telecine
        (which made perfect IVTC impossible) that start or end in a solid color.

        Steps between the frames are not adjusted, so they will remain uneven depending on the telecine pattern,
        but the decombing is blur-free, ensuring minimum information loss. However, this may cause small amounts
        of combing to remain due to error amplification, especially near the solid-color end of the fade.

        This is an improved version of the Fix-Telecined-Fades plugin.

        Make sure to run this *after* IVTC!

        :param clip:                            Clip to process.
        :param colors:                          Fade source/target color (floating-point plane averages).

        :return:                                Clip with fades to/from `colors` accurately deinterlaced.
                                                Frames that don't contain such fades may be damaged.
        """
        func = func or self.__class__

        if not complexpr_available:
            raise ExprVars._get_akarin_err()(func=func)

        f = FunctionUtil(clip, func, planes, None, 32)

        fields = f.work_clip.std.Limiter().std.SeparateFields(tff=True)

        for i in f.norm_planes:
            fields = fields.std.PlaneStats(None, i, f'P{i}')

        props_clip = core.akarin.PropExpr(
            [f.work_clip, fields[::2], fields[1::2]], lambda: {  # type: ignore[misc]
                f'f{t}Avg{i}': f'{c}.P{i}Average {color} -'  # type: ignore[has-type]
                for t, c in ['ty', 'bz']
                for i, color in zip(f.norm_planes, f.norm_seq(colors))
            }
        )

        expr_header = 'Y 2 % x.fbAvg{i} x.ftAvg{i} ? AVG! AVG@ 0 = x x {color} - '
        expr_footer = ' AVG@ / * ? {color} +'

        expr_mode, expr_mode_chroma = (
            ('+ 2 /', '+ 2 /') if self == self.Average else (('min', '<') if self == self.Darken else ('max', '>'))
        )

        fix = norm_expr(
            props_clip, (
                # luma
                expr_header + 'x.ftAvg{i} x.fbAvg{i} {expr_mode}' + expr_footer,
                # chroma
                expr_header + 'x.ftAvg{i} abs x.fbAvg{i} abs {expr_mode} x.ftAvg{i} x.fbAvg{i} ?' + expr_footer
            ),
            planes, i=f.norm_planes, expr_mode=(expr_mode, expr_mode_chroma),
            color=colors, force_akarin=func,
        )

        return f.return_clip(fix)


def vinverse(
    clip: vs.VideoNode,
    comb_blur: GenericVSFunction | Sequence[int] = [1, 2, 1],
    contra_blur: GenericVSFunction | Sequence[int] = [1, 4, 6, 4, 1],
    contra_str: float = 2.7, amnt: int = 255, scl: float = 0.25, planes: PlanesT = None,
    **kwargs: Any
) -> vs.VideoNode:
    """
    A simple but effective plugin to remove residual combing. Based on an AviSynth script by Didée.

    :param clip:            Clip to process.
    :param comb_blur:       Filter used to remove combing.
    :param contra_blur:     Filter used to calculate contra sharpening.
    :param contra_str:      Strength of contra sharpening.
    :param amnt:            Change no pixel by more than this in 8bit (default is 255, unrestricted).
    :param scl:             Scale factor for vshrpD*vblurD < 0.
    """

    func = FunctionUtil(clip, vinverse, planes, vs.YUV, 32)

    def_k = KwargsT(mode=ConvMode.VERTICAL)

    kwrg_a, kwrg_b = not callable(comb_blur), not callable(contra_blur)

    if not callable(comb_blur):
        comb_blur = BlurMatrix(comb_blur)

    if not callable(contra_blur):
        contra_blur = BlurMatrix(contra_blur)

    blurred = comb_blur(func.work_clip, planes=planes, **((def_k | kwargs) if kwrg_a else kwargs))
    blurred2 = contra_blur(blurred, planes=planes, **((def_k | kwargs) if kwrg_b else kwargs))

    combed = norm_expr(
        [func.work_clip, blurred, blurred2],
        'y z - {sstr} * D1! x y - D2! D1@ abs D1A! D2@ abs D2A! '
        'D1@ D2@ xor D1A@ D2A@ < D1@ D2@ ? {scl} * D1A@ D2A@ < D1@ D2@ ? ? y + '
        'LIM! x {amnt} + LIM@ < x {amnt} + x {amnt} - LIM@ > x {amnt} - LIM@ ? ?',
        planes, sstr=contra_str, amnt=scale_8bit(func.work_clip, amnt), scl=scl,
    )

    return func.return_clip(combed)


fix_interlaced_fades = cast(FixInterlacedFades, FixInterlacedFades.Average)


EquilinesFilter = PyPluginCython[None]


def equilines(clip: vs.VideoNode, thr: int = 65535) -> vs.VideoNode:
    @EquilinesFilter(depth(clip, 16), cython_kernel=('equilines', 'equilines_filter'))
    def out(self: EquilinesFilter, src: EquilinesFilter.DT, dst: EquilinesFilter.DT) -> None:
        self.kernel.equilines_filter(src, dst, thr)

    return depth(out, clip)
