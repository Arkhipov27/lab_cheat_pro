from __future__ import annotations

from functools import reduce
from typing import Optional, Tuple, Union, Callable, SupportsFloat, Sequence

import matplotlib.patches as _mp
import matplotlib.pyplot as plt
from numpy import array, linspace, sqrt

from .var import Var, GroupVar


class Figure:
    """
    This class is made to show graphs
    """

    def __init__(self, x_label: str = '', y_label: str = '', bold_axes: bool = True, zero_in_corner: bool = True,
                 label_near_arrow: bool = True, my_func: Optional[Callable] = None,
                 x_label_coords: Sequence[SupportsFloat] = array([1.03, -0.03]),
                 y_label_coords: Sequence[SupportsFloat] = array([-0.06, 1]),
                 legend_props: Optional[dict] = None):
        """
        :param x_label: label near xt axis
        :param y_label: label near y_exp axis
        :param bold_axes: need of y_exp=0 and xt=0 lines
        :param zero_in_corner: need of showing (0, 0) wherever the other dots
        :param label_near_arrow: if True: labels will be shown in the corners near arrows of axis.
                                 if False: labels will be near centers of appropriate axis.
        :param my_func: function that takes an object of the matplotlib.axes._subplots.AxesSubplot class as an argument.
        It will be called before fixing axes and adding lines. You may use it to add anything you want.
        :param x_label_coords: coordinates of placing x_label.
        It's easy to shift x_label in this way: figure.x_label_coords+=array([0.03, -0.04])
        :param y_label_coords: similar to x_label_coords
        :param legend_props: dictionary for
        """
        self.x_label, self.y_label = x_label, y_label
        self.x_label_coords, self.y_label_coords = x_label_coords, y_label_coords
        self.bold_axes = bold_axes
        self.zero_in_corner = zero_in_corner
        self.label_near_arrow = label_near_arrow
        self.my_func = my_func
        self.legend_props: dict = legend_props if legend_props is not None else {}
        self.alpha = 0.25  # the transparency of v_lines and h_lines
        # params for 'plot' method
        self._scatters_kwargs = []
        self._errorbars_kwargs = []
        # params for 'line' method
        self._lines_params = []
        self._v_lines_params = []
        self._h_lines_params = []
        self._func_graphs_before_fixing_axes = []
        self._func_graphs_after_fixing_axes = []

    def line(self, k: Union[float, int, Var], b: Union[float, int, Var], colour: Optional[str] = None,
             line_style: Optional[str] = None, label: Optional[str] = None) -> Figure:
        """
        Draws a straight

        :param k: coefficient of inclination of the line
        :param b: the second coefficient of the line
        :param colour: colour of the line
        :param line_style: style of the line
        :param label: the inscription that will reflect in the legend

        :return: straight
        """
        if isinstance(k, Var):
            k = k.val()
        if isinstance(b, Var):
            b = b.val()
        self._lines_params.append((k, b, colour, line_style, label))
        return self

    def v_line(self, x: Union[float, int, Var], colour: Optional[str] = None, line_style: Optional[str] = None,
               label: Optional[str] = None) -> Figure:
        """
        Draws a vertical straight

        :param x: array of points
        :param colour: colour of the line
        :param line_style: style of the line
        :param label: the inscription that will reflect in the legend

        :return: vertical straight
        """
        self._v_lines_params.append((x, colour, line_style, label))
        return self

    def h_line(self, y: Union[float, int, Var], colour: Optional[str] = None, line_style: Optional[str] = None,
               label: Optional[str] = None) -> Figure:
        """
        Draws a horizontal straight

        :param y: array of points
        :param colour: colour of the line
        :param line_style: style of the line
        :param label: the inscription that will reflect in the legend

        :return: horizontal straight
        """
        self._h_lines_params.append((y, colour, line_style, label))
        return self

    def func_graph(self, func: Callable[[array], array], x_min: Union[int, float, Var], x_max: Union[int, float, Var],
                   N: int = 1000, line_style: Optional[str] = None, colour: Optional[str] = None,
                   label: Optional[str] = None, add_before_fixing_axes: bool = True) -> Figure:
        """
        Draws the graph of the function

        :param func: the function we want to graph
        :param x_min: the minimum of x
        :param x_max: the maximum of x
        :param N: amount of dots on the graph
        :param line_style: style of line
        :param colour: colour of line
        :param label: the inscription that will reflect in the legend
        :param add_before_fixing_axes: a value

        :return: the graph of the function
        """
        # todo: doc and ;сделать возможность проводить линию до края графика по оси xt влево или вправо, если указано None;
        if isinstance(x_min, Var):
            x_min = x_min.val()
        if isinstance(x_max, Var):
            x_max = x_max.val()
        x = linspace(x_min, x_max, N)
        y = func(x)
        ({False: self._func_graphs_before_fixing_axes, True: self._func_graphs_after_fixing_axes}
         )[add_before_fixing_axes].append((x, y, line_style, colour, label))
        return self

    def plot(self, x: Union[GroupVar, Sequence], y: Union[GroupVar, Sequence],
             capsize=3, s=1, colour=None, marker=None, label=None) -> Figure:
        """
        Draws the graph of dots with errors

        :param x: array of dots on x-axis
        :param y: array of dots on y-axis
        :param capsize: a size for error bar
        :param s: square of the dots
        :param colour: colour of the line
        :param marker: display of dots on the graph
        :param label: the inscription that will reflect in the legend

        :return: the graph
        """
        def val_err(t):
            if isinstance(t, GroupVar):
                return t.val_err()
            nonlocal capsize
            capsize = 0
            return array(t), array([0] * len(t))

        x_val, x_err = val_err(x)
        y_val, y_err = val_err(y)
        self._scatters_kwargs.append(
            dict(x=x_val, y=y_val, s=s, c=colour, marker=marker, label=label))
        self._errorbars_kwargs.append(
            dict(x=x_val, y=y_val, xerr=x_err, yerr=y_err, capsize=capsize, capthick=1, fmt='none', c=colour))
        return self

    def show(self):
        """
        Generates matplotlib.Figure and shows it.

        :return: None
        """
        axes = plt.figure().add_subplot()
        self._grid_lines(axes)
        self._show_plots(axes)
        self._show_func_graphs_before_fixing_axes(axes)
        # maybe user wants to do something by himself
        if self.my_func is not None:
            self.my_func(axes)
        xy_limits = self._fix_axes(axes)
        self._v_lines(axes, *xy_limits)
        self._h_lines(axes, *xy_limits)
        self._set_label(axes)
        self._arrows(axes)
        if self.bold_axes is True:
            self._bold_axes(axes, *xy_limits)
        self._show_lines(axes, self.legend_props, *xy_limits)
        self._show_func_graphs_after_fixing_axes(axes)
        plt.show()

    @staticmethod
    def _grid_lines(axes):
        axes.grid(axis='both', which='major', linestyle='--', linewidth=1)
        axes.grid(axis='both', which='minor', linestyle='--', linewidth=0.5)
        axes.minorticks_on()

    def _show_plots(self, axes):
        for scatter_kwargs, errorbar_kwargs in zip(self._scatters_kwargs, self._errorbars_kwargs):
            axes.scatter(**scatter_kwargs)
            axes.errorbar(**errorbar_kwargs)

    def _show_func_graphs_before_fixing_axes(self, axes):
        for x, y, line_style, colour, label in self._func_graphs_before_fixing_axes:
            axes.plot(x, y, color=colour, linestyle=line_style, label=label)

    def _fix_axes(self, axes):
        x_min, x_max = axes.get_xlim()
        y_min, y_max = axes.get_ylim()
        if self.zero_in_corner is True:
            x_min = min(0, x_min)
            y_min = min(0, y_min)
        axes.set_xlim(x_min, x_max)
        axes.set_ylim(y_min, y_max)
        return x_min, x_max, y_min, y_max

    def _v_lines(self, axes, y_min, y_max):
        for x, colour, line_style, label in self._v_lines_params:
            if isinstance(x, Var):
                x_val, x_err = x.val_err()
                plot, = axes.plot((x_val, x_val), (y_min, y_max), color=colour, linestyle=line_style, label=label)
                axes.plot((x_val - x_err, x_val - x_err), (y_min, y_max), color=plot.get_color(), linestyle=':')
                axes.plot((x_val + x_err, x_val + x_err), (y_min, y_max), color=plot.get_color(), linestyle=':')
                axes.fill_between((x_val - x_err, x_val + x_err), (y_min, y_min), (y_max, y_max),
                                  color=plot.get_color(), alpha=self.alpha)
            else:
                axes.plot((x, x), (y_min, y_max), color=colour, linestyle=line_style, label=label)

    def _h_lines(self, axes, x_min, x_max):
        for y, colour, line_style, label in self._h_lines_params:
            if isinstance(y, Var):
                y_val, y_err = y.val_err()
                plot, = axes.plot((x_min, x_max), (y_val, y_val), color=colour, linestyle=line_style, label=label)
                axes.plot((x_min, x_max), (y_val - y_err, y_val - y_err), color=plot.get_color(), linestyle=':')
                axes.plot((x_min, x_max), (y_val + y_err, y_val + y_err), color=plot.get_color(), linestyle=':')
                axes.fill_between((x_min, x_max), (y_val - y_err, y_val - y_err), (y_val + y_err, y_val + y_err),
                                  color=plot.get_color(), alpha=self.alpha)
            else:
                axes.plot((x_min, x_max), (y, y), color=colour, linestyle=line_style, label=label)

    def _set_label(self, axes):
        for set_label, axis, label, label_coords in ((axes.set_xlabel, axes.xaxis, self.x_label, self.x_label_coords),
                                                     (axes.set_ylabel, axes.yaxis, self.y_label, self.y_label_coords)):
            if label.rstrip() != '':
                label_prop = {True: dict(rotation=0), False: {}}[self.label_near_arrow]
                set_label('$' + label + '$', label_prop)
                if self.label_near_arrow is True:
                    axis.set_label_coords(*label_coords)

    def _arrows(self, axes):
        arrowprops = dict(arrowstyle=_mp.ArrowStyle.CurveB(head_length=1), color='black')
        axes.annotate('', xy=(1.05, 0), xycoords='axes fraction', xytext=(-0.03, 0), arrowprops=arrowprops)
        axes.annotate('', xy=(0, 1.06), xycoords='axes fraction', xytext=(0, -0.03), arrowprops=arrowprops)

    def _bold_axes(self, axes, x_min, x_max, y_min, y_max):
        axes.hlines(0, x_min, x_max, linewidth=1, colors='black')
        axes.vlines(0, y_min, y_max, linewidth=1, colors='black')

    def _show_lines(self, axes, legend_props, x_min, x_max, y_min, y_max):
        for k, b, c, ls, label in self._lines_params:
            points = []
            if y_min <= k * x_min + b <= y_max:
                points.append((x_min, k * x_min + b))
            if y_min <= k * x_max + b <= y_max:
                points.append((x_max, k * x_max + b))
            if len(points) < 2 and x_min < (y_max - b) / k < x_max:
                points.append(((y_max - b) / k, y_max))
            if len(points) < 2 and x_min < (y_min - b) / k < x_max:
                points.append(((y_min - b) / k, y_min))
            axes.plot((points[0][0], points[1][0]), (points[0][1], points[1][1]), c=c, ls=ls, label=label)
        if len(axes.get_legend_handles_labels()[1]) != 0:
            plt.legend(**legend_props)

    def _show_func_graphs_after_fixing_axes(self, axes):
        for x, y, line_style, colour, label in self._func_graphs_after_fixing_axes:
            axes.plot(x, y, color=colour, linestyle=line_style, label=label)


def mnk(x: Union[GroupVar, Sequence], y: Union[GroupVar, Sequence], figure: Optional[Figure] = None,
        colour: Optional[str] = None,
        line_style: Optional[str] = None, label: Optional[str] = None) -> Tuple[Var, Var]:
    """
    This method counts two types of errors: those caused by errors and those caused by statistics.
    If the points lie well on the line, then the error due to errors will prevail.
    If the points are measured extremely accurately, but they lie badly on a straight line,
    then the statistical error prevails.
    The resulting error is the root sum of the squares of the two types of these errors.

    :param x: the sequence of points on x-axis
    :param y: the sequence of points on y-axis
    :param figure: the spot where the fpath will display
    :param colour: colour of the line
    :param line_style: style of the line
    :param label: the inscription that will reflect in the legend

    :return: coefficients of the straight
    """
    # TODO: учитывать точки с весом обратным квадрату ошибки
    if len(x) != len(y):
        raise TypeError('"x" and "y" must be the same length')
    if len(x) == 0 or len(x) == 1 or len(x) == 2:
        raise ValueError('The number of points must be at least 3')

    if not isinstance(x, GroupVar):
        x = GroupVar(x, 0)
    if not isinstance(y, GroupVar):
        y = GroupVar(y, 0)
    x_sum: Var = reduce(lambda res, x_var: res + x_var, x)
    y_sum: Var = reduce(lambda res, y_var: res + y_var, y)
    k_ex: Var = (len(x) * reduce(lambda res, i: res + x[i] * y[i], range(len(x)), 0) - x_sum * y_sum) / \
                (len(x) * reduce(lambda res, x_var: res + x_var * x_var, x, 0) - x_sum * x_sum)
    b_ex: Var = (y_sum - k_ex * x_sum) / len(x)
    k_stat_err, b_stat_err = _find_stat_errors(array(x.val()), array(y.val()), k_ex.val(), b_ex.val())
    k, b = [Var(val, sqrt(stat_err ** 2 + exact_err ** 2)) for val, exact_err, stat_err in
            [(*k_ex.val_err(), k_stat_err),
             (*b_ex.val_err(), b_stat_err)]]
    if figure is not None:
        figure.line(k.val(), b.val(), colour=colour, line_style=line_style, label=label)
    return k, b


def mnk_through0(x: GroupVar, y: GroupVar, figure: Optional[Figure] = None, colour: Optional[str] = None,
                 line_style: Optional[str] = None, label: Optional[str] = None) -> Var:
    """
    This method builds mnk through the point (0, 0)

    :param x: the sequence of points on x-axis
    :param y: the sequence of points on y-axis
    :param figure: the spot where the fpath will display
    :param colour: colour of the line
    :param line_style: style of the line
    :param label: the inscription that will reflect in the legend

    :return: coefficient of the straight
    """
    # todo: добавить статистическую ошибку
    if len(x) != len(y):
        raise TypeError('"x" and "y" must be the same length')
    k: Var = reduce(lambda res, i: res + x[i] * y[i], range(len(x)), 0) / \
             reduce(lambda res, x_var: res + x_var * x_var, x[1:], x[0] * x[0])
    if figure is not None:
        figure.line(k.val(), 0, colour=colour, line_style=line_style, label=label)
    return k


def _find_stat_errors(x: array, y: array, k, b):
    """
    This method calculates statistical error of the graph

    :param x: the sequence of points on x-axis
    :param y: the sequence of points on y-axis
    :param k: the coefficient of inclination of the straight
    :param b: the second coefficient

    :return: statistical error of two coefficients
    """
    Sy = sum((y - b - k * x) ** 2) / (len(x) - 2)
    D = len(x) * sum(x ** 2) - (sum(x)) ** 2
    # returns dk, db
    return sqrt(Sy * len(x) / D), sqrt(Sy * sum(x ** 2) / D)
