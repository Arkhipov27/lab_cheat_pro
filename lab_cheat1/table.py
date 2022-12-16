import tkinter as tk
from string import printable as english_symbols
from typing import List, Tuple

from .var import Var, GroupVar, digital_normalize_tuple


def rus_tex_formula(formula: str) -> str:
    """
    This method turns tex-formula into russian language

    :param formula: tex-formula
    :return: russian interpretation
    """
    ret, rus_now = '', False
    for char in formula:
        if rus_now is (char in english_symbols):
            ret += {True: '}', False: '\\text{'}[rus_now]
            rus_now = not rus_now
        ret += char
    if rus_now is True: ret += '}'
    return ret


def to_table(Excel: str, transpose=False) -> List[List[float]]:
    """
    This method translates excel-file to table

    :param Excel: a string consisting from array of data
    :param transpose: parameter to transpose the table to list of tuples

    :return: list of values from excel-file
    """
    table = tuple(map(lambda x: x.split('\t'), Excel.split('\n')))[:-1]
    if transpose: table = list(zip(*table))
    return [[float(table[j][i]) for j in range(len(table)) if table[j][i] != ''] for i in range(len(table[0]))]


class TexTable:
    """
    This class is made to build and show tex-tables from array of data
    """
    def __init__(self):
        self._numbers: List[Tuple[str, ...], ...] = []
        self._titles: List[str, ...] = []

    def add(self, group_var: GroupVar, title: str, show_err=False):
        """
        This method adds values and errors to the table

        :param group_var: the variable that we want to add to the table
        :param title: title of columns
        :param show_err: parameter to show errors in the table

        :return: the table
        """
        values, errors = group_var.val_err()
        self._numbers.append(tuple(str(digital_normalize_tuple(Var(values[i], errors[i]))[0])
                                   for i in range(len(group_var))))
        self._titles.append(title)
        if show_err is True:
            self._numbers.append(tuple(str(digital_normalize_tuple(Var(values[i], errors[i]))[1])
                                       for i in range(len(group_var))))
            self._titles.append('\\Delta ' + title)
        return self

    def show(self, numerate: bool = True, colours=('C0C0C0', 'EFEFEF', 'C0C0C0'), table=False, floatrow=False,
             color_frequency: int = 2):
        """
        This method shows the table in tkinter-window. To copy the content hold Ctrl+A and Ctrl+C.

        :param numerate: to numerate strings in the table
        :param colours: colours that are displayed in the table
        :param table: to set table-environment in the table. Default is center-environment
        :param floatrow: to accommodate some tables in the row
        :param color_frequency: frequency of the colours

        :return: the table in tkinter-window
        """
        if numerate:
            self._numerating()
        if table:
            self._show_tk_window(self._begin_table(floatrow) + self._write_titles(colours) +
                                 self._write_numbers(numerate, colours, color_frequency) + self._end_table(floatrow))
        else:
            self._show_tk_window(self._begin_center(floatrow) + self._write_titles(colours) +
                                 self._write_numbers(numerate, colours, color_frequency) + self._end_center(floatrow))

    def _numerating(self):
        self._titles = [' '] + self._titles
        max_num = max(map(len, self._numbers))
        self._numbers = [tuple(str(i) for i in range(1, max_num+1))] + self._numbers

    def _begin_center(self, floatrow: bool = False):
        if floatrow:
            return "\\begin{center} \\Topfloatboxes \n" + \
                   "\\textbf{Таблица } \\\\ \n" + \
                   "\\begin{floatrow} \n" + \
                   "\\begin{tabular}{|" + "".join(['c|'] * len(self._titles)) + "}\n"
        return "\\begin{center} \n" + \
               "\\textbf{Таблица } \\\\ \n" + \
               "\\begin{tabular}{|"+"".join(['c|'] * len(self._titles)) + "}\n"

    def _begin_table(self, floatrow: bool = False):
        if floatrow:
            return "\\begin{table} \\Topfloatboxes \n" + \
                   "\\caption{Таблица } \n" + \
                   "\\begin{floatrow} \n" + \
                   "\\begin{tabular}{|"+"".join(['c|'] * len(self._titles)) + "}\n"
        return "\\begin{table} \n" + \
               "\\caption{Таблица } \n" + \
               "\\begin{tabular}{|" + "".join(['c|'] * len(self._titles)) + "}\n"

    def _write_titles(self, colours):
        return "\\hline\n" + \
               "\\rowcolor[HTML]{" + colours[0] + "}\n" + \
               "".join([' $' + rus_tex_formula(title) + '$ &' for title in self._titles])[:-1] + "\\\\ \\hline\n"

    def _write_numbers(self, numerate, colours, color_frequency):
        result = ''
        for string in range(max(map(len, self._numbers))):
            result += ("\\rowcolor[HTML]{" + colours[1] + "}\n" if (string + 1) % color_frequency == 0 else '') + \
                      ("\\cellcolor[HTML]{" + colours[2] + "} " if numerate else '')
            for column in range(len(self._titles)):
                result += _safe_get(self._numbers[column], string, ' ') + ' & '
            result = result[:-2] + '\\\\ \\hline\n'
        return result

    @staticmethod
    def _end_center(floatrow: bool = False):
        if floatrow:
            return "\\end{tabular}\n" + \
                   "\\end{floatrow}\n" + \
                   "\end{center}\n"
        return "\\end{tabular}\n" + \
               "\end{center}\n"

    @staticmethod
    def _end_table(floatrow: bool = False):
        if floatrow:
            return "\\end{tabular}\n" + \
                   "\\end{floatrow}\n" + \
                   "\end{center}\n"
        return "\\end{tabular}\n" + \
               "\end{center}\n"

    @staticmethod
    def _show_tk_window(text):
        root = tk.Tk()
        text_tk = tk.Text(width=100, height=30, wrap=tk.WORD)
        text_tk.insert(float(0), text)
        text_tk.pack(expand=tk.YES, fill=tk.BOTH)
        # ctrl+A does not mean selecting all automatically, that's why i make it by myself

        def select_all(event):
            event.widget.tag_add(tk.SEL, '1.0', tk.END)
            return 'break'
        text_tk.bind('<Control-a>', select_all)
        root.mainloop()


def _safe_get(lst: Tuple, i: int, default):
    return lst[i] if i < len(lst) else default
