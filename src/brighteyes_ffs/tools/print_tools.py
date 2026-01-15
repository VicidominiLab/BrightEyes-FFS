# -*- coding: utf-8 -*-
"""
Created on Fri Nov 21 09:18:22 2025

@author: eslenders
"""

def print_table(table, width=8, max_cols=9):
    """
    Print 2D numeric table in a nice tabular way
    with row/column labels and column truncation.

    Parameters
    ----------
    table : np.array
        2D array.
    width : int, optional
        Width of each cell. The default is 8.
    max_cols : int, optional
        Maximum number of data columns to display.
    """

    n_rows, n_cols = table.shape
    show_cols = min(n_cols, max_cols)
    truncated = n_cols > max_cols

    # Header
    header = [""] + [f"curve {i+1}" for i in range(show_cols)]
    if truncated:
        header.append("...")

    print("".join(f"{h:>{width}}" for h in header))

    # Rows
    for i, row in enumerate(table):
        row_label = f"param {i+1}"
        print(f"{row_label:>{width}}", end="")

        for v in row[:show_cols]:
            print(f"{v:{width}.2f}", end="")

        if truncated:
            print(f"{'...':>{width}}", end="")

        print()

