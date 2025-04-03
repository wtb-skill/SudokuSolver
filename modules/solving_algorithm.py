import time
import numpy as np

class NorvigSolver:
    def __init__(self):
        self.digits = '123456789'
        self.rows = 'ABCDEFGHI'
        self.cols = self.digits
        self.squares = self.cross(self.rows, self.cols)
        self.unitlist = ([self.cross(self.rows, c) for c in self.cols] +
                         [self.cross(r, self.cols) for r in self.rows] +
                         [self.cross(rs, cs) for rs in ('ABC', 'DEF', 'GHI') for cs in ('123', '456', '789')])
        self.units = {s: [u for u in self.unitlist if s in u] for s in self.squares}
        self.peers = {s: set(sum(self.units[s], [])) - {s} for s in self.squares}

    def cross(self, A, B):
        return [a + b for a in A for b in B]

    def parse_grid(self, grid):
        values = {s: self.digits for s in self.squares}
        for s, d in self.grid_values(grid).items():
            if d in self.digits and not self.assign(values, s, d):
                return False
        return values

    def grid_values(self, grid):
        chars = [c for c in grid if c in self.digits or c in '0.']
        assert len(chars) == 81
        return dict(zip(self.squares, chars))

    def assign(self, values, s, d):
        other_values = values[s].replace(d, '')
        if all(self.eliminate(values, s, d2) for d2 in other_values):
            return values
        return False

    def eliminate(self, values, s, d):
        if d not in values[s]:
            return values
        values[s] = values[s].replace(d, '')
        if len(values[s]) == 0:
            return False
        elif len(values[s]) == 1:
            d2 = values[s]
            if not all(self.eliminate(values, s2, d2) for s2 in self.peers[s]):
                return False
        for u in self.units[s]:
            dplaces = [s for s in u if d in values[s]]
            if len(dplaces) == 0:
                return False
            elif len(dplaces) == 1:
                if not self.assign(values, dplaces[0], d):
                    return False
        return values

    def display(self, values):
        width = 1 + max(len(values[s]) for s in self.squares)
        line = '+'.join(['-' * (width * 3)] * 3)
        for r in self.rows:
            print(''.join(values[r + c].center(width) + ('|' if c in '36' else '') for c in self.cols))
            if r in 'CF':
                print(line)
        print()

    def solve(self, grid):
        return self.search(self.parse_grid(grid))

    def search(self, values):
        if values is False:
            return False
        if all(len(values[s]) == 1 for s in self.squares):
            return values
        n, s = min((len(values[s]), s) for s in self.squares if len(values[s]) > 1)
        for d in values[s]:
            result = self.search(self.assign(values.copy(), s, d))
            if result:
                return result

    def solved(self, values):
        def unitsolved(unit):
            return set(values[s] for s in unit) == set(self.digits)

        return values is not False and all(unitsolved(unit) for unit in self.unitlist)

    def solve_all(self, grids, name=''):
        times, results = zip(*[self.time_solve(grid) for grid in grids])
        N = len(results)
        if N > 1:
            print(
                f"Solved {sum(results)} of {N} {name} puzzles (avg {sum(times) / N:.2f} secs, max {max(times):.2f} secs).")

    def time_solve(self, grid):
        start = time.time()
        values = self.solve(grid)
        t = time.time() - start
        return (t, self.solved(values))

    def test(self):
        assert len(self.squares) == 81
        assert len(self.unitlist) == 27
        assert all(len(self.units[s]) == 3 for s in self.squares)
        assert all(len(self.peers[s]) == 20 for s in self.squares)
        print('All tests pass.')


class SudokuConverter:
    @staticmethod
    def board_to_string(digit_board: np.ndarray) -> str:
        """
        Converts a 9x9 numpy array (digit_board) into a Sudoku grid string.

        Parameters:
            digit_board (np.ndarray): A 9x9 numpy array representing the Sudoku board.

        Returns:
            str: A string of 81 characters representing the Sudoku grid (0 for empty cells).
        """
        return ''.join(str(digit_board[row, col]) for row in range(9) for col in range(9))

    @staticmethod
    def dict_to_board(sudoku_dict):
        """
        Converts a Sudoku solution dictionary to a 9x9 numpy array.

        Parameters:
            sudoku_dict (dict): A dictionary with square keys (e.g., 'A1', 'B2') and digit values ('1' to '9').

        Returns:
            np.ndarray: A 9x9 numpy array representing the Sudoku grid.
        """
        # Create an empty 9x9 numpy array to store the result
        board = np.zeros((9, 9), dtype=int)

        # Loop through each row (A to I) and each column (1 to 9)
        for row_idx, row in enumerate('ABCDEFGHI'):
            for col_idx, col in enumerate('123456789'):
                square = row + col  # Create the key like 'A1', 'B2', ...
                board[row_idx, col_idx] = int(sudoku_dict[square])  # Assign the value from the dictionary

        return board


if __name__ == '__main__':
    solver = NorvigSolver()
    solver.test()
    grid1 = '003020600900305001001806400008102900700000008006708200002609500800203009005010300'
    solved_grid = solver.solve(grid1)
    solver.display(solved_grid)
