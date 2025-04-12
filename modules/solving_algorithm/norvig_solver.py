import time
from typing import Dict, List, Tuple, Union


class NorvigSolver:
    """
    A constraint-propagation and depth-first search based Sudoku solver
    inspired by Peter Norvig's elegant algorithm.
    """

    def __init__(self) -> None:
        self.digits: str = '123456789'
        self.rows: str = 'ABCDEFGHI'
        self.cols: str = self.digits
        self.squares: List[str] = self.cross(self.rows, self.cols)

        self.unitlist: List[List[str]] = (
            [self.cross(self.rows, c) for c in self.cols] +
            [self.cross(r, self.cols) for r in self.rows] +
            [self.cross(rs, cs) for rs in ('ABC', 'DEF', 'GHI') for cs in ('123', '456', '789')]
        )

        self.units: Dict[str, List[List[str]]] = {
            s: [u for u in self.unitlist if s in u] for s in self.squares
        }

        self.peers: Dict[str, set[str]] = {
            s: set(sum(self.units[s], [])) - {s} for s in self.squares
        }

    @staticmethod
    def cross(A: str, B: str) -> List[str]:
        """Cross product of elements in string A and string B."""
        return [a + b for a in A for b in B]

    def grid_values(self, grid: str) -> Dict[str, str]:
        """
        Convert grid string into a dictionary of {square: char}, with '0' or '.' as placeholders.

        Args:
            grid (str): A string representation of the Sudoku puzzle.

        Returns:
            Dict[str, str]: Mapping from each square to its value (or '.'/0 for empty).
        """
        chars: List[str] = [c for c in grid if c in self.digits or c in '0.']
        assert len(chars) == 81, "Input grid must be 81 characters long"
        return dict(zip(self.squares, chars))

    def parse_grid(self, grid: str) -> Union[Dict[str, str], bool]:
        """
        Parse a grid into a dict of possible values, {square: digits}, or return False if a contradiction is found.

        Args:
            grid (str): The Sudoku grid string.

        Returns:
            Union[Dict[str, str], bool]: Dictionary of possible values per cell, or False if invalid.
        """
        values: Dict[str, str] = {s: self.digits for s in self.squares}
        for s, d in self.grid_values(grid).items():
            if d in self.digits:
                if not self.assign(values, s, d):
                    return False
        return values

    def assign(self, values: Dict[str, str], s: str, d: str) -> Union[Dict[str, str], bool]:
        """
        Eliminate all other values (except d) from values[s] and propagate.

        Args:
            values (Dict[str, str]): Current possible values for all squares.
            s (str): Square to update.
            d (str): Digit to assign.

        Returns:
            Union[Dict[str, str], bool]: Updated values or False if contradiction occurs.
        """
        other_values = values[s].replace(d, '')
        if all(self.eliminate(values, s, d2) for d2 in other_values):
            return values
        return False

    def eliminate(self, values: Dict[str, str], s: str, d: str) -> Union[Dict[str, str], bool]:
        """
        Eliminate digit d from values[s]; propagate when values or places <= 2.

        Args:
            values (Dict[str, str]): Current state of the puzzle.
            s (str): Square to update.
            d (str): Digit to eliminate.

        Returns:
            Union[Dict[str, str], bool]: Updated values or False if contradiction occurs.
        """
        if d not in values[s]:
            return values  # Already eliminated

        values[s] = values[s].replace(d, '')

        if len(values[s]) == 0:
            return False  # Contradiction: removed last value
        elif len(values[s]) == 1:
            # If only one value left, eliminate it from peers
            d2 = values[s]
            if not all(self.eliminate(values, s2, d2) for s2 in self.peers[s]):
                return False

        # Check for units where d can only go in one place
        for u in self.units[s]:
            dplaces = [sq for sq in u if d in values[sq]]
            if len(dplaces) == 0:
                return False  # Contradiction
            elif len(dplaces) == 1:
                if not self.assign(values, dplaces[0], d):
                    return False
        return values

    def display(self, values: Dict[str, str]) -> None:
        """
        Display the Sudoku grid in a readable format.

        Args:
            values (Dict[str, str]): Current values for each square.
        """
        width = 1 + max(len(values[s]) for s in self.squares)
        line = '+'.join(['-' * (width * 3)] * 3)
        for r in self.rows:
            print(''.join(values[r + c].center(width) + ('|' if c in '36' else '') for c in self.cols))
            if r in 'CF':
                print(line)
        print()

    def search(self, values: Union[Dict[str, str], bool]) -> Union[Dict[str, str], bool]:
        """
        Using depth-first search and propagation, try all possible values.

        Args:
            values (Dict[str, str] or bool): Current grid state.

        Returns:
            Union[Dict[str, str], bool]: Final solved values or False if unsolvable.
        """
        if values is False:
            return False
        if all(len(values[s]) == 1 for s in self.squares):
            return values  # Solved

        # Choose square with the fewest possibilities
        _, s = min((len(values[s]), s) for s in self.squares if len(values[s]) > 1)
        for d in values[s]:
            result = self.search(self.assign(values.copy(), s, d))
            if result:
                return result
        return False

    def solve(self, grid: str) -> Union[Dict[str, str], bool]:
        """
        Solve a Sudoku puzzle.

        Args:
            grid (str): 81-character Sudoku string.

        Returns:
            Union[Dict[str, str], bool]: Solved values or False.
        """
        return self.search(self.parse_grid(grid))

    def solved(self, values: Union[Dict[str, str], bool]) -> bool:
        """
        Check if the puzzle is completely and correctly solved.

        Args:
            values (Dict[str, str] or bool): Current state of the puzzle.

        Returns:
            bool: True if solved correctly, else False.
        """
        def unitsolved(unit: List[str]) -> bool:
            return set(values[s] for s in unit) == set(self.digits)

        return values is not False and all(unitsolved(unit) for unit in self.unitlist)

    def solve_all(self, grids: List[str], name: str = '') -> None:
        """
        Solve multiple Sudoku puzzles and report timing.

        Args:
            grids (List[str]): List of puzzle strings.
            name (str): Optional name for this batch.
        """
        times, results = zip(*[self.time_solve(grid) for grid in grids])
        N = len(results)
        if N > 1:
            print(f"Solved {sum(results)} of {N} {name} puzzles (avg {sum(times)/N:.2f} secs, max {max(times):.2f} secs).")

    def time_solve(self, grid: str) -> Tuple[float, bool]:
        """
        Time how long it takes to solve a single puzzle.

        Args:
            grid (str): Puzzle string.

        Returns:
            Tuple[float, bool]: Time taken and success flag.
        """
        start = time.time()
        values = self.solve(grid)
        t = time.time() - start
        return t, self.solved(values)

    def test(self) -> None:
        """
        Run internal tests to verify the board setup and constraints.
        """
        assert len(self.squares) == 81
        assert len(self.unitlist) == 27
        assert all(len(self.units[s]) == 3 for s in self.squares)
        assert all(len(self.peers[s]) == 20 for s in self.squares)
        print('All tests pass.')
