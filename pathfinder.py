import pygame
from queue import PriorityQueue
from typing import List, Tuple, Callable, Dict
import random



# Define constants for colors
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 255, 0)
YELLOW = (255, 255, 0)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
PURPLE = (128, 0, 128)
ORANGE = (255, 165, 0)
GREY = (128, 128, 128)
TURQUOISE = (64, 224, 208)

# Initialize Pygame and set up the window
WIDTH = 800
WIN = pygame.display.set_mode((WIDTH, WIDTH))
pygame.display.set_caption("A* Path Finding Algorithm")

# Add this line for font initialization
pygame.font.init()
FONT = pygame.font.Font(None, 36)

# Define the Spot class representing each grid cell
class Spot:
    def __init__(self, row: int, col: int, width: int, total_rows: int) -> None:
        """
        Initialize a Spot object.

        Parameters:
        - row: Row index of the spot.
        - col: Column index of the spot.
        - width: Width of the spot.
        - total_rows: Total number of rows in the grid.
        """
        self.row = row
        self.col = col
        self.x = row * width
        self.y = col * width
        self.color = WHITE
        self.neighbors = []
        self.width = width
        self.total_rows = total_rows

    def get_pos(self) -> Tuple[int, int]:
        """Return the position (row, col) of the spot."""
        return self.row, self.col

    # Methods for checking the state of the spot
    def is_closed(self) -> bool:
        """Check if the spot is in the closed set."""
        return self.color == RED

    def is_open(self) -> bool:
        """Check if the spot is in the open set."""
        return self.color == GREEN

    def is_barrier(self) -> bool:
        """Check if the spot is a barrier."""
        return self.color == BLACK

    def is_start(self) -> bool:
        """Check if the spot is the start node."""
        return self.color == ORANGE

    def is_end(self) -> bool:
        """Check if the spot is the end node."""
        return self.color == TURQUOISE

    def reset(self) -> None:
        """Reset the spot to the default state (white color)."""
        self.color = WHITE

    # Methods for setting the state of the spot
    def make_start(self) -> None:
        """Set the spot as the start node."""
        self.color = ORANGE

    def make_closed(self) -> None:
        """Set the spot as part of the closed set."""
        self.color = RED

    def make_open(self) -> None:
        """Set the spot as part of the open set."""
        self.color = GREEN

    def make_barrier(self) -> None:
        """Set the spot as a barrier."""
        self.color = BLACK

    def make_end(self) -> None:
        """Set the spot as the end node."""
        self.color = TURQUOISE

    def make_path(self) -> None:
        """Set the spot as part of the final path."""
        self.color = PURPLE

    def draw(self, win: pygame.Surface) -> None:
        """Draw the spot on the window."""
        pygame.draw.rect(win, self.color, (self.x, self.y, self.width, self.width))

    def update_neighbors(self, grid: List[List['Spot']]) -> None:
        """Update the list of neighboring spots."""
        self.neighbors = []
        if self.row < self.total_rows - 1 and not grid[self.row + 1][self.col].is_barrier():  # DOWN
            self.neighbors.append(grid[self.row + 1][self.col])

        if self.row > 0 and not grid[self.row - 1][self.col].is_barrier():  # UP
            self.neighbors.append(grid[self.row - 1][self.col])

        if self.col < self.total_rows - 1 and not grid[self.row][self.col + 1].is_barrier():  # RIGHT
            self.neighbors.append(grid[self.row][self.col + 1])

        if self.col > 0 and not grid[self.row][self.col - 1].is_barrier():  # LEFT
            self.neighbors.append(grid[self.row][self.col - 1])

    def __lt__(self, other: 'Spot') -> bool:
        """Comparison method used for priority queue ordering."""
        return False


def h(p1: Tuple[int, int], p2: Tuple[int, int]) -> int:
    """
    Heuristic function (Manhattan distance) to estimate the cost from one point to another.

    Parameters:
    - p1: Tuple (x1, y1) representing the coordinates of the first point.
    - p2: Tuple (x2, y2) representing the coordinates of the second point.

    Returns:
    - The estimated cost (heuristic value).
    """
    x1, y1 = p1
    x2, y2 = p2
    return abs(x1 - x2) + abs(y1 - y2)

def reconstruct_path(came_from: Dict[Spot, Spot], current: Spot, draw: Callable) -> int:
    """
    Reconstruct the path from the end node to the start node and return the length of the path.

    Parameters:
    - came_from: Dictionary containing the mapping of each node to its predecessor.
    - current: The current node (end node).
    - draw: Function to redraw the grid.

    Returns:
    - The length of the path.
    """
    path_length = 0
    while current in came_from:
        current = came_from[current]
        current.make_path()
        draw()
        path_length += 1
    return path_length


def algorithm(draw: Callable, grid: List[List[Spot]], start: Spot, end: Spot) -> Tuple[bool, int]:
    """
    A* pathfinding algorithm.

    Parameters:
    - draw: Function to redraw the grid.
    - grid: 2D list representing the grid of spots.
    - start: Start node.
    - end: End node.

    Returns:
    - A tuple containing a boolean indicating whether a path is found and the length of the path.
    """
    count = 0
    open_set = PriorityQueue()
    open_set.put((0, count, start))
    came_from = {}
    g_score = {spot: float("inf") for row in grid for spot in row}
    g_score[start] = 0
    f_score = {spot: float("inf") for row in grid for spot in row}
    f_score[start] = h(start.get_pos(), end.get_pos())
    open_set_hash = {start}

    while not open_set.empty():
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()

        current = open_set.get()[2]
        open_set_hash.remove(current)

        if current == end:
            path_length = reconstruct_path(came_from, end, draw)
            end.make_end()
            return True, path_length

        for neighbor in current.neighbors:
            temp_g_score = g_score[current] + 1

            if temp_g_score < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = temp_g_score
                f_score[neighbor] = temp_g_score + h(neighbor.get_pos(), end.get_pos())
                if neighbor not in open_set_hash:
                    count += 1
                    open_set.put((f_score[neighbor], count, neighbor))
                    open_set_hash.add(neighbor)
                    neighbor.make_open()

        draw()

        if current != start:
            current.make_closed()

    return False, 0

def make_grid(rows: int, width: int) -> List[List[Spot]]:
    """
    Create a 2D grid of Spot objects.

    Parameters:
    - rows: Number of rows in the grid.
    - width: Width of the grid.

    Returns:
    - 2D list representing the grid.
    """
    grid = []
    gap = width // rows
    for i in range(rows):
        grid.append([])
        for j in range(rows):
            spot = Spot(i, j, gap, rows)
            grid[i].append(spot)

    return grid

def make_grid_with_maze(rows: int, width: int) -> List[List[Spot]]:
    """
    Create a 2D grid of Spot objects with a random maze layout.

    Parameters:
    - rows: Number of rows in the grid.
    - width: Width of the grid.

    Returns:
    - 2D list representing the grid with a maze layout.
    """
    grid = []
    gap = width // rows

    # Create a grid with barriers (walls)
    for i in range(rows):
        grid.append([])
        for j in range(rows):
            spot = Spot(i, j, gap, rows)
            if random.random() < 0.3:  # Adjust the probability to control maze density
                spot.make_barrier()
            grid[i].append(spot)

    return grid


def make_grid_warehouse(rows: int, width: int) -> List[List[Spot]]:
    """
    Create a 2D grid of Spot objects with a warehouse-like maze layout.

    Parameters:
    - rows: Number of rows in the grid.
    - width: Width of the grid.

    Returns:
    - 2D list representing the grid with a warehouse-like maze layout.
    """
    grid = []
    gap = width // rows

    # Create a grid with barriers in a structured pattern
    for i in range(rows):
        grid.append([])
        for j in range(rows):
            spot = Spot(i, j, gap, rows)
            spot.make_barrier()
            grid[i].append(spot)

    # Create open paths
    for i in range(0, rows , 5):
        for j in range(rows):
            grid[i][j].reset()
            grid[j][i].reset()

    return grid


def draw_grid(win: pygame.Surface, rows: int, width: int) -> None:
    """
    Draw grid lines on the window.

    Parameters:
    - win: Pygame window.
    - rows: Number of rows in the grid.
    - width: Width of the grid.
    """
    gap = width // rows
    for i in range(rows):
        pygame.draw.line(win, GREY, (0, i * gap), (width, i * gap))
        for j in range(rows):
            pygame.draw.line(win, GREY, (j * gap, 0), (j * gap, width))


def draw(win: pygame.Surface, grid: List[List[Spot]], rows: int, width: int) -> None:
    """
    Draw the entire grid on the window.

    Parameters:
    - win: Pygame window.
    - grid: 2D list representing the grid.
    - rows: Number of rows in the grid.
    - width: Width of the grid.
    """
    win.fill(WHITE)

    for row in grid:
        for spot in row:
            spot.draw(win)

    draw_grid(win, rows, width)
    pygame.display.update()


def get_clicked_pos(pos: Tuple[int, int], rows: int, width: int) -> Tuple[int, int]:
    """
    Convert mouse click position to grid coordinates.

    Parameters:
    - pos: Tuple (x, y) representing the mouse click position.
    - rows: Number of rows in the grid.
    - width: Width of the grid.

    Returns:
    - Tuple (row, col) representing the grid coordinates.
    """
    gap = width // rows
    y, x = pos
    row = y // gap
    col = x // gap
    return row, col

def draw_path_length(win: pygame.Surface, length: int, width: int) -> None:
    """
    Draw the length of the path on the window.

    Parameters:
    - win: Pygame window.
    - length: Length of the path.
    - width: Width of the window.
    """
    pygame.font.init()  # Initialize the Pygame font module
    font = pygame.font.Font(None, 36)
    text = font.render(f"Path Length: {length}", True, "yellow", "red")
    win.blit(text, (WIDTH // 2.5, 10))

def main(win: pygame.Surface, width: int, grid_type: int) -> None:
    """
    Main function to run the Pygame application.

    Parameters:
    - win: Pygame window.
    - width: Width of the window.
    """
    ROWS = 50
    if grid_type == 0: # free style grid
        grid = make_grid(ROWS, width)
    elif grid_type == 1: #r andom waze
        grid = make_grid_with_maze(ROWS, width)
    else: # warehouse grid
        grid = make_grid_warehouse(ROWS, width)

    start = None
    end = None

    run = True
    path_length = 0
    while run:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False

            if pygame.mouse.get_pressed()[0]:  # LEFT mouse button
                pos = pygame.mouse.get_pos()
                row, col = get_clicked_pos(pos, ROWS, width)
                spot = grid[row][col]
                if not start and spot != end:
                    start = spot
                    start.make_start()

                elif not end and spot != start:
                    end = spot
                    end.make_end()

                elif spot != end and spot != start:
                    spot.make_barrier()

            elif pygame.mouse.get_pressed()[2]:  # RIGHT mouse button
                pos = pygame.mouse.get_pos()
                row, col = get_clicked_pos(pos, ROWS, width)
                spot = grid[row][col]
                spot.reset()
                start, end = None if spot == start else start, None if spot == end else end

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE and start and end:
                    for row in grid:
                        for spot in row:
                            spot.update_neighbors(grid)

                    path_found, path_length = algorithm(lambda: draw(win, grid, ROWS, width), grid, start, end)
                    if path_found:
                        draw(win, grid, ROWS, width)
                        draw_path_length(win, path_length, width)
                        pygame.display.update()

                if event.key == pygame.K_c:
                    start = None
                    end = None
                    grid = make_grid(ROWS, width)

        draw(win, grid, ROWS, width)
        draw_path_length(win, path_length, width)
        pygame.display.update()

    pygame.quit()



# Run the main function
# Gris type: 0=free style, 1=rando waze, 2=warehouse grid
if __name__ == "__main__":
    main(win=WIN, width=WIDTH, grid_type=2)



