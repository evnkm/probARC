"""Common library for ARC"""

import numpy as np
import random


class Color:
    """
    Enum for colors

    Color.BLACK, Color.BLUE, Color.RED, Color.GREEN, Color.YELLOW, Color.GREY, Color.PINK, Color.ORANGE, Color.TEAL, Color.MAROON

    Use Color.ALL_COLORS for `set` of all possible colors
    Use Color.NOT_BLACK for `set` of all colors except black

    Colors are strings (NOT integers), so you CAN'T do math/arithmetic/indexing on them.
    (The exception is Color.BLACK, which is 0)
    """

def flood_fill(grid, x, y, color, connectivity=4):
    """
    Fill the connected region that contains the point (x, y) with the specified color.

    connectivity: 4 or 8, for 4-way or 8-way connectivity. 8-way counts diagonals as connected, 4-way only counts cardinal directions as connected.
    """

def _flood_fill(grid, x, y, color, old_color, connectivity):
    """
    internal function not used by LLM
    """

def draw_line(grid, x, y, end_x=None, end_y=None, length=None, direction=None, color=None, stop_at_color=[]):
    """
    Draws a line starting at (x, y) extending to (end_x, end_y) or of the specified length in the specified direction
    Direction should be a vector with elements -1, 0, or 1.
    If length is None, then the line will continue until it hits the edge of the grid.

    stop_at_color: optional list of colors that the line should stop at. If the line hits a pixel of one of these colors, it will stop.

    Returns the endpoint of the line.

    Example:
    # blue diagonal line from (0, 0) to (2, 2)
    stop_x, stop_y = draw_line(grid, 0, 0, length=3, color=blue, direction=(1, 1))
    draw_line(grid, 0, 0, end_x=2, end_y=2, color=blue)
    assert (stop_x, stop_y) == (2, 2)
    """

def find_connected_components(
    grid, background=Color.BLACK, connectivity=4, monochromatic=True
):
    """
    Find the connected components in the grid. Returns a list of connected components, where each connected component is a numpy array.

    connectivity: 4 or 8, for 4-way or 8-way connectivity.
    monochromatic: if True, each connected component is assumed to have only one color. If False, each connected component can include multiple colors.
    """

def randomly_scatter_points(grid, color, density=0.5, background=Color.BLACK):
    """
    Randomly scatter points of the specified color in the grid with specified density.

    Example usage:
    randomly_scatter_points(grid, color=a_color, density=0.5, background=background_color)
    """

def scale_sprite(sprite, factor):
    """
    Scales the sprite by the specified factor.

    Example usage:
    scaled_sprite = scale_sprite(sprite, factor=3)
    original_width, original_height = sprite.shape
    scaled_width, scaled_height = scaled_sprite.shape
    assert scaled_width == original_width * 3 and scaled_height == original_height * 3
    """

def blit(grid, sprite, x=0, y=0, background=None):
    """
    Copies the sprite into the grid at the specified location. Modifies the grid in place.

    background: color treated as transparent. If specified, only copies the non-background pixels of the sprite.
    """

def blit_object(grid, obj, background=Color.BLACK):
    """
    Draws an object onto the grid using its current location.

    Example usage:
    blit_object(output_grid, an_object, background=background_color)
    """

def blit_sprite(grid, sprite, x, y, background=Color.BLACK):
    """
    Draws a sprite onto the grid at the specified location.

    Example usage:
    blit_sprite(output_grid, the_sprite, x=x, y=y, background=background_color)
    """

def bounding_box(grid, background=Color.BLACK):
    """
    Find the bounding box of the non-background pixels in the grid.
    Returns a tuple (x, y, width, height) of the bounding box.

    Example usage:
    objects = find_connected_components(input_grid, monochromatic=True, background=Color.BLACK, connectivity=8)
    teal_object = [ obj for obj in objects if np.any(obj == Color.TEAL) ][0]
    teal_x, teal_y, teal_w, teal_h = bounding_box(teal_object)
    """

def bounding_box_mask(grid, background=Color.BLACK):
    """
    Find the bounding box of the non-background pixels in the grid.
    Returns a mask of the bounding box.

    Example usage:
    objects = find_connected_components(input_grid, monochromatic=True, background=Color.BLACK, connectivity=8)
    teal_object = [ obj for obj in objects if np.any(obj == Color.TEAL) ][0]
    teal_bounding_box_mask = bounding_box_mask(teal_object)
    # teal_bounding_box_mask[x, y] is true if and only if (x, y) is in the bounding box of the teal object
    """

def object_position(obj, background=Color.BLACK, anchor="upper left"):
    """
    (x,y) position of the provided object. By default, the upper left corner.

    anchor: "upper left", "upper right", "lower left", "lower right", "center", "upper center", "lower center", "left center", "right center"

    Example usage:
    x, y = object_position(obj, background=background_color, anchor="upper left")
    middle_x, middle_y = object_position(obj, background=background_color, anchor="center")
    """

def object_colors(obj, background=Color.BLACK):
    """
    Returns a list of colors in the object.

    Example usage:
    colors = object_colors(obj, background=background_color)
    """

def crop(grid, background=Color.BLACK):
    """
    Crop the grid to the smallest bounding box that contains all non-background pixels.

    Example usage:
    # Extract a sprite from an object
    sprite = crop(an_object, background=background_color)
    """

def translate(obj, x, y, background=Color.BLACK):
    """
    Translate by the vector (x, y). Fills in the new pixels with the background color.

    Example usage:
    red_object = ... # extract some object
    shifted_red_object = translate(red_object, x=1, y=1)
    blit_object(output_grid, shifted_red_object, background=background_color)
    """

def collision(
    _=None, object1=None, object2=None, x1=0, y1=0, x2=0, y2=0, background=Color.BLACK
):
    """
    Check if object1 and object2 collide when object1 is at (x1, y1) and object2 is at (x2, y2).

    Example usage:

    # Check if a sprite can be placed onto a grid at (X,Y)
    collision(object1=output_grid, object2=a_sprite, x2=X, y2=Y)

    # Check if two objects collide
    collision(object1=object1, object2=object2, x1=X1, y1=Y1, x2=X2, y2=Y2)
    """

def contact(
    _=None,
    object1=None,
    object2=None,
    x1=0,
    y1=0,
    x2=0,
    y2=0,
    background=Color.BLACK,
    connectivity=4,
):
    """
    Check if object1 and object2 touch each other (have contact) when object1 is at (x1, y1) and object2 is at (x2, y2).
    They are touching each other if they share a border, or if they overlap. Collision implies contact, but contact does not imply collision.

    connectivity: 4 or 8, for 4-way or 8-way connectivity. (8-way counts diagonals as touching, 4-way only counts cardinal directions as touching)

    Example usage:

    # Check if a sprite touches anything if it were to be placed at (X,Y)
    contact(object1=output_grid, object2=a_sprite, x2=X, y2=Y)

    # Check if two objects touch each other
    contact(object1=object1, object2=object2)
    """

def randomly_spaced_indices(max_len, n_indices, border_size=1, padding=1):
    """
    Generate randomly-spaced indices guaranteed to not be adjacent.
    Useful for generating random dividers.

    padding: guaranteed empty space in between indices
    border_size: guaranteed empty space at the border

    Example usage:
    x_indices = randomly_spaced_indices(grid.shape[0], num_dividers, border_size=1, padding=2) # make sure each region is at least 2 pixels wide
    for x in x_indices:
        grid[x, :] = divider_color
    """

def check_between_objects(obj1, obj2, x, y, padding = 0, background=Color.BLACK):
    """
    Check if a pixel is between two objects.

    padding: minimum distance from the edge of the objects

    Example usage:
    if check_between_objects(obj1, obj2, x, y, padding=1, background=background_color):
        # do something
    """

def random_free_location_for_sprite(
    grid,
    sprite,
    background=Color.BLACK,
    border_size=0,
    padding=0,
    padding_connectivity=8,
):
    """
    Find a random free location for the sprite in the grid
    Returns a tuple (x, y) of the top-left corner of the sprite in the grid, which can be passed to `blit_sprite`

    border_size: minimum distance from the edge of the grid
    background: color treated as transparent
    padding: if non-zero, the sprite will be padded with a non-background color before checking for collision
    padding_connectivity: 4 or 8, for 4-way or 8-way connectivity when padding the sprite

    Example usage:
    x, y = random_free_location_for_sprite(grid, sprite, padding=1, padding_connectivity=8, border_size=1, background=Color.BLACK) # find the location, using generous padding
    assert not collision(object1=grid, object2=sprite, x2=x, y2=y)
    blit_sprite(grid, sprite, x, y)

    If no free location can be found, raises a ValueError.
    """

def random_free_location_for_object(*args, **kwargs):
    """
    internal function not used by LLM

    exists for backward compatibility
    """

def object_interior(grid, background=Color.BLACK):
    """
    Computes the interior of the object (including edges)

    returns a new grid of `bool` where True indicates that the pixel is part of the object's interior.

    Example usage:
    interior = object_interior(obj, background=Color.BLACK)
    for x, y in np.argwhere(interior):
        # x,y is either inside the object or at least on its edge
    """

def object_boundary(grid, background=Color.BLACK):
    """
    Computes the boundary of the object (excluding interior)

    returns a new grid of `bool` where True indicates that the pixel is part of the object's boundary.

    Example usage:
    boundary = object_boundary(obj, background=Color.BLACK)
    assert np.all(obj[boundary] != Color.BLACK)
    """

def object_neighbors(grid, background=Color.BLACK, connectivity=4):
    """
    Computes a mask of the points that neighbor or border the object, but are not part of the object.

    returns a new grid of `bool` where True indicates that the pixel is part of the object's border neighbors5.

    Example usage:
    neighbors = object_neighbors(obj, background=Color.BLACK)
    assert np.all(obj[neighbors] == Color.BLACK)
    """

class Symmetry:
    """
    Symmetry transformations, which transformed the 2D grid in ways that preserve visual structure.
    Returned by `detect_rotational_symmetry`, `detect_translational_symmetry`, `detect_mirror_symmetry`.
    """

    def apply(self, x, y, iters=1):
        """
        Apply the symmetry transformation to the point (x, y) `iters` times.
        Returns the transformed point (x',y')
        """

def orbit(grid, x, y, symmetries):
    """
    Compute the orbit of the point (x, y) under the symmetry transformations `symmetries`.
    The orbit is the set of points that the point (x, y) maps to after applying the symmetry transformations different numbers of times.
    Returns a list of points in the orbit.

    Example:
    symmetries = detect_rotational_symmetry(input_grid)
    for x, y in np.argwhere(input_grid != Color.BLACK):
        # Compute orbit on to the target grid, which is typically the output
        symmetric_points = orbit(output_grid, x, y, symmetries)
        # ... now we do something with them like copy colors or infer missing colors
    """

class TranslationalSymmetry(Symmetry):
    """
    Translation symmetry transformation, which repeatedly translates by a fixed vector

    Example usage:
    # Create a translational symmetry that translates by (dx, dy)
    symmetry = TranslationalSymmetry(translate_x=dx, translate_y=dy)
    # example of using orbit to tile the entire canvas
    for x, y in np.argwhere(input_grid != Color.BLACK):
        # Compute orbit on to the target grid, which is typically the output
        symmetric_points = orbit(output_grid, x, y, [symmetry])
        for x, y in symmetric_points:
            output_grid[x, y] = input_grid[x, y]
    """

def detect_translational_symmetry(grid, ignore_colors=[Color.BLACK], background=None):
    """
    Finds translational symmetries in a grid.
    Satisfies: grid[x, y] == grid[x + translate_x, y + translate_y] for all x, y, as long as neither pixel is in `ignore_colors`, and as long as x,y is not background.

    Returns a list of Symmetry objects, each representing a different translational symmetry.

    Example:
    symmetries = detect_translational_symmetry(grid, ignore_colors=[occluder_color], background=background_color)
    for x, y in np.argwhere(grid != occluder_color & grid != background_color):
        # Compute orbit on to the target grid
        # When copying to an output, this is usually the output grid
        symmetric_points = orbit(grid, x, y, symmetries)
        for x, y in symmetric_points:
            assert grid[x, y] == grid[x, y] or grid[x, y] == occluder_color
    """

class MirrorSymmetry():
    """
    Mirror symmetry transformation, which flips horizontally and/or vertically

    Example usage:
    symmetry = MirrorSymmetry(mirror_x=x if "horizontal" else None, mirror_y=y if "vertical" else None)

    # Flip mirrored_object over the symmetry and draw to the output
    for x, y in np.argwhere(mirrored_object != background):
        x2, y2 = symmetry.apply(x, y)
        output_grid[x2, y2] = mirrored_object[x, y]
    
    """

def detect_mirror_symmetry(grid, ignore_colors=[Color.BLACK], background=None):
    """
    Returns list of mirror symmetries.
    Satisfies: grid[x, y] == grid[2*mirror_x - x, 2*mirror_y - y] for all x, y, as long as neither pixel is in `ignore_colors`

    Example:
    symmetries = detect_mirror_symmetry(grid, ignore_colors=[Color.RED], background=Color.BLACK) # ignore_color: In case parts of the object have been removed and occluded by red
    for x, y in np.argwhere(grid != Color.BLACK & grid != Color.RED): # Everywhere that isn't background and isn't occluded
        for sym in symmetries:
            symmetric_x, symmetric_y = sym.apply(x, y)
            assert grid[symmetric_x, symmetric_y] == grid[x, y] or grid[symmetric_x, symmetric_y] == Color.RED

    If the grid has both horizontal and vertical mirror symmetries, the returned list will contain two elements.
    """

def detect_rotational_symmetry(grid, ignore_colors=[Color.BLACK], background=None):
    """
    Finds rotational symmetry in a grid, or returns None if no symmetry is possible.
    Satisfies: grid[x, y] == grid[y - rotate_center_y + rotate_center_x, -x + rotate_center_y + rotate_center_x] # clockwise
               grid[x, y] == grid[-y + rotate_center_y + rotate_center_x, x - rotate_center_y + rotate_center_x] # counterclockwise
               for all x, y, as long as neither pixel is in `ignore_colors`, and as long as x, y is not `background`.

    Example:
    sym = detect_rotational_symmetry(grid, ignore_colors=[Color.GREEN], background=Color.BLACK) # ignore_color: In case parts of the object have been removed and occluded by black
    for x, y in np.argwhere(grid != Color.GREEN):
        rotated_x, rotated_y = sym.apply(x, y, iters=1) # +1 clockwise, -1 counterclockwise
        assert grid[rotated_x, rotated_y] == grid[x, y] or grid[rotated_x, rotated_y] == Color.GREEN or grid[x, y] == Color.BLACK
    print(sym.center_x, sym.center_y) # In case these are needed, they are floats
    """

def _score_symmetry(grid, symmetry, ignore_colors, background=None):
    """
    internal function not used by LLM

    Given a grid, scores how well the grid satisfies the symmetry.

    Returns:
     the number of nonbackground pixels that are perfectly preserved by the symmetry
     the number of nonbackground pixels that are mapped outside the canvas (kind of bad)

     the number of nonbackground pixels that are mapped to a different color (very bad)
    """

def show_colored_grid(grid, text=True):
    """
    internal function not used by LLM
    Not used by the language model, used by the rest of the code for debugging
    """

def visualize(input_generator, transform, n_examples=5, n_attempts=100):
    """
    internal function not used by LLM
    """

def apply_symmetry(sprite, symmetry_type, background=Color.BLACK):
    """
    internal function not used by LLM
    Apply the specified symmetry within the bounds of the sprite.
    """

def apply_diagonal_symmetry(sprite, background=Color.BLACK):
    """
    internal function not used by LLM
    Apply diagonal symmetry within the bounds of the sprite. Assumes square sprite.
    """

def is_contiguous(bitmask, background=Color.BLACK, connectivity=4):
    """
    Check if an array is contiguous.

    background: Color that counts as transparent (default: Color.BLACK)
    connectivity: 4 or 8, for 4-way (only cardinal directions) or 8-way connectivity (also diagonals) (default: 4)

    Returns True/False
    """

def generate_sprite(
    n,
    m,
    symmetry_type,
    fill_percentage=0.5,
    max_colors=9,
    color_palate=None,
    connectivity=4,
    background=Color.BLACK
):
    """
    internal function not used by LLM
    """

def random_sprite(n, m, density=0.5, symmetry=None, color_palette=None, connectivity=4, background=Color.BLACK):
    """
    Generate a sprite (an object), represented as a numpy array.

    n, m: dimensions of the sprite. If these are lists, then a random value will be chosen from the list.
    symmetry: optional type of symmetry to apply to the sprite. Can be 'horizontal', 'vertical', 'diagonal', 'radial', 'mirror', 'not_symmetric'. If None, a random symmetry type will be chosen.
    color_palette: optional list of colors to use in the sprite. If None, a random color palette will be chosen.

    Returns an (n,m) NumPy array representing the sprite.
    """

def detect_objects(grid, _=None, predicate=None, background=Color.BLACK, monochromatic=False, connectivity=None, allowed_dimensions=None, colors=None, can_overlap=False):
    """
    Detects and extracts objects from the grid that satisfy custom specification.

    predicate: a function that takes a candidate object as input and returns True if it counts as an object
    background: color treated as transparent
    monochromatic: if True, each object is assumed to have only one color. If False, each object can include multiple colors.
    connectivity: 4 or 8, for 4-way or 8-way connectivity. If None, the connectivity is determined automatically.
    allowed_dimensions: a list of tuples (n, m) specifying the allowed dimensions of the objects. If None, objects of any size are allowed.
    colors: a list of colors that the objects are allowed to have. If None, objects of any color are allowed.
    can_overlap: if True, objects can overlap. If False, objects cannot overlap.

    Returns a list of objects, where each object is a numpy array.
    """
