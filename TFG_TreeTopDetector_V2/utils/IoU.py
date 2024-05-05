
IMAGE_HEIGHT = 640
IMAGE_WIDTH = 640

def get_bounding_box_coordinates(test_img_center_x, test_img_center_y,test_img_width, test_img_height):

    #Calculate coordinates
    x_center_pxl = test_img_center_x * IMAGE_WIDTH
    y_center_pxl = test_img_center_y * IMAGE_HEIGHT

    half_width = test_img_width * IMAGE_WIDTH / 2
    half_height = test_img_height * IMAGE_HEIGHT / 2

    x_min = int(x_center_pxl - half_width)
    y_min = int(y_center_pxl - half_height)

    x_max = int(x_center_pxl + half_width)
    y_max = int(y_center_pxl + half_height)

    

    return (x_min, y_min), (x_max, y_max)


def intersection_area(coord1_tl, coord1_br, coord2_tl, coord2_br):
    """
    Calculate the intersection area of two rectangles given tuples of coordinates representing their opposite corners.

    Parameters:
    - coord1_tl (tuple): Tuple of coordinates (x1, y1) of the top-left corner of the first rectangle.
    - coord1_br (tuple): Tuple of coordinates (x2, y2) of the bottom-right corner of the first rectangle.
    - coord2_tl (tuple): Tuple of coordinates (x1, y1) of the top-left corner of the second rectangle.
    - coord2_br (tuple): Tuple of coordinates (x2, y2) of the bottom-right corner of the second rectangle.

    Returns:
    - float: Intersection area of the two rectangles.
    """
    x_overlap = max(0, min(coord1_br[0], coord2_br[0]) - max(coord1_tl[0], coord2_tl[0]))
    y_overlap = max(0, min(coord1_br[1], coord2_br[1]) - max(coord1_tl[1], coord2_tl[1]))
    return x_overlap * y_overlap

def rectangle_area(coord_tl, coord_br):
    """
    Calculate the area of a rectangle given tuples of coordinates representing its opposite corners.

    Parameters:
    - coord_tl (tuple): Tuple of coordinates (x1, y1) of the top-left corner of the rectangle.
    - coord_br (tuple): Tuple of coordinates (x2, y2) of the bottom-right corner of the rectangle.

    Returns:
    - float: Area of the rectangle.
    """
    width = abs(coord_br[0] - coord_tl[0])
    height = abs(coord_br[1] - coord_tl[1])
    return width * height

def union_area(coord1_tl, coord1_br, coord2_tl, coord2_br):
    """
    Calculate the union area of two rectangles given tuples of coordinates representing their opposite corners.

    Parameters:
    - coord1_tl (tuple): Tuple of coordinates (x1, y1) of the top-left corner of the first rectangle.
    - coord1_br (tuple): Tuple of coordinates (x2, y2) of the bottom-right corner of the first rectangle.
    - coord2_tl (tuple): Tuple of coordinates (x1, y1) of the top-left corner of the second rectangle.
    - coord2_br (tuple): Tuple of coordinates (x2, y2) of the bottom-right corner of the second rectangle.

    Returns:
    - float: Union area of the two rectangles.
    """
    intersection = intersection_area(coord1_tl, coord1_br, coord2_tl, coord2_br)
    area_rect1 = rectangle_area(coord1_tl, coord1_br)
    area_rect2 = rectangle_area(coord2_tl, coord2_br)
    union = area_rect1 + area_rect2 - intersection
    return union, intersection

# Example usage:
# coord1_tl = (1, 1)
# coord1_br = (5, 4)
# coord2_tl = (3, 2)
# coord2_br = (7, 6)

coord1_tl, coord1_br = get_bounding_box_coordinates(0.886379, 0.801462, 0.0608061, 0.0647972)
coord2_tl, coord2_br = get_bounding_box_coordinates(0.88828125, 0.80390625, 0.0484375, 0.0625)

union_area_value, intersection_area_area_value = union_area(coord1_tl, coord1_br, coord2_tl, coord2_br)
print("Union area:", union_area_value)

print("Intersection area:", intersection_area_area_value)

IoU = intersection_area_area_value / union_area_value

print("IoU:", IoU)