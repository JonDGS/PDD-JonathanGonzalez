
import cv2

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


def draw_rectangle(image_name, original_vertex1, original_vertex2, predicted_vertex1, predicted_vertex2):
    """
    Draw a rectangle on an image given the coordinates of its top-left and bottom-right vertices.

    Parameters:
    - image_path (str): Path to the image file.
    - vertex1 (tuple): Coordinates of the top-left vertex of the rectangle in the format (x, y).
    - vertex2 (tuple): Coordinates of the bottom-right vertex of the rectangle in the format (x, y).
    - color (tuple): BGR color of the rectangle (default is green).
    - thickness (int): Thickness of the rectangle border (default is 2).
    """
    # Load the image
    image_path = 'Tree Counting Original/test/images/{}.jpg'.format(image_name)
    image = cv2.imread(image_path)

    # Draw the rectangle on the image
    thickness=2
    original_color=(0, 255, 0)
    predicted_color=(0, 0, 255)
    cv2.rectangle(image, original_vertex1, original_vertex2, original_color, thickness)
    cv2.rectangle(image, predicted_vertex1, predicted_vertex2, predicted_color, thickness)

    # Display the image with the rectangle
    cv2.imshow('Image with Rectangle', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Example usage:
# Specify the image path


# Define the coordinates of the rectangle vertices
# Format: (x, y)
# Top-left vertex
original_vertex1, original_vertex2 = get_bounding_box_coordinates(0.88828125, 0.80390625, 0.0484375, 0.0625)
predicted_vertex1, predicted_vertex2 = get_bounding_box_coordinates(0.886379, 0.801462, 0.0608061, 0.0647972)


# Bottom-right vertex

# Call the function to draw the rectangle
draw_rectangle('test_1', original_vertex1, original_vertex2, predicted_vertex1, predicted_vertex2)