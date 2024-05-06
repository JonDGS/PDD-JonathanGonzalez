import cv2

IMAGE_HEIGHT = 640
IMAGE_WIDTH = 640

def read_file_columns(file_path):
    centers_x = []
    centers_y = []
    widths = []
    heights = []
    with open(file_path, 'r') as file:
        for line in file:
            # Split each line by spaces and convert values to float
            values = line.strip().split()
            # Extract values of the 2nd and 3rd columns (indexes 1 and 2)
            center_x, center_y = float(values[1]), float(values[2])
            width, height = float(values[3]), float(values[4])
            # Create a tuple from the extracted values and append it to the data list
            centers_x.append(center_x)
            centers_y.append(center_y)
            widths.append(width)
            heights.append(height)
    return centers_x,centers_y,widths,heights


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



def draw_rectangle(image_name):
    """
    Draw a rectangle on an image given the coordinates of its top-left and bottom-right vertices.

    Parameters:
    - image_path (str): Path to the image file.
    - vertex1 (tuple): Coordinates of the top-left vertex of the rectangle in the format (x, y).
    - vertex2 (tuple): Coordinates of the bottom-right vertex of the rectangle in the format (x, y).
    - color (tuple): BGR color of the rectangle (default is green).
    - thickness (int): Thickness of the rectangle border (default is 2).
    """

    image_path = './../Tree Counting Original/test/images/{}.jpg'.format(image_name)
    real_values_path = './../Tree Counting Original/test/labels/{}.txt'.format(image_name)
    inferit_values_path = './../UI/runs/detect/predict/labels/{}.txt'.format(image_name)
    image = cv2.imread(image_path)

    thickness=2
    original_color=(0, 255, 0)
    predicted_color=(0, 0, 255)
    r_centers_x,r_centers_y,r_widths,r_heights = read_file_columns(real_values_path)
    i_centers_x,i_centers_y,i_widths,i_heights = read_file_columns(inferit_values_path)
    for i in range(len(r_centers_x)):
        original_vertex1, original_vertex2 = get_bounding_box_coordinates(r_centers_x[i],r_centers_y[i],r_widths[i],r_heights[i])    
        cv2.rectangle(image, original_vertex1, original_vertex2, original_color, thickness)
    for i in range(len(i_centers_x)):
        predicted_vertex1, predicted_vertex2 = get_bounding_box_coordinates(i_centers_x[i],i_centers_y[i],i_widths[i],i_heights[i])
        cv2.rectangle(image, predicted_vertex1, predicted_vertex2, predicted_color, thickness)

    # Display the image with the rectangle
    cv2.imshow('Image with Rectangle', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


draw_rectangle('test_10')
