import os
import time
import shutil
import datetime
from tkinter import *
from tkinter import filedialog
from PIL import ImageTk, Image
from ultralytics import YOLO
from torchvision import transforms # This import is not used in the provided code, consider removing if not needed.

CURRENT_DIR = os.getcwd()

WINDOW_WIDTH = 950
WINDOW_HEIGHT  = 700

DEFAULT_IMAGE = os.path.join(CURRENT_DIR, 'miselaneos/default_image_bg.png')
FOLDER_PATH = os.path.join(CURRENT_DIR, 'runs/detect/predict/')
FOLDER_PATH_ORIGINAL = os.path.join(CURRENT_DIR, 'test/labels/')

# Define model directories
DETECTION_MODELS_DIR = os.path.join(CURRENT_DIR, 'modelos/detect')
CLASSIFICATION_MODELS_DIR = os.path.join(CURRENT_DIR, 'modelos/classify')

class TreeTopViewer():

    def __init__(self, main_window):

        self.flag_image = False
        self.make_prediction = False # This flag could be refined to indicate if models are loaded.
        self.main_window = main_window

        self.detection_model = None # Initialize detection model as None
        self.classification_model = None # Initialize classification model as None
        
        self.tree_counts = {} # Dictionary to store counts for each tree class
        
        #defines the main window's title
        self.main_window.title("Tree Top Detector v.3")

        #defines the main window's size
        screen_width = self.main_window.winfo_screenwidth()
        screen_height = self.main_window.winfo_screenheight()

        x = int((screen_width - WINDOW_WIDTH) / 2)
        y = int((screen_height - WINDOW_HEIGHT) / 2)

        self.main_window.geometry(f"{WINDOW_WIDTH}x{WINDOW_HEIGHT}+{x}+{y}")

        #defines the main window background color
        self.main_window.configure(
            bg='#27374D'
        )

        self.image_canvas = PanedWindow(main_window, height=680,  width=680, background='#27374D')
        self.image_canvas.place(x=10, y=10)
        #create the image canvas
        image_defualt = Image.open(DEFAULT_IMAGE)
        image_defualt.thumbnail((640,640))
        image_defualt = ImageTk.PhotoImage(image_defualt)

        self.image_upload = Label(self.image_canvas, bd=0)
        self.image_upload.config(image=image_defualt)
        self.image_upload.image = image_defualt
        self.image_upload.pack()

        self.results_paned = PanedWindow(main_window, height=640,  width=270, background='#DDE6ED')
        self.results_paned.place(x=665, y=10)

        # --- Detection Model Selection Dropdown ---
        self.available_detection_models = self.get_available_models(DETECTION_MODELS_DIR)
        self.selected_detection_model_name = StringVar(self.main_window)

        if self.available_detection_models:
            # Set initial selected model to 'best4.onnx' if it exists, otherwise the first one
            initial_model = 'best4.onnx' # Original model was 'best4.onnx'
            if initial_model in self.available_detection_models:
                self.selected_detection_model_name.set(initial_model)
            else:
                self.selected_detection_model_name.set(self.available_detection_models[0])
            self.update_detection_model_selection(self.selected_detection_model_name.get()) # Initialize self.detection_model
        else:
            self.selected_detection_model_name.set("No detection models found")
            self.make_prediction = False # Prevent prediction if no model is found
            print("Warning: No detection models found in 'modelos' directory.")

        self.detection_model_selector_label = Label(self.results_paned, text="Detection Model:", background='#DDE6ED', font=("MontserratRoman", 10))
        self.detection_model_selector_label.place(x=5, y=5)

        self.detection_model_selector = OptionMenu(
            self.results_paned,
            self.selected_detection_model_name,
            *self.available_detection_models,
            command=self.update_detection_model_selection
        )
        self.detection_model_selector.config(bg='#9DB2BF', fg='black', font=('MontserratRoman', 10), width=15)
        self.detection_model_selector["menu"].config(bg='#DDE6ED', fg='black', font=('MontserratRoman', 10))
        self.detection_model_selector.place(x=5, y=25)

        # --- Classification Model Selection Dropdown ---
        self.available_classification_models = self.get_available_models(CLASSIFICATION_MODELS_DIR)
        self.selected_classification_model_name = StringVar(self.main_window)

        if self.available_classification_models:
            # Set initial selected classification model to 'best1.onnx' if it exists, otherwise the first one
            initial_cls_model = 'best1.onnx'
            if initial_cls_model in self.available_classification_models:
                self.selected_classification_model_name.set(initial_cls_model)
            else:
                self.selected_classification_model_name.set(self.available_classification_models[0])
            self.update_classification_model_selection(self.selected_classification_model_name.get()) # Initialize self.classification_model
        else:
            self.selected_classification_model_name.set("No classification models found")
            # Prediction for classification results won't work, but detection might still.
            print("Warning: No classification models found in 'modelos/classify' directory.")

        self.classification_model_selector_label = Label(self.results_paned, text="Classification Model:", background='#DDE6ED', font=("MontserratRoman", 10))
        self.classification_model_selector_label.place(x=135, y=5) # Placed next to detection label

        self.classification_model_selector = OptionMenu(
            self.results_paned,
            self.selected_classification_model_name,
            *self.available_classification_models,
            command=self.update_classification_model_selection
        )
        self.classification_model_selector.config(bg='#9DB2BF', fg='black', font=('MontserratRoman', 10), width=15)
        self.classification_model_selector["menu"].config(bg='#DDE6ED', fg='black', font=('MontserratRoman', 10))
        self.classification_model_selector.place(x=135, y=25) # Placed next to detection dropdown

        # Adjust existing elements' y-coordinates to make space for the two dropdowns
        # Dropdowns occupy roughly y=5 to y=50. Subsequent elements start from ~y=60.
        REAL_COUNT_PLACE_Y = 60
        
        self.count_label_o = Label(self.results_paned, text='Total de Arboles Reales:', background='#DDE6ED', font=("MontserratRoman", 12))
        self.count_label_o.place(x=10, y=REAL_COUNT_PLACE_Y)

        self.count_text_box_o = Label(self.results_paned, text='', background='#DDE6ED', font=("MontserratRoman", 12))
        self.count_text_box_o.place(x=200, y=REAL_COUNT_PLACE_Y)

        INF_COUNT_PLACE_Y = 80 # Adjusted from 60
        
        self.count_label = Label(self.results_paned, text='Total de Arboles Inferidos:', background='#DDE6ED', font=("MontserratRoman", 12))
        self.count_label.place(x=10, y=INF_COUNT_PLACE_Y)

        self.count_text_box = Label(self.results_paned, text='', background='#DDE6ED', font=("MontserratRoman", 12))
        self.count_text_box.place(x=200, y=INF_COUNT_PLACE_Y)

        PRECISION_PLACE_Y = 120 # Adjusted from 100
        
        self.precision_label = Label(self.results_paned, text='Precisión:', background='#DDE6ED', font=("MontserratRoman", 12))
        self.precision_label.place(x=10, y=PRECISION_PLACE_Y)

        self.precision_box = Label(self.results_paned, text='', background='#DDE6ED', font=("MontserratRoman", 12))
        self.precision_box.place(x=200, y=PRECISION_PLACE_Y)
        
        self.warning_image = Label(self.results_paned, text='', background='#DDE6ED', foreground='yellow',font=("MontserratRoman", 14))
        self.warning_image.place(x=10, y=530)

        CLASSIFICATION_RESULTS_PLACE_Y = 160 # Adjusted from 140
        self.classification_results_label = Label(self.results_paned, text='Resultados de Clasificación:\n', background='#DDE6ED', font=("MontserratRoman", 12), justify=LEFT)
        self.classification_results_label.place(x=10, y=CLASSIFICATION_RESULTS_PLACE_Y)

        self.btn_save = Button(self.results_paned, text= 'Guardar Resultado', command=self.save_image, background='#9DB2BF', foreground='black',font=('MontserratRoman', 12), width=21)
        self.btn_save.place(x=10, y=520)
        
        self.btn_upload = Button(self.results_paned, text= 'Cargar Imagen', command=self.load_image, background='#9DB2BF', foreground='black',font=('MontserratRoman', 12), width=21)
        self.btn_upload.place(x=10, y=560)

        self.btn_proccess = Button(self.results_paned, text= 'Procesar Imagen', command=self.predict_image, background='#9DB2BF', foreground='black',font=('MontserratRoman', 12),width=21)
        self.btn_proccess.place(x=10, y=600)

    def get_available_models(self, directory):
        """
        Gets a list of .onnx models from the specified directory.
        """
        models = []
        if os.path.exists(directory):
            for filename in os.listdir(directory):
                if filename.endswith('.onnx') and os.path.isfile(os.path.join(directory, filename)):
                    models.append(filename)
        return sorted(models)

    def update_detection_model_selection(self, selectedModel):
        """
        Updates the detection model when a new one is selected from the dropdown.
        """
        self.detection_model = YOLO(os.path.join(DETECTION_MODELS_DIR, selectedModel))
        self.make_prediction = True # Re-enable prediction if a model is selected
        print(f"Detection model set to: {selectedModel}")

    def update_classification_model_selection(self, selectedModel):
        """
        Updates the classification model when a new one is selected from the dropdown.
        """
        self.classification_model = YOLO(os.path.join(CLASSIFICATION_MODELS_DIR, selectedModel))
        print(f"Classification model set to: {selectedModel}")

    def load_image(self):
        self.file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.png;*.jpg;*.jpeg;*.gif")])
        self.filename = self.file_path.split("/")[-1]
        print(self.filename)
        if self.file_path:
            img = Image.open(self.file_path)
            img.thumbnail((640, 640))
            img = ImageTk.PhotoImage(img)
            self.image_upload.config(image=img)
            self.image_upload.image = img
            self.valor_real = self.tree_count_original()
            self.count_text_box_o.config(text=self.valor_real)
            self.count_text_box.config(text="")
            self.precision_box.config(text="")
            self.classification_results_label.config(text="Resultados de Clasificación:\n") # Clear previous classification results
            self.warning_image.config(text="") # Clear any previous warnings
            self.flag_image = True
            if os.path.exists('runs'):
                shutil.rmtree('runs')

    def save_image(self):
        if self.make_prediction and os.path.exists(os.path.join(CURRENT_DIR, 'runs/detect/predict')):
            time_now = datetime.datetime.now()
            time_now = time_now.strftime("%Y-%m-%d--%H-%M-%S")
            save_path = os.path.join(CURRENT_DIR, 'saves/predict-{}'.format(time_now))
            try:
                shutil.copytree(os.path.join(CURRENT_DIR, 'runs/detect/predict'), save_path)
                print("Contenido de la carpeta copiado correctamente.")
            except shutil.Error as e:
                print(f"Error al copiar la carpeta: {e}")
            except OSError as e:
                print(f"Error: {e.strerror}")
        else:
            self.warning_image.config(text="NO HAY RESULTADOS PARA GUARDAR")


    def predict_image(self):
        self.make_prediction = True
        self.runs_directory = os.path.join(CURRENT_DIR, 'runs')
        if os.path.exists(self.runs_directory):
                shutil.rmtree(self.runs_directory)

        # Ensure an image is loaded and models are selected
        if not self.flag_image:
            self.warning_image.config(text="NO SE HA SELECCIONADO LA IMAGEN")
            return
        if not self.detection_model:
            self.warning_image.config(text="NO SE HA SELECCIONADO EL MODELO DE DETECCIÓN")
            return
        if not self.classification_model:
            self.warning_image.config(text="NO SE HA SELECCIONADO EL MODELO DE CLASIFICACIÓN")
            return

        self.warning_image.config(text="") # Clear previous warnings
        self.classification_results_label.config(text="Resultados de Clasificación:\n") # Clear previous results before displaying new ones
        
        self.results = self.detection_model.predict( # Use self.detection_model
            source=self.file_path, 
            save = True,
            save_crop = False, 
            iou = 0.5,
            augment = True,
            show_labels = False,
            show_conf = True, 
            save_txt = True,
            project = CURRENT_DIR + "/runs/detect")  # predict on an image
        
        image = Image.open(self.file_path)        
        for filename in os.listdir(FOLDER_PATH):
            if filename.endswith(('.jpg')):
                img_path = os.path.join(FOLDER_PATH, filename)

        txt_folder_path = FOLDER_PATH+'/labels'
        for filename in os.listdir(txt_folder_path):
            if filename.endswith(('.txt')):
                txt_path = os.path.join(txt_folder_path, filename)

        with open(txt_path, 'r') as f:
            detections = [line.strip().split() for line in f]

        # Crop and save each detected tree
        output_folder = os.path.join(CURRENT_DIR, 'runs/crops')
        os.makedirs(output_folder, exist_ok=True)
        
        height, width = image.size
        
        # Reset tree counts for each prediction
        self.tree_counts = {}

        for i, det in enumerate(detections):
            class_id, x_center, y_center, w, h = map(float, det)
            
            box_width = w * width
            box_height = h * height
            x1 = int((x_center * width) - (box_width / 2))
            y1 = int((y_center * height) - (box_height / 2))
            x2 = int(x1 + box_width)
            y2 = int(y1 + box_height)
            
            # Ensure coordinates are within image bounds
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(width, x2)
            y2 = min(height, y2)
            
            # Crop the image
            tree_crop = image.crop((x1, y1, x2, y2))

            results = self.classification_model.predict(source=tree_crop, verbose=False) # Use self.classification_model
            
            class_name = results[0].names[results[0].probs.top1].strip()  # Get the class name with highest probability
            
            # Update the tree counts
            if class_name in self.tree_counts:
                self.tree_counts[class_name] += 1
            else:
                self.tree_counts[class_name] = 1
            
        img = Image.open(img_path)
        img.thumbnail((640, 640))  # Resize the image to fit in the window
        img = ImageTk.PhotoImage(img)
        self.image_upload.config(image=img)
        self.image_upload.image = img  # Keep a reference to avoid garbage collection
        self.inferencias = self.tree_count()
        self.count_text_box.config(text=self.inferencias)
        self.precision_box.config(text=self.calculate_precision())

        # Display the classification results in the new label
        classification_text = "Resultados de Clasificación:\n" + "\n".join([f"{tree_type}: {count}" for tree_type, count in self.tree_counts.items()])
        self.classification_results_label.config(text=classification_text)
         
    def tree_count(self):
        txt_folder_path = FOLDER_PATH+'/labels'
        for filename in os.listdir(txt_folder_path):
            if filename.endswith(('.txt')):
                txt_path = os.path.join(txt_folder_path, filename)
                with open(txt_path, 'r') as f:
                    num_filas = sum(1 for linea in f)
                return num_filas
        return 0 # Return 0 if no txt file found
    
    def tree_count_original(self):
        # Handle case where original labels might not exist for the loaded image
        txt_path = FOLDER_PATH_ORIGINAL+self.filename.replace('jpg', 'txt')
        if os.path.exists(txt_path):
            with open(txt_path, 'r') as f:
                num_filas = sum(1 for linea in f)
            return num_filas
        else:
            self.warning_image.config(text="NO SE ENCONTRARON ETIQUETAS ORIGINALES")
            return "N/A" # Indicate that original count is not available
    
    def calculate_precision(self):
        if isinstance(self.valor_real, int) and self.valor_real > 0:
            return f"{self.inferencias/self.valor_real:.2f}"
        return "N/A" # Cannot calculate precision if original count is not available or zero


def main():
    main_window = Tk()
    app = TreeTopViewer(main_window)
    main_window.mainloop()

if __name__ == "__main__":
    main()