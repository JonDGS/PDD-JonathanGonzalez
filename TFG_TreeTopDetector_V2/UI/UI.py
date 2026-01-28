import os
import time
import shutil
import datetime
from tkinter import *
from tkinter import filedialog
from PIL import ImageTk, Image
from ultralytics import YOLO

CURRENT_DIR = os.getcwd()

WINDOW_WIDTH = 950
WINDOW_HEIGHT  = 700

DEFAULT_IMAGE = os.path.join(CURRENT_DIR, 'miselaneos/default_image_bg.png')
FOLDER_PATH_ORIGINAL = os.path.join(CURRENT_DIR, 'test/labels/')

# Define model directories
DETECTION_MODELS_DIR = os.path.join(CURRENT_DIR, 'modelos/detect')
CLASSIFICATION_MODELS_DIR = os.path.join(CURRENT_DIR, 'modelos/classify')
TYPE_MODELS_DIR = os.path.join(CURRENT_DIR, 'modelos/types')

class TreeTopViewer():

    def __init__(self, main_window):
        self.flag_image = False
        self.make_prediction = False 
        self.main_window = main_window

        self.detection_model = None 
        self.classification_model = None 
        self.type_model = None
        self.last_output_dir = None # To store the latest prediction output directory
        
        self.tree_counts = {} 
        
        self.main_window.title("Tree Top Detector v.3")

        screen_width = self.main_window.winfo_screenwidth()
        screen_height = self.main_window.winfo_screenheight()

        x = int((screen_width - WINDOW_WIDTH) / 2)
        y = int((screen_height - WINDOW_HEIGHT) / 2)

        self.main_window.geometry(f"{WINDOW_WIDTH}x{WINDOW_HEIGHT}+{x}+{y}")

        self.main_window.configure(
            bg='#27374D'
        )

        self.image_canvas = PanedWindow(main_window, height=680,  width=680, background='#27374D')
        self.image_canvas.place(x=10, y=10)
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
            initial_model = 'best3.onnx' 
            if initial_model in self.available_detection_models:
                self.selected_detection_model_name.set(initial_model)
            else:
                self.selected_detection_model_name.set(self.available_detection_models[0])
            self.update_detection_model_selection(self.selected_detection_model_name.get()) 
        else:
            self.selected_detection_model_name.set("No detection models found")
            self.make_prediction = False 
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
            initial_cls_model = 'best3.onnx'
            if initial_cls_model in self.available_classification_models:
                self.selected_classification_model_name.set(initial_cls_model)
            else:
                self.selected_classification_model_name.set(self.available_classification_models[0])
            self.update_classification_model_selection(self.selected_classification_model_name.get()) 
        else:
            self.selected_classification_model_name.set("No classification models found")
            print("Warning: No classification models found in 'modelos/classify' directory.")

        self.classification_model_selector_label = Label(self.results_paned, text="Classification Model:", background='#DDE6ED', font=("MontserratRoman", 10))
        self.classification_model_selector_label.place(x=135, y=5) 

        self.classification_model_selector = OptionMenu(
            self.results_paned,
            self.selected_classification_model_name,
            *self.available_classification_models,
            command=self.update_classification_model_selection
        )
        self.classification_model_selector.config(bg='#9DB2BF', fg='black', font=('MontserratRoman', 10), width=15)
        self.classification_model_selector["menu"].config(bg='#DDE6ED', fg='black', font=('MontserratRoman', 10))
        self.classification_model_selector.place(x=135, y=25)

        # --- Type Model Selection Dropdown ---
        self.available_type_models = self.get_available_models(TYPE_MODELS_DIR)
        self.selected_type_model_name = StringVar(self.main_window)

        if self.available_type_models:
            initial_type_model = 'best3.onnx'
            if initial_type_model in self.available_type_models:
                self.selected_type_model_name.set(initial_type_model)
            else:
                self.selected_type_model_name.set(self.available_type_models[0])
            self.update_type_model_selection(self.selected_type_model_name.get()) 
        else:
            self.selected_type_model_name.set("No type models found")
            print("Warning: No type models found in 'modelos/type' directory.")

        self.type_model_selector_label = Label(self.results_paned, text="Type Model:", background='#DDE6ED', font=("MontserratRoman", 10))
        self.type_model_selector_label.place(x=5, y=65) 

        self.type_model_selector = OptionMenu(
            self.results_paned,
            self.selected_type_model_name,
            *self.available_type_models,
            command=self.update_type_model_selection
        )
        self.type_model_selector.config(bg='#9DB2BF', fg='black', font=('MontserratRoman', 10), width=15)
        self.type_model_selector["menu"].config(bg='#DDE6ED', fg='black', font=('MontserratRoman', 10))
        self.type_model_selector.place(x=5, y=85)

        REAL_COUNT_PLACE_Y = 120
        
        self.count_label_o = Label(self.results_paned, text='Total de Arboles Reales:', background='#DDE6ED', font=("MontserratRoman", 12))
        self.count_label_o.place(x=10, y=REAL_COUNT_PLACE_Y)

        self.count_text_box_o = Label(self.results_paned, text='', background='#DDE6ED', font=("MontserratRoman", 12))
        self.count_text_box_o.place(x=200, y=REAL_COUNT_PLACE_Y)

        INF_COUNT_PLACE_Y = REAL_COUNT_PLACE_Y + 20 
        
        self.count_label = Label(self.results_paned, text='Total de Arboles Inferidos:', background='#DDE6ED', font=("MontserratRoman", 12))
        self.count_label.place(x=10, y=INF_COUNT_PLACE_Y)

        self.count_text_box = Label(self.results_paned, text='', background='#DDE6ED', font=("MontserratRoman", 12))
        self.count_text_box.place(x=200, y=INF_COUNT_PLACE_Y)

        PRECISION_PLACE_Y = INF_COUNT_PLACE_Y + 40 
        
        self.precision_label = Label(self.results_paned, text='Precisión:', background='#DDE6ED', font=("MontserratRoman", 12))
        self.precision_label.place(x=10, y=PRECISION_PLACE_Y)

        self.precision_box = Label(self.results_paned, text='', background='#DDE6ED', font=("MontserratRoman", 12))
        self.precision_box.place(x=200, y=PRECISION_PLACE_Y)
        
        self.warning_image = Label(self.results_paned, text='', background='#DDE6ED', foreground='yellow',font=("MontserratRoman", 14))
        self.warning_image.place(x=10, y=530)

        CLASSIFICATION_RESULTS_PLACE_Y = 160 
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
        self.make_prediction = True 
        print(f"Detection model set to: {selectedModel}")

    def update_classification_model_selection(self, selectedModel):
        """
        Updates the classification model when a new one is selected from the dropdown.
        """
        self.classification_model = YOLO(os.path.join(CLASSIFICATION_MODELS_DIR, selectedModel))
        print(f"Classification model set to: {selectedModel}")

    def update_type_model_selection(self, selectedModel):
        """
        Updates the type model when a new one is selected from the dropdown.
        """
        self.classification_model = YOLO(os.path.join(TYPE_MODELS_DIR, selectedModel))
        print(f"Type model set to: {selectedModel}")

    def load_image(self):
        self.file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.png *.jpg *.jpeg *.gif")])
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
            self.classification_results_label.config(text="Resultados de Clasificación:\n") 
            self.warning_image.config(text="") 
            self.flag_image = True
            # Clear previous prediction results to avoid confusion
            if os.path.exists(os.path.join(CURRENT_DIR, 'runs')):
                shutil.rmtree(os.path.join(CURRENT_DIR, 'runs'))
            self.last_output_dir = None # Clear previous output directory reference

    def save_image(self):
        # Use self.last_output_dir to refer to the most recent prediction results
        if hasattr(self, 'last_output_dir') and self.last_output_dir and os.path.exists(self.last_output_dir):
            time_now = datetime.datetime.now()
            time_now = time_now.strftime("%Y-%m-%d--%H-%M-%S")
            save_path = os.path.join(CURRENT_DIR, 'saves/predict-{}'.format(time_now))
            try:
                shutil.copytree(self.last_output_dir, save_path)
                print(f"Contenido de la carpeta '{self.last_output_dir}' copiado a '{save_path}' correctamente.")
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

        self.warning_image.config(text="") 
        self.classification_results_label.config(text="Resultados de Clasificación:\n") 
        
        # Run detection model
        self.results = self.detection_model.predict(
            source=self.file_path, 
            save = True,
            save_crop = False, 
            iou = 0.5,
            augment = True,
            show_labels = False,
            show_conf = True, 
            save_txt = True,
            project = CURRENT_DIR + "/runs/detect")  
        
        # Get actual output directory from results
        output_dir = self.results[0].save_dir
        self.last_output_dir = output_dir # Store for save_image method

        # Path to the processed image saved by YOLO
        processed_img_path = os.path.join(output_dir, os.path.basename(self.file_path))
        
        # Path to the labels folder
        txt_folder_path = os.path.join(output_dir, 'labels')
        
        # Find the label file (there should only be one for a single image prediction)
        txt_path = None
        detections = []
        if os.path.exists(txt_folder_path):
            for filename in os.listdir(txt_folder_path):
                if filename.endswith(('.txt')):
                    txt_path = os.path.join(txt_folder_path, filename)
                    break 

        # Read detections if the label file exists
        if txt_path and os.path.exists(txt_path):
            with open(txt_path, 'r') as f:
                detections = [line.strip().split() for line in f]

        # Handle case where no trees were detected
        if not detections:
            self.warning_image.config(text="NO SE DETECTARON ÁRBOLES EN LA IMAGEN")
            self.count_text_box.config(text="0")
            self.precision_box.config(text="N/A")
            self.classification_results_label.config(text="Resultados de Clasificación:\nNo se detectaron árboles.")
            # Display the processed image even if no detections
            img_to_display = Image.open(processed_img_path)
            img_to_display.thumbnail((640, 640)) 
            img_to_display = ImageTk.PhotoImage(img_to_display)
            self.image_upload.config(image=img_to_display)
            self.image_upload.image = img_to_display  
            return

        # Load the original image for cropping
        original_image = Image.open(self.file_path)        
        width, height = original_image.size
        
        self.tree_counts = {} # Reset tree counts for each prediction

        for i, det in enumerate(detections):
            # Ensure detection has enough elements before unpacking
            if len(det) < 5:
                print(f"Skipping malformed detection: {det}")
                continue
            
            class_id, x_center, y_center, w, h = map(float, det[:5]) # Take only first 5 elements

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
            tree_crop = original_image.crop((x1, y1, x2, y2))

            # Predict classification
            results = self.classification_model.predict(source=tree_crop, verbose=False)
            
            # Check if classification results contain probabilities and names
            if results and results[0].probs is not None and results[0].names is not None:
                class_name = results[0].names[results[0].probs.top1].strip()  
                # Update the tree counts
                if class_name in self.tree_counts:
                    self.tree_counts[class_name] += 1
                else:
                    self.tree_counts[class_name] = 1
            else:
                print(f"Warning: No valid classification result for crop {i}. Skipping.")

        # Display the processed image (with detections, if any)
        img_to_display = Image.open(processed_img_path)
        img_to_display.thumbnail((640, 640)) 
        img_to_display = ImageTk.PhotoImage(img_to_display)
        self.image_upload.config(image=img_to_display)
        self.image_upload.image = img_to_display  

        self.inferencias = len(detections) # Total inferred trees is the count of detections
        self.count_text_box.config(text=self.inferencias)
        self.precision_box.config(text=self.calculate_precision())

        # Display the classification results in the new label
        classification_text = "Resultados de Clasificación:\n" + "\n".join([f"{tree_type}: {count}" for tree_type, count in self.tree_counts.items()])
        self.classification_results_label.config(text=classification_text)
         
    # The tree_count method is no longer needed as self.inferencias is directly assigned.
    # def tree_count(self):
    #     txt_folder_path = FOLDER_PATH+'/labels'
    #     for filename in os.listdir(txt_folder_path):
    #         if filename.endswith(('.txt')):
    #             txt_path = os.path.join(txt_folder_path, filename)
    #             with open(txt_path, 'r') as f:
    #                 num_filas = sum(1 for linea in f)
    #             return num_filas
    #     return 0 
    
    def tree_count_original(self):
        # Handle case where original labels might not exist for the loaded image
        txt_path = FOLDER_PATH_ORIGINAL+self.filename.replace('.jpg', '.txt').replace('.jpeg', '.txt').replace('.png', '.txt') # Handle various image extensions
        if os.path.exists(txt_path):
            with open(txt_path, 'r') as f:
                num_filas = sum(1 for linea in f)
            return num_filas
        else:
            self.warning_image.config(text="NO SE ENCONTRARON ETIQUETAS ORIGINALES")
            return "N/A" # Indicate that original count is not available
    
    def calculate_precision(self):
        if isinstance(self.valor_real, int) and self.valor_real > 0:
            if self.inferencias is not None: # Ensure inferencias has a value
                return f"{self.inferencias/self.valor_real:.2f}"
        return "N/A" 


def main():
    main_window = Tk()
    app = TreeTopViewer(main_window)
    main_window.mainloop()

if __name__ == "__main__":
    main()