from tkinter import *
from tkinter import filedialog
from PIL import ImageTk, Image
from ultralytics import YOLO
import os
import time
import shutil
import datetime

CURRENT_DIR = os.getcwd()

WINDOW_WIDTH = 950
WINDOW_HEIGHT  = 700

DEFAULT_IMAGE = os.path.join(CURRENT_DIR, 'miselaneos/deafult_image_bg.png')
FOLDER_PATH = os.path.join(CURRENT_DIR, 'runs/detect/predict/')
FOLDER_PATH_ORIGINAL = os.path.join(CURRENT_DIR, 'test/labels/')
MODEL = YOLO(os.path.join(CURRENT_DIR, 'modelos/best4.onnx'))  # load an official detection model
# DEFAULT_IMAGE = 'miselaneos/deafult_image_bg.png'
# FOLDER_PATH = './runs/detect/predict'
# FOLDER_PATH_ORIGINAL = './../Tree Counting Original/test/labels/'
# MODEL = YOLO('modelos/best.onnx')  # load an official detection model
MODELS_DIR = os.path.join(CURRENT_DIR, 'modelos')
MODEL = None

class TreeTopViewer():

    def __init__(self, main_window):

        self.flag_image = False
        self.make_prediction = False
        self.main_window = main_window
        
        #defines the main window's title
        self.main_window.title("Tree Top Detector v.2")

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

        # self.image_upload = Label(self.main_window)
        self.image_upload = Label(self.image_canvas, bd=0)
        self.image_upload.config(image=image_defualt)
        self.image_upload.image = image_defualt
        self.image_upload.pack()

        self.results_paned = PanedWindow(main_window, height=640,  width=270, background='#DDE6ED')
        self.results_paned.place(x=665, y=10)

        # --- Model Selection Dropdown ---
        self.available_models = self.get_available_models()
        self.selected_model_name = StringVar(self.main_window)

        if self.available_models:
            # Set initial selected model to 'best.onnx' if it exists, otherwise the first one
            initial_model = 'best.onnx'
            if initial_model in self.available_models:
                self.selected_model_name.set(initial_model)
            else:
                self.selected_model_name.set(self.available_models[0])
            self.update_model_selection(self.selected_model_name.get()) # Initialize global MODEL
        else:
            self.selected_model_name.set("No models found")
            self.make_prediction = False # Prevent prediction if no model is found
            print("Warning: No models found in 'modelos' directory.")


        self.model_selector_label = Label(self.results_paned, text="Select Model:", background='#DDE6ED', font=("MontserratRoman", 10))
        self.model_selector_label.place(x=10, y=10)

        self.model_selector = OptionMenu(
            self.results_paned, # Place it inside results_paned
            self.selected_model_name,
            *self.available_models,
            command=self.update_model_selection
        )
        self.model_selector.config(bg='#9DB2BF', fg='black', font=('MontserratRoman', 10), width=15)
        self.model_selector["menu"].config(bg='#DDE6ED', fg='black', font=('MontserratRoman', 10))
        self.model_selector.place(x=5, y=5) # Position within results_paned

        # Adjust existing elements' y-coordinates to make space for the dropdown
        y_offset = 40

        REAL_COUNT_PLACE_Y = 40
        
        self.count_label_o = Label(self.results_paned, text='Total de Arboles Reales:', background='#DDE6ED', font=("MontserratRoman", 12))
        self.count_label_o.place(x=10, y=REAL_COUNT_PLACE_Y)

        self.count_text_box_o = Label(self.results_paned, text='', background='#DDE6ED', font=("MontserratRoman", 12))
        self.count_text_box_o.place(x=200, y=REAL_COUNT_PLACE_Y)

        INF_COUNT_PLACE_Y = 60
        
        self.count_label = Label(self.results_paned, text='Total de Arboles Inferidos:', background='#DDE6ED', font=("MontserratRoman", 12))
        self.count_label.place(x=10, y=INF_COUNT_PLACE_Y)

        self.count_text_box = Label(self.results_paned, text='', background='#DDE6ED', font=("MontserratRoman", 12))
        self.count_text_box.place(x=200, y=INF_COUNT_PLACE_Y)

        PRECISION_PLACE_Y = 100
        
        self.precision_label = Label(self.results_paned, text='Precisión:', background='#DDE6ED', font=("MontserratRoman", 12))
        self.precision_label.place(x=10, y=PRECISION_PLACE_Y)

        self.precision_box = Label(self.results_paned, text='', background='#DDE6ED', font=("MontserratRoman", 12))
        self.precision_box.place(x=200, y=PRECISION_PLACE_Y)
        
        self.warning_image = Label(self.results_paned, text='', background='#DDE6ED', foreground='yellow',font=("MontserratRoman", 14))
        self.warning_image.place(x=10, y=530)

        self.btn_save = Button(self.results_paned, text= 'Guardar Resultado', command=self.save_image, background='#9DB2BF', foreground='black',font=('MontserratRoman', 12), width=21)
        self.btn_save.place(x=10, y=520)
        
        self.btn_upload = Button(self.results_paned, text= 'Cargar Imagen', command=self.load_image, background='#9DB2BF', foreground='black',font=('MontserratRoman', 12), width=21)
        self.btn_upload.place(x=10, y=560)

        self.btn_proccess = Button(self.results_paned, text= 'Procesar Imagen', command=self.predict_image, background='#9DB2BF', foreground='black',font=('MontserratRoman', 12),width=21)
        self.btn_proccess.place(x=10, y=600)

    def get_available_models(self):
        models = []
        if os.path.exists(MODELS_DIR):
            for filename in os.listdir(MODELS_DIR):
                if filename.endswith('.onnx') and os.path.isfile(os.path.join(MODELS_DIR, filename)):
                    models.append(filename)
        return sorted(models)

    def update_model_selection(self, selectedModel):
        global MODEL
        MODEL = YOLO(os.path.join(CURRENT_DIR, 'modelos/'+selectedModel))

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
            self.flag_image = True
            if os.path.exists('runs'):
                shutil.rmtree('runs')
    def save_image(self):
        if self.make_prediction:
            time_now = datetime.datetime.now()
            time_now = time_now.strftime("%Y-%m-%d--%H-%M-%S")
            save_path = os.path.join(CURRENT_DIR, 'saves/predict-{}'.format(time_now))
            try:
                shutil.copytree(os.path.join(CURRENT_DIR, 'runs/detect/predict'), save_path)
                print("Contenido de la carpeta copiado correctamente.")
            except shutil.Error as e:
                print(f"Error al copiar la carpeta: {e}")
            except OSError as e:
                print("XXXXXXXXXXXX")
                print(f"Error: {e.strerror}")

    def predict_image(self):
        self.make_prediction = True
        self.runs_directory = os.path.join(CURRENT_DIR, 'runs')
        if os.path.exists(self.runs_directory):
                shutil.rmtree(self.runs_directory)
        if self.flag_image:
            self.warning_image.config(text="")
            self.results = MODEL.predict(
                source=self.file_path, 
                save = True,
                save_crop = False, 
                iou = 0.5,
                augment = True,
                show_labels = False,
                show_conf = True, 
                save_txt = True,
                project = CURRENT_DIR + "/runs/detect")  # predict on an image
            
            # while(not os.path.exists(FOLDER_PATH)):
            #     time.sleep(2)
            #     if try_number > 10:
            #         print("No se ha creado la carpeta")
            #         break
            #     try_number+=1
            
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
                
                # Save the cropped image
                crop_filename = f'tree_{i}.png'
                crop_path = os.path.join(output_folder, crop_filename)
                tree_crop.save(crop_path)

            # print('#########################/n')        
            # print(img_path)        
            img = Image.open(img_path)
            img.thumbnail((640, 640))  # Resize the image to fit in the window
            img = ImageTk.PhotoImage(img)
            self.image_upload.config(image=img)
            self.image_upload.image = img  # Keep a reference to avoid garbage collection
            self.inferencias = self.tree_count()
            self.count_text_box.config(text=self.inferencias)
            self.precision_box.config(text=self.calculate_precision())
        else:
            self.warning_image.config(text="NO SE HA SELECCIONADO LA IMAGEN")
         
    def tree_count(self):
        txt_folder_path = FOLDER_PATH+'/labels'
        for filename in os.listdir(txt_folder_path):
            if filename.endswith(('.txt')):
                txt_path = os.path.join(txt_folder_path, filename)
        with open(txt_path, 'r') as f:
            num_filas = sum(1 for linea in f)
        return num_filas
    
    def tree_count_original(self):
        txt_path = FOLDER_PATH_ORIGINAL+self.filename.replace('jpg', 'txt')
        with open(txt_path, 'r') as f:
            num_filas = sum(1 for linea in f)
        return num_filas
        # return 10
    
    def calculate_precision(self):
        return self.inferencias/self.valor_real

def main():
    main_window = Tk()
    app = TreeTopViewer(main_window)
    main_window.mainloop()

if __name__ == "__main__":
    main()