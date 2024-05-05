from tkinter import *
from tkinter import filedialog
from PIL import ImageTk, Image
from ultralytics import YOLO
import os
import time
import shutil

WINDOW_WIDTH = 950
WINDOW_HEIGHT  = 700

DEFAULT_IMAGE = 'miselaneos/deafult_image_bg.png'
FOLDER_PATH = './runs/detect/predict'
FOLDER_PATH_ORIGINAL = './../Tree Counting Original/test/labels/'
MODEL = YOLO('modelos/best.onnx')  # load an official detection model

class TreeTopViewer():

    def __init__(self, main_window):

        self.flag_image = False
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

        self.count_label_o = Label(self.results_paned, text='Total de Arboles Reales:', background='#DDE6ED', font=("MontserratRoman", 12))
        self.count_label_o.place(x=10, y=10)

        self.count_text_box_o = Label(self.results_paned, text='', background='#DDE6ED', font=("MontserratRoman", 12))
        self.count_text_box_o.place(x=200, y=10)
        
        self.count_label = Label(self.results_paned, text='Total de Arboles Inferidos:', background='#DDE6ED', font=("MontserratRoman", 12))
        self.count_label.place(x=10, y=30)

        self.count_text_box = Label(self.results_paned, text='', background='#DDE6ED', font=("MontserratRoman", 12))
        self.count_text_box.place(x=200, y=30)
        
        self.precision_label = Label(self.results_paned, text='Precisión:', background='#DDE6ED', font=("MontserratRoman", 12))
        self.precision_label.place(x=10, y=70)

        self.precision_box = Label(self.results_paned, text='', background='#DDE6ED', font=("MontserratRoman", 12))
        self.precision_box.place(x=200, y=70)
        
        self.warning_image = Label(self.results_paned, text='', background='#DDE6ED', foreground='yellow',font=("MontserratRoman", 14))
        self.warning_image.place(x=10, y=530)

        self.btn_upload = Button(self.results_paned, text= 'Cargar Imagen', command=self.load_image, background='#9DB2BF', foreground='black',font=('MontserratRoman', 12), width=21)
        # self.btn_upload.pack(side=BOTTOM, pady=5)
        self.btn_upload.place(x=10, y=560)

        self.btn_proccess = Button(self.results_paned, text= 'Procesar Imagen', command=self.predict_image, background='#9DB2BF', foreground='black',font=('MontserratRoman', 12),width=21)
        # self.btn_proccess.pack(side=BOTTOM, pady=5)
        self.btn_proccess.place(x=10, y=600)



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

    def predict_image(self):
        if os.path.exists('runs'):
                shutil.rmtree('runs')
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
                save_txt = True)  # predict on an image

            try_number = 0
            
            while(not os.path.exists(FOLDER_PATH)):
                time.sleep(2)
                if try_number > 10:
                    print("No se ha creado la carpeta")
                    break
                try_number+=1
                    
            for filename in os.listdir(FOLDER_PATH):
                if filename.endswith(('.jpg')):
                    img_path = os.path.join(FOLDER_PATH, filename)
    

            # print('#########################\n')        
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