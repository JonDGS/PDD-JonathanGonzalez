#module import

from tkinter import *
from tkinter import filedialog
from tkinter import ttk
from PIL import ImageTk, Image
from ultralytics import YOLO
import os
import time


WINDOW_WIDTH = 1040
WINDOW_HEIGHT  = 700

DEFAULT_IMAGE = 'deafult_image_bg.png'
FOLDER_PATH = './runs/detect/predict'
MODEL = YOLO('best.onnx')  # load an official detection model

class TreeTopViewer():

    def __init__(self, main_window):
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
            bg='#DFD0B8'
        )


        self.image_canvas = PanedWindow(main_window, height=680,  width=680, background='#DFD0B8')
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

        self.btn_upload = Button(self.image_canvas, text= 'Cargar Imagen', command=self.load_image, background='#153448', foreground='#DFD0B8',font=('MontserratRoman', 16) )
        self.btn_upload.pack(side=BOTTOM, pady=5)
        

        # button_style = ttk.Style()
        # button_style.configure(style='custom.TButton', background='#153448', foreground='#153448', font=('MontserratRoman', 16))

        # ttk.Button(
        #     self.image_canvas,
        #     text = 'Cargar Imagen',
        #     command = self.load_image,
        #     style='custom.TButton',
        #     ).pack(side=BOTTOM)

        self.results_paned = PanedWindow(main_window, height=640,  width=360, background='#3C5B6F')
        self.results_paned.place(x=665, y=10)

        self.count_label = Label(self.results_paned, text='Total de Arboles:')
        self.count_label.pack(pady=5)

        self.count_text_box = Text(self.results_paned, height=5, width=40)
        self.count_text_box.pack(pady=5)

        self.btn_proccess = Button(self.results_paned, text= 'Procesar Imagen', command=self.predict_image, background='#153448', foreground='#DFD0B8',font=('MontserratRoman', 16) )
        self.btn_proccess.pack(side=BOTTOM, pady=5)

    def load_image(self):
        self.file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.png;*.jpg;*.jpeg;*.gif")])
        if self.file_path:
            img = Image.open(self.file_path)
            img.thumbnail((640, 640))  # Resize the image to fit in the window
            img = ImageTk.PhotoImage(img)
            self.image_upload.config(image=img)
            self.image_upload.image = img  # Keep a reference to avoid garbage collection
        

    def predict_image(self):
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
   

        print('#########################\n')        
        print(img_path)        
        img = Image.open(img_path)
        img.thumbnail((640, 640))  # Resize the image to fit in the window
        img = ImageTk.PhotoImage(img)
        self.image_upload.config(image=img)
        self.image_upload.image = img  # Keep a reference to avoid garbage collection

        self.count_text_box.insert("end", self.tree_count())

         
    def tree_count(self):
        txt_folder_path = FOLDER_PATH+'/labels'
        for filename in os.listdir(txt_folder_path):
            if filename.endswith(('.txt')):
                txt_path = os.path.join(txt_folder_path, filename)
        with open(txt_path, 'r') as f:
            num_filas = sum(1 for linea in f)
        return num_filas
def main():
    main_window = Tk()
    app = TreeTopViewer(main_window)
    main_window.mainloop()

if __name__ == "__main__":
    main()