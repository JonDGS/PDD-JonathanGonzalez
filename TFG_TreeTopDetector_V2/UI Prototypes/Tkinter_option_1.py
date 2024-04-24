import tkinter as tk
from tkinter import filedialog
from PIL import ImageTk, Image

class ImageViewerApp:
    def __init__(self, master):
        self.master = master
        self.master.title("Image Viewer")
        
        self.img_label = tk.Label(self.master)
        self.img_label.pack()

        self.btn_select_image = tk.Button(self.master, text="Select Image", command=self.load_image)
        self.btn_select_image.pack()

    def load_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.png;*.jpg;*.jpeg;*.gif")])
        if file_path:
            img = Image.open(file_path)
            img.thumbnail((300, 300))  # Resize the image to fit in the window
            img = ImageTk.PhotoImage(img)
            self.img_label.config(image=img)
            self.img_label.image = img  # Keep a reference to avoid garbage collection

def main():
    root = tk.Tk()

    root.title("Tree Top Detector v.2")

    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()

    window_width = 1100
    window_height = 700

    x = int((screen_width - window_width) / 2)
    y = int((screen_height - window_height) / 2)

    root.geometry(f"{window_width}x{window_height}+{x}+{y}")
    app = ImageViewerApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()