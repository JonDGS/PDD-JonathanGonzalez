import kivy

from kivy.app import App
from kivy.uix.label import Label
from kivy.uix.image import Image

class MyApp(App):
    def build(self):
        # return Label(text = 'Hello Mariano')

        img = Image(source='result.jpg')  # Replace 'your_image.jpg' with the path to your image

        return img

if __name__ == '__main__':
    MyApp().run()