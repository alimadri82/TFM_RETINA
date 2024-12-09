from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.textinput import TextInput
from kivy.uix.image import Image
from kivy.uix.label import Label
from kivy.uix.filechooser import FileChooserListView
from kivy.uix.popup import Popup
from kivy.core.window import Window
from kivy.lang import Builder
from tensorflow.keras.models import load_model
import numpy as np
import cv2

class RetinaClassificationForm(BoxLayout):
    def __init__(self, **kwargs):
        super(RetinaClassificationForm, self).__init__(**kwargs)
        self.orientation = 'vertical'
        Window.size = (800, 500)  # Aumentar el tamaño del formulario
        Window.clearcolor = (239 / 255, 244 / 255, 249 / 255, 1)

        # Cargar el modelo entrenado 
        self.model = load_model('modeloretinaentrenado.h5')

        # Título arriba del todo
        self.title_label = Label(text='Proyecto de clasificación de retina', color=(0, 0, 0, 1), font_size='30sp', halign='left', valign='middle', pos_hint={'x': 0.1, 'top': 1}, height=50) 
        self.title_label.bind(size=self.title_label.setter('text_size')) 
        self.add_widget(self.title_label)
        
        # Descripción breve 
        self.description_label = Label(text='Selecciona una imagen para ver si la retina está o no enferma.', color=(0, 0, 0, 1), font_size='16sp', halign='left', valign='middle', pos_hint={'x': 0.1, 'top': 1}, height=50) 
        self.description_label.bind(size=self.description_label.setter('text_size')) 
        self.add_widget(self.description_label)
        
        # Layout para el resultado 
        result_layout = BoxLayout(orientation='horizontal', spacing=10, size_hint=(1, None), height=50) 
        self.result_label = Label(text="Resultado:", size_hint=(None, 1), width=130, color=(0, 0, 0, 1), halign='left', valign='middle', pos_hint={'x': 0.1, 'top': 1}, height=50)
        self.result_label.bind(size=self.result_label.setter('text_size')) 
        result_layout.add_widget(self.result_label) 
        self.result_image = Image(source='resultado.jpg', size_hint=(None, None), size=(50, 50)) 
        result_layout.add_widget(self.result_image) 
        self.add_widget(result_layout)

        # Layout principal
        main_layout = BoxLayout(orientation='horizontal', padding=20, spacing=20)
        self.add_widget(main_layout)

        # Bloque izquierdo
        left_block = BoxLayout(orientation='vertical', padding=20, spacing=10, size_hint=(0.7, 1))
        ruta_layout = BoxLayout(orientation='horizontal', spacing=5, size_hint=(1, None), height=30)
        self.ruta_label = Label(text="Ruta:", size_hint=(None, 1), width=50, color=(0, 0, 0, 1), halign='left', valign='middle') 
        self.ruta_label.bind(size=self.ruta_label.setter('text_size')) 
        ruta_layout.add_widget(self.ruta_label) 
        self.ruta_input = TextInput(hint_text='Ingrese la ruta', size_hint=(None, None), height=30, width=350,halign='left', readonly=True)
        ruta_layout.add_widget(self.ruta_input) 
        # Botón para elegir ruta 
        self.choose_button = Button(text='Elegir ruta', size_hint=(None, None), height=30, width=100) 
        self.choose_button.bind(on_press=self.show_filechooser) 
        ruta_layout.add_widget(self.choose_button) 
        left_block.add_widget(ruta_layout) 
        main_layout.add_widget(left_block)


        # Bloque derecho
        right_block = BoxLayout(orientation='vertical', padding=20, spacing=20, size_hint=(0.3, 1))
        right_block.add_widget(Image(source='UOC.jpg', size_hint=(None, None), size=(200, 200), pos_hint={'x': 0, 'top': 1.75}))
        right_block.add_widget(Image(source='Retina_Portada.jpg', size_hint=(None, None), size=(200, 200), pos_hint={'x': 0, 'top': 1.5}))
        buttons_layout = BoxLayout(orientation='horizontal', size_hint=(1, None), height=40, padding=(0, 0, 10, 0))
        self.accept_button = Button(text='Aceptar', size_hint=(None, None), height=30, width=100)
        self.accept_button.bind(on_press=self.on_accept)
        buttons_layout.add_widget(self.accept_button)
        self.exit_button = Button(text='Salir', size_hint=(None, None), height=30, width=100)
        self.exit_button.bind(on_press=self.on_exit)
        buttons_layout.add_widget(self.exit_button)
        right_block.add_widget(buttons_layout)
        main_layout.add_widget(right_block)

    def show_filechooser(self, instance):
        filechooser = FileChooserListView(filters=['*.png', '*.jpg', '*.jpeg'])
        filechooser_layout = BoxLayout(orientation='vertical')
        filechooser_layout.add_widget(filechooser)
        accept_button = Button(text='Aceptar', size_hint_y=None, height=40)
        accept_button.bind(on_press=lambda x: self.on_filechooser_dismiss(filechooser, popup))
        filechooser_layout.add_widget(accept_button)
        popup = Popup(title='Seleccionar archivo', content=filechooser_layout, size_hint=(0.9, 0.9))
        popup.open()

    def on_filechooser_dismiss(self, filechooser, popup):
        if filechooser.selection:
            self.ruta_input.text = filechooser.selection[0]
        popup.dismiss()

    def preprocess_image(self, image_path): 
        image = cv2.imread(image_path) 
        image = cv2.resize(image, (224, 224)) # Redimensionar a 224x224 píxeles 
        image = image / 255.0 # Normalizar 
        image = np.expand_dims(image, axis=0) # Añadir una dimensión para el batch 
        return image 
    
    def predict(self, image_path): 
        image = self.preprocess_image(image_path) 
        prediction = self.model.predict(image) 
        predicted_class = np.argmax(prediction, axis=1) 
        return predicted_class
    
    def on_accept(self, instance):
        if self.ruta_input.text: 
            prediction = self.predict(self.ruta_input.text) 
            result = f"Clase predicha: {prediction[0]}"
            self.result_label.text = f"Resultado: {result}" 
            print(f'Ruta seleccionada: {self.ruta_input.text}') 
            print(f'Resultado: {result}') 
        else: 
            print('Por favor, selecciona una imagen.')

    def on_exit(self, instance):
        App.get_running_app().stop()

class MyApp(App):
    def build(self):
        return RetinaClassificationForm()

if __name__ == '__main__':
    MyApp().run()
