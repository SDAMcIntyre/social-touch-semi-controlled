from kivy.app import App
from kivy.clock import Clock
from kivy.core.window import Window
from kivy.lang import Builder
from kivy.properties import (
    NumericProperty, ReferenceListProperty, ObjectProperty
)
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.image import Image
from kivy.uix.label import Label
from kivy.uix.widget import Widget
from kivy.vector import Vector

from random import randint

Builder.load_file("images.kv")


class ImageBox():
    pass


class VariableLayout(BoxLayout):
    orientation = 'vertical'
    spacing = 1

    def __init__(self, **kwargs):
        super(VariableLayout, self).__init__(**kwargs)


class GlobalLayout(Widget):
    pass


class UserInterfaceApp(App):
    def build(self):
        Window.clearcolor = (0, 0, 0, 0)
        return GlobalLayout()

    def run(self):
        super(UserInterfaceApp, self).run()
        print("meow")


if __name__ == '__main__':
    UserInterfaceApp().run()