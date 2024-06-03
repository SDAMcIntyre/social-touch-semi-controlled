
import tkinter as tk


class WaitForButtonPressPopup:

    def __init__(self):
        self.root = tk.Tk()
        self.root.geometry("200x100")
        self.root.title("Press a key")

        def on_key_press(event):
            #print("Key pressed:", event.char)
            self.close_window()

        label = tk.Label(self.root, text="Press any key")
        label.pack()

        self.root.bind("<Key>", on_key_press)
        self.root.lift()
        self.root.after(1, lambda: self.root.focus_force())
        self.root.mainloop()

    def __del__(self):
        self.close_window()

    def close_window(self):
        try:
            self.root.quit()
        except:
            pass
        try:
            self.root.destroy()
        except:
            pass
