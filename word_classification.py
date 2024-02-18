import cv2
import os
import tkinter as tk
from PIL import Image, ImageTk  # Import the Image and ImageTk modules from the PIL library


class ImageLabeler:
    def __init__(self, root, image_folder, output_folder):
        self.root = root
        self.image_folder = image_folder
        self.output_folder = output_folder
        # Check if the output folder exists, and create it if not
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        self.image_list = os.listdir(image_folder)
        self.current_index = 0

        self.label_entry = tk.Entry(root)
        self.label_entry.pack()
        self.label_entry.bind("<Return>", lambda event: self.next_image())

        self.previous_button = tk.Button(root, text="Previous", command=self.previous_image)
        self.previous_button.pack()

        self.next_button = tk.Button(root, text="Next", command=self.next_image)
        self.next_button.pack()

        self.delete_button = tk.Button(root, text="Delete", command=self.delete_current_image)
        self.delete_button.pack()
        self.label_entry.bind("<Delete>", lambda event: self.delete_current_image())

        self.skip_button = tk.Button(root, text="Skip", command=self.skip_image)
        self.skip_button.pack()

        self.image_label = tk.Label(root)
        self.image_label.pack()

        self.load_image()

    def load_image(self):
        if 0 <= self.current_index < len(self.image_list):
            image_path = os.path.join(self.image_folder, self.image_list[self.current_index])
            self.image = cv2.imread(image_path)
            self.display_image()
        else:
            print("All images labeled!")

    def display_image(self):
        image_rgb = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        height, width, channels = image_rgb.shape

        image_pil = Image.fromarray(image_rgb)
        image_tk = ImageTk.PhotoImage(image=image_pil)

        self.image_label.config(image=image_tk)
        self.image_label.image = image_tk

        self.root.update_idletasks()
        self.root.update()

    def next_image(self, event=None):
        label = self.label_entry.get()

        if label.lower() == "/delete":
            self.delete_current_image()
        else:
            if label:
                current_image_path = os.path.join(self.image_folder, self.image_list[self.current_index])
                new_image_name = f"{label}_{self.current_index + 1}.png"
                new_image_path = os.path.join(self.output_folder, new_image_name)

                i = 0
                while os.path.exists(new_image_path):
                    i += 1
                    new_image_path = os.path.join(self.output_folder, f"{label}_{self.current_index + i}.png")

                try:
                    os.rename(current_image_path, new_image_path)
                except FileExistsError:
                    print(f"Error: File '{new_image_name}' already exists. Renaming to {new_image_path}")

            self.current_index += 1
            self.label_entry.delete(0, tk.END)

            if self.current_index < len(self.image_list):
                self.load_image()
            else:
                print("All images labeled!")
                self.root.destroy()

    def delete_current_image(self):
        current_image_path = os.path.join(self.image_folder, self.image_list[self.current_index])
        os.remove(current_image_path)

        self.current_index += 1
        self.label_entry.delete(0, tk.END)
        self.load_image()

        if self.current_index < len(self.image_list):
            self.load_image()
        else:
            print("All images labeled!")
            self.root.destroy()

    def skip_image(self):
        self.current_index += 1
        self.label_entry.delete(0, tk.END)

        if self.current_index < len(self.image_list):
            self.load_image()
        else:
            print("All images labeled!")
            self.root.destroy()

    def previous_image(self):
        if self.current_index != 0:
            self.current_index -= 1
        if self.current_index >= 0:
            self.label_entry.delete(0, tk.END)
            self.load_image()
        else:
            print("You are at the first image.")


if __name__ == "__main__":

    # when classifying do NOT include spaces
    root = tk.Tk()
    root.title("Image Labeler")

    image_folder = "words_to_label"
    output_folder = "labelled_images_school"
    labeler = ImageLabeler(root, image_folder, output_folder)

    root.mainloop()
