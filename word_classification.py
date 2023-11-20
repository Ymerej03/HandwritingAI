import cv2
import os
import tkinter as tk

import cv2
import os
import tkinter as tk
from PIL import Image, ImageTk  # Import the Image and ImageTk modules from the PIL library

class ImageLabeler:
    # chatGPT did this whole function to speed up image classification so i could focus on the image processing and
    # ai model training
    def __init__(self, root, image_folder):
        self.root = root
        self.image_folder = image_folder
        self.image_list = os.listdir(image_folder)
        self.current_index = 0

        # Create UI components
        self.label_entry = tk.Entry(root)
        self.label_entry.pack()
        self.label_entry.bind("<Return>", lambda event: self.next_image())

        self.next_button = tk.Button(root, text="Next", command=self.next_image)
        self.next_button.pack()

        # Create a Label widget for displaying images
        self.image_label = tk.Label(root)
        self.image_label.pack()

        # Load and display the first image
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

        # Convert the image to a PhotoImage format
        image_pil = Image.fromarray(image_rgb)
        image_tk = ImageTk.PhotoImage(image=image_pil)

        # Set the PhotoImage to the Label widget
        self.image_label.config(image=image_tk)
        self.image_label.image = image_tk  # Keep a reference to avoid garbage collection

        self.root.update_idletasks()
        self.root.update()

    def next_image(self, event=None):
        # Get the label entered by the user
        label = self.label_entry.get()

        # Check for delete command
        if label.lower() == "/delete":
            self.delete_current_image()
        else:
            # Rename the current image file
            if label:
                current_image_path = os.path.join(self.image_folder, self.image_list[self.current_index])
                new_image_name = f"{label}_{self.current_index + 1}.png"
                new_image_path = os.path.join(self.image_folder, new_image_name)

                # Handle existing file
                if os.path.exists(new_image_path):
                    print(f"Error: File '{new_image_name}' already exists.")
                    # Skip to the next image without incrementing the index
                    self.label_entry.delete(0, tk.END)  # Clear the entry
                    self.load_image()
                    return

                try:
                    os.rename(current_image_path, new_image_path)
                except FileExistsError:
                    print(f"Error: File '{new_image_name}' already exists.")

            # Move to the next image
            self.current_index += 1
            self.label_entry.delete(0, tk.END)  # Clear the entry

            # Check if all images are labeled
            if self.current_index < len(self.image_list):
                self.load_image()
            else:
                # Close the Tkinter window when all images are labeled
                print("All images labeled!")
                self.root.destroy()

    def delete_current_image(self):
        # Delete the current image file
        current_image_path = os.path.join(self.image_folder, self.image_list[self.current_index])
        os.remove(current_image_path)

        # Move to the next image
        self.current_index += 1
        self.label_entry.delete(0, tk.END)  # Clear the entry
        self.load_image()


if __name__ == "__main__":
    root = tk.Tk()
    root.title("Image Labeler")

    image_folder = "individual_letters"
    labeler = ImageLabeler(root, image_folder)

    root.mainloop()
