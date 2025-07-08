import tkinter as tk
from tkinter import Button, Label
import numpy as np
from PIL import Image, ImageDraw
import pickle
from neural_network import NeuralNetwork # when using pickle, we need to import class
# Load pre-trained model (assuming the model is saved as 'network.p' file)
nn = pickle.load(open('../models/network.p', 'rb'))

class DrawingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Draw a Digit")
        
        # Canvas for drawing (28x28 grid with visible grid lines)
        self.canvas = tk.Canvas(self.root, width=280, height=280, bg="black")
        self.canvas.grid(row=0, column=0, padx=10, pady=10)
        
        self.canvas.bind("<B1-Motion>", self.paint)
        self.canvas.bind("<ButtonRelease-1>", self.on_release)
        
        # Image to store the drawing (28x28 grid)
        self.image = Image.new("L", (28, 28), color=0)  # 'L' mode for grayscale
        self.draw = ImageDraw.Draw(self.image)

        # Initialize previous mouse coordinates
        self.prev_x = None
        self.prev_y = None

        # Create buttons
        self.submit_button = Button(self.root, text="Submit", command=self.submit)
        self.submit_button.grid(row=1, column=0, pady=10)

        self.clear_button = Button(self.root, text="Clear", command=self.clear)
        self.clear_button.grid(row=2, column=0, pady=10)

        self.quit_button = Button(self.root, text="Quit", command=self.root.quit)
        self.quit_button.grid(row=3, column=0, pady=10)

        # Label to show prediction
        self.prediction_label = Label(self.root, text="Predicted Digit: None", font=("Arial", 14))
        self.prediction_label.grid(row=4, column=0, pady=10)

        # Set grid size for drawing (28x28 grid)
        self.grid_size = 28  # 28x28 grid
        self.cell_size = 280 / self.grid_size  # scale it to fit the canvas

        # Draw the grid on the canvas
        self.draw_grid()

    def draw_grid(self):
        """Draw the grid lines on the canvas."""
        for i in range(self.grid_size + 1):
            # Draw vertical grid lines
            self.canvas.create_line(i * self.cell_size, 0, i * self.cell_size, 280, fill="gray")
            # Draw horizontal grid lines
            self.canvas.create_line(0, i * self.cell_size, 280, i * self.cell_size, fill="gray")

    def paint(self, event):
        # Get coordinates relative to the grid
        x, y = event.x, event.y
        
        # Map the mouse coordinates to the grid
        grid_x = int(x // self.cell_size)
        grid_y = int(y // self.cell_size)

        # Gradually "whiten" the pixel
        if self.prev_x is not None and self.prev_y is not None:
            prev_grid_x = int(self.prev_x // self.cell_size)
            prev_grid_y = int(self.prev_y // self.cell_size)
            self.draw_line(prev_grid_x, prev_grid_y, grid_x, grid_y)

        # Draw on the canvas to show progress
        self.canvas.create_rectangle(grid_x * self.cell_size, grid_y * self.cell_size,
                                      (grid_x + 1) * self.cell_size, (grid_y + 1) * self.cell_size,
                                      fill="white", outline="white")
        
        # Update the previous coordinates
        self.prev_x, self.prev_y = x, y

    def draw_line(self, prev_grid_x, prev_grid_y, grid_x, grid_y):
        """Draw a line smoothly between previous and current grid cells."""
        line_length = max(abs(grid_x - prev_grid_x), abs(grid_y - prev_grid_y))

        # Prevent division by zero if the user does not move the mouse
        if line_length == 0:
            return

        for i in range(line_length + 1):
            inter_x = int(prev_grid_x + i * (grid_x - prev_grid_x) / line_length)
            inter_y = int(prev_grid_y + i * (grid_y - prev_grid_y) / line_length)
            
            # Ensure inter_x and inter_y are within the bounds of the 28x28 image
            inter_x = max(0, min(27, inter_x))  # Clamp between 0 and 27
            inter_y = max(0, min(27, inter_y))  # Clamp between 0 and 27
            
            self.image.putpixel((inter_x, inter_y), 255)  # Set pixel to white

    def on_release(self, event):
        """When the mouse is released, finalize the image data for prediction."""
        # Resize the image to 28x28
        self.image_resized = self.image.resize((28, 28), Image.LANCZOS)

        # Convert to numpy array and normalize (divide by 255)
        self.image_data = np.array(self.image_resized) / 255.0

    def center_and_resize_image(self, img):
        """Center the drawn image in the 28x28 grid, resize it to fit and normalize."""
        # Convert image to numpy array for processing
        img_array = np.array(img)

        # Find the non-zero pixels (bounding box)
        non_zero_pixels = np.argwhere(img_array > 0)

        if non_zero_pixels.size == 0:
            return img  # If no drawing, return the original image

        # Get the bounding box of the non-zero pixels
        min_y, min_x = non_zero_pixels.min(axis=0)
        max_y, max_x = non_zero_pixels.max(axis=0)

        # Crop the image to the bounding box
        cropped_img = img.crop((min_x, min_y, max_x + 1, max_y + 1))

        # Get the new image size (after cropping)
        cropped_width, cropped_height = cropped_img.size

        # Resize the cropped image to a smaller size (e.g., 20x20) if necessary
        max_size = 20  # We want the digit to occupy a small space, so limit it to 20x20
        if cropped_width > max_size or cropped_height > max_size:
            # Scale the image to fit within the max_size (20x20)
            cropped_img = cropped_img.resize((max_size, max_size), Image.LANCZOS)

        # Create a new 28x28 black image to place the cropped image in the center
        centered_img = Image.new("L", (28, 28), color=0)

        # Get the new image size
        new_width, new_height = cropped_img.size

        # Calculate where to paste the cropped (or resized) image
        offset_x = (28 - new_width) // 2
        offset_y = (28 - new_height) // 2

        # Paste the cropped image into the centered image
        centered_img.paste(cropped_img, (offset_x, offset_y))

        # Resize the final image to 28x28 (if necessary)
        centered_img = centered_img.resize((28, 28), Image.LANCZOS)

        return centered_img

    def submit(self):
        """Submit the drawn image to the model for prediction."""
        # Center and resize the image first
        centered_image = self.center_and_resize_image(self.image)

        # Convert to numpy array and normalize (divide by 255)
        self.image_data = np.array(centered_image) / 255.0

        # Predict the digit based on the image data (without using np.argmax)
        prediction = nn.predict(self.image_data.reshape(1, 784))  # Model expects (1, 784)
        
        # Show the prediction as a raw output (no argmax applied)
        self.prediction_label.config(text=f"Predicted Output: {prediction[0]}")
        # print(f"Predicted Output: {prediction}")

    def clear(self):
        """Clear the canvas and reset the image."""
        self.canvas.delete("all")
        self.image = Image.new("L", (28, 28), color=0)
        self.draw = ImageDraw.Draw(self.image)
        self.prev_x = None
        self.prev_y = None
        self.prediction_label.config(text="Predicted Digit: None")
        self.draw_grid()  # Redraw the grid
    
    def show(self):
        self.root.mainloop()

def main():
    """Main function to start the app."""
    # Create root window
    root = tk.Tk()
    app = DrawingApp(root)
    app.show()

if __name__ == "__main__":
    main()
