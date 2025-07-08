import matplotlib.pyplot as plt
from matplotlib.widgets import Button

class ImageNavigator:
    def __init__(self, images, predictions):
        """Initialize the navigator with images and predictions."""
        self.images = images  # List of images (2D numpy arrays)
        self.predictions = predictions  # Corresponding predictions
        self.current_idx = 0  # Initial image index

        # Create a figure for displaying a single image
        self.fig, self.ax = plt.subplots(figsize=(5, 5.6))

        # Add a message on the figure to guide the user
        self.fig.text(0.5, 0.03, 'Use left and right arrow keys to navigate (or use buttons)', 
                      ha='center', va='center', fontsize=12, color='black')

        # Display the first image
        self.update_image()
        
        # Create "Previous" and "Next" buttons
        axprev = plt.axes([0.05, 0.06, 0.2, 0.075])  # Positioning the "Previous" button
        axnext = plt.axes([0.75, 0.06, 0.2, 0.075])  # Positioning the "Next" button
        
        self.prev_button = Button(axprev, 'Previous', color='lightgoldenrodyellow', hovercolor='lightcoral')
        self.next_button = Button(axnext, 'Next', color='lightgoldenrodyellow', hovercolor='lightcoral')

        # Bind button events
        self.prev_button.on_clicked(self.prev_image)
        self.next_button.on_clicked(self.next_image)

        # Connect the key press event to the handler
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)

    def update_image(self):
        """Update the displayed image based on the current index."""
        # Clear the axis and plot the new image
        self.ax.clear()
        image = self.images[self.current_idx] * 255.0  # Rescale image to 0-255
        prediction = self.predictions[self.current_idx]

        # Custom title based on the prediction
        title = f"Test Image {self.current_idx + 1} | PREDICTION: {prediction}"
        
        self.ax.imshow(image, cmap='gray')
        self.ax.set_title(title)
        self.ax.axis('off')  # Hide axes for better visualization
        self.fig.canvas.draw()  # Redraw the canvas

    def prev_image(self, event):
        """Move to the previous image when the 'Previous' button is clicked."""
        self.current_idx = max(self.current_idx - 1, 0)
        self.update_image()

    def next_image(self, event):
        """Move to the next image when the 'Next' button is clicked."""
        self.current_idx = min(self.current_idx + 1, len(self.images) - 1)
        self.update_image()

    def on_key(self, event):
        """Handle key press events to navigate through images."""
        if event.key == 'right':  # Right arrow key: go to the next image
            self.current_idx = min(self.current_idx + 1, len(self.images) - 1)
        elif event.key == 'left':  # Left arrow key: go to the previous image
            self.current_idx = max(self.current_idx - 1, 0)

        self.update_image()  # Update the image displayed
    
    def show(self):
        plt.show()

def reshape_images(X):
    """Reshape the (28000, 784) array to a list of 28x28 images."""
    return [X[i].reshape(28, 28) for i in range(X.shape[0])]
