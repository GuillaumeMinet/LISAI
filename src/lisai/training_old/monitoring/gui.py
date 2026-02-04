import tkinter as tk
from tkinter import ttk
import json
import matplotlib.pyplot as plt
import matplotlib.backends.backend_tkagg as tkagg
import os
from pathlib import Path
_logfile = Path(os.getcwd()) / "src/training/monitoring/training_log.json"

class TrainingMonitor(tk.Tk):
    def __init__(self):
        super().__init__()

        self.log_file = _logfile
        self.title("Training Monitor")
        self.geometry("800x600")

        self.status_label = tk.Label(self, text="No training ongoing")
        self.status_label.pack()

        self.ongoing_listbox = tk.Listbox(self, height=10, width=50)
        self.ongoing_listbox.pack()

        self.plot_button = tk.Button(self, text="Show Loss Curve", command=self.show_loss_curve)
        self.plot_button.pack()

        self.after(1000, self.update_log)  # Update the log every second

    def update_log(self):
        try:
            with open(self.log_file, 'r') as file:
                logs = json.load(file)
                self.ongoing_listbox.delete(0, tk.END)
                for entry in logs:
                    status = entry['status']
                    if status == 'ongoing':
                        self.ongoing_listbox.insert(tk.END, f"Train {entry['train_id']} - Epoch {entry['epoch']} - Loss {entry['loss']}")

                # Refresh GUI every second
                self.after(1000, self.update_log)
        except Exception as e:
            print(f"Error reading log file: {e}")

    def show_loss_curve(self):
        try:
            with open(self.log_file, 'r') as file:
                logs = json.load(file)
            
            # Extract loss and validation loss for plotting
            epochs = [entry['epoch'] for entry in logs]
            losses = [entry['loss'] for entry in logs]
            val_losses = [entry['val_loss'] for entry in logs]

            # Plotting the loss curves
            fig, ax = plt.subplots()
            ax.plot(epochs, losses, label='Train Loss')
            ax.plot(epochs, val_losses, label='Validation Loss')
            ax.set_xlabel('Epochs')
            ax.set_ylabel('Loss')
            ax.legend()

            # Show the plot in Tkinter window
            canvas = tkagg.FigureCanvasTkAgg(fig, self)
            canvas.get_tk_widget().pack()
            canvas.draw()
        except Exception as e:
            print(f"Error plotting loss curve: {e}")

if __name__ == "__main__":
    app = TrainingMonitor()
    app.mainloop()
