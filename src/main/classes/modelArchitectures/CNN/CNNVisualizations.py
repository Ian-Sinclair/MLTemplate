from src.main.classes.projectUtils import DirectoryUtil

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

from typing_extensions import Self

class CNNVisualizations():
    class Line_Plot_Handler:
        def __init__(self, x_label : str, y_labels : list[str], y_axis_title = "Loss", plot_title = "Loss Plot"):
            self.y_axis_title = y_axis_title
            self.plot_title = plot_title
            self.fig, self.ax = plt.subplots()
            self.x_label : str = x_label
            self.ax.set_xlabel(self.x_label)
            self.ax.set_ylabel(y_axis_title)
            self.ax.set_title(plot_title)
            self.ax.grid(True)
            self.y_labels = {label : [] for label in y_labels}
            self.data = {self.x_label: [], **self.y_labels}

        def update_plot(self, x_data : int, y_data : dict) -> Self:
            self.ax.clear()  # Clear the plot before updating

            self.ax.set_xlabel(self.x_label)
            self.ax.set_ylabel(self.y_axis_title)
            self.ax.set_title(self.plot_title)

            # Update plot data
            self.ax.grid(True)
            self.data[self.x_label].append(x_data)
            for label, item in y_data.items():
                if label in self.y_labels :
                    self.data[label].append(item)
                    self.ax.plot(self.data[self.x_label], self.data[label], label=label, linewidth=0.5)

            self.ax.legend()
            return self

        def show_plot(self) -> Self:
            plt.show()
            return self

        def save_plot(self, save_path : str, file_name : str = "loss_plot.jpg") -> Self:
            if save_path is None :
                return self
            if not DirectoryUtil.isValidDirectory(save_path):
                print(f"Directory Not Found at [{save_path}]")
                try : 
                    DirectoryUtil.promptToCreateDirectory(save_path) # Throws value error
                except ValueError as e:
                    print(e)
                    return self
            
            if DirectoryUtil.isProtectedFile(save_path,file_name) :
                print(f"ABORTING: File [{os.path.join(save_path, file_name)}] is protected - cannot overwrite file")
                return self
            
            self.fig.savefig(os.path.join(save_path,file_name))
            plt.close(self.fig)
            return self
