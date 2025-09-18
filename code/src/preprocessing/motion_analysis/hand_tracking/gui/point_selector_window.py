import pyvista as pv
from PyQt5.QtWidgets import (
    QWidget, QHBoxLayout, QVBoxLayout, QRadioButton, QGroupBox, QButtonGroup,
    QPushButton, QMessageBox, QSlider, QLabel
)
from PyQt5.QtCore import Qt, pyqtSignal
from pyvistaqt import QtInteractor


class PointSelectorWindow(QWidget):
    """
    A widget for selecting points on a 3D mesh.

    This window is non-modal and uses a signal to communicate the final
    set of selections back to its parent upon validation.
    """
    selectionsValidated = pyqtSignal(dict)

    def __init__(self, mesh: pv.PolyData, point_labels: list[str], existing_points: dict, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Point Selector")
        self.setGeometry(200, 200, 800, 600)

        self.setAttribute(Qt.WA_DeleteOnClose)

        self.mesh = mesh
        self.point_labels = point_labels
        self.selections = {}
        self.colors = ["yellow", "blue", "green", "red", "purple", "cyan", "magenta", "orange", "pink"]
        
        self.sphere_radius = 1.0

        # --- UI Setup ---
        self.main_layout = QHBoxLayout(self)
        self.controls_panel = QWidget()
        self.controls_layout = QVBoxLayout(self.controls_panel)
        self.labels_groupbox = QGroupBox("Select Point Label")
        self.labels_layout = QVBoxLayout()
        self.radio_button_group = QButtonGroup()

        for i, label in enumerate(self.point_labels):
            radio_button = QRadioButton(label)
            self.labels_layout.addWidget(radio_button)
            self.radio_button_group.addButton(radio_button, i)
        
        if self.radio_button_group.buttons():
            self.radio_button_group.buttons()[0].setChecked(True)

        self.labels_groupbox.setLayout(self.labels_layout)
        self.controls_layout.addWidget(self.labels_groupbox)

        # --- Sphere Size Slider ---
        self.size_label = QLabel("Sphere Size: " + str(self.sphere_radius))
        self.controls_layout.addWidget(self.size_label)

        self.size_slider = QSlider(Qt.Horizontal)
        self.size_slider.setRange(10, 300)  
        self.size_slider.setValue(int(self.sphere_radius * 100))
        self.size_slider.setToolTip("Adjust the size of the selection spheres.")
        self.size_slider.valueChanged.connect(self._update_sphere_size)
        self.controls_layout.addWidget(self.size_slider)

        # --- Buttons and Spacer ---
        self.controls_layout.addStretch()

        self.help_button = QPushButton("Help (?)")
        self.help_button.setToolTip("Show instructions on how to use the selector.")
        self.help_button.clicked.connect(self._show_help_message)
        self.controls_layout.addWidget(self.help_button)

        self.validate_button = QPushButton("Validate Selections")
        self.validate_button.setToolTip("Enabled when all points are selected.")
        self.validate_button.setEnabled(False)
        self.validate_button.clicked.connect(self._on_validate_and_close)
        self.controls_layout.addWidget(self.validate_button)

        # --- PyVista Plotter Setup ---
        self.plotter = QtInteractor(self)
        self.plotter.add_mesh(self.mesh, name="hand_mesh", color="tan", show_edges=True)
        self.plotter.iren.add_observer("LeftButtonPressEvent", self._on_left_click)

        self.main_layout.addWidget(self.controls_panel, 1)
        self.main_layout.addWidget(self.plotter.interactor, 4)

        self.load_existing_selections(existing_points)

    def _on_validate_and_close(self):
        """
        Emits the final selections and closes the window.
        This method is the designated "save" action.
        """
        final_points = {label: data['id'] for label, data in self.selections.items()}
        self.selectionsValidated.emit(final_points)
        self.close()

    def _show_help_message(self):
        """Displays a message box with instructions for using the point selector."""
        help_text = (
            "<b>How to Select Points:</b>"
            "<ul>"
            "<li>Choose the desired point label from the list on the left.</li>"
            "<li>Hold down the <b>Ctrl</b> key on your keyboard.</li>"
            "<li>While holding <b>Ctrl</b>, <b>left-click</b> on the mesh to select a vertex.</li>"
            "</ul>"
            "A colored sphere will mark your selection. You can update any "
            "selection by repeating the process for that label."
        )
        QMessageBox.information(self, "Help", help_text)

    def _update_sphere_size(self, value: int):
        """Callback for the size_slider. Updates the radius and redraws all spheres."""
        self.sphere_radius = value / 100.0
        self.size_label.setText(f"Sphere Size: {self.sphere_radius:.2f}")

        for label, data in self.selections.items():
            point_id = data['id']
            old_actor = data['actor']
            
            point_coords = self.mesh.points[point_id]
            color = old_actor.prop.color

            self.plotter.remove_actor(old_actor)
            
            new_actor = self.plotter.add_mesh(
                pv.Sphere(radius=self.sphere_radius, center=point_coords),  
                color=color
            )
            
            self.selections[label]['actor'] = new_actor
    
    def _on_left_click(self, interactor, event):
        """Callback for the 'LeftButtonPressEvent'."""
        if interactor.GetControlKey():
            click_pos = interactor.GetEventPosition()
            
            picker = interactor.GetPicker()
            picker.Pick(click_pos[0], click_pos[1], 0, self.plotter.renderer)

            hand_mesh_actor = self.plotter.actors.get("hand_mesh")

            if picker.GetActor() is hand_mesh_actor:
                pick_position = picker.GetPickPosition()
                closest_point_id = self.mesh.find_closest_point(pick_position)

                if closest_point_id >= 0:
                    self._handle_pick(closest_point_id)
        else:
            interactor.GetInteractorStyle().OnLeftButtonDown()

    def _handle_pick(self, point_id: int):
        """
        Processes a successful point pick. Updates the internal selection state
        and visualizes the picked point.
        """
        checked_button = self.radio_button_group.checkedButton()
        if not checked_button:
            return

        current_label = checked_button.text()
        label_index = self.point_labels.index(current_label)
        point_coords = self.mesh.points[point_id]

        if current_label in self.selections:
            old_actor = self.selections[current_label].get('actor')
            if old_actor:
                self.plotter.remove_actor(old_actor)

        color = self.colors[label_index % len(self.colors)]
        
        highlight_actor = self.plotter.add_mesh(
            pv.Sphere(radius=self.sphere_radius, center=point_coords), color=color
        )
        self.selections[current_label] = {'id': point_id, 'actor': highlight_actor}
        
        self._check_selection_completion()

    def load_existing_selections(self, existing_points: dict):
        """Loads and visualizes previously selected points passed during initialization."""
        for label, point_id in existing_points.items():
            if point_id is not None and label in self.point_labels:
                label_index = self.point_labels.index(label)
                point_coords = self.mesh.points[point_id]
                color = self.colors[label_index % len(self.colors)]
                
                highlight_actor = self.plotter.add_mesh(
                    pv.Sphere(radius=self.sphere_radius, center=point_coords), color=color
                )
                self.selections[label] = {'id': point_id, 'actor': highlight_actor}
        
        self._check_selection_completion()

    def _check_selection_completion(self):
        """Checks if all required point labels have a selection."""
        all_selected = len(self.selections) == len(self.point_labels)
        self.validate_button.setEnabled(all_selected)

    # --- ADDED METHOD ---
    def closeEvent(self, event):
        """
        Overrides the default close event to ensure a clean shutdown of the plotter.
        
        This releases graphics resources before the window is destroyed, preventing
        VTK OpenGL errors that occur when the renderer tries to access a context
        that is already being torn down.
        """
        if self.plotter:
            self.plotter.close()
        super().closeEvent(event)