import sys
import yaml
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QLineEdit, QPushButton, QVBoxLayout, QHBoxLayout, QFormLayout, QMessageBox, QCheckBox

class Menu(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()
    
    def read_file(self):
        with open(f'D:/multimodality/Trace/examples/config.yaml', 'r', encoding = 'utf-8') as file:
            self.config = yaml.safe_load(file)
    
    def write_file(self):
        with open(f'D:/multimodality/Trace/examples/config.yaml', 'w', encoding = 'utf-8') as file:
            yaml.safe_dump(self.config, file)
    
    def save(self):
        self.config["device_frame_rate"] = int(self.textboox1.text())
        self.config["mkv_file_path"] = str(self.textboox2.text())
        self.config["mkv_frame_rate"] = int(self.textboox3.text())
        self.config["playback_frame_rate"] = int(self.textboox4.text())
        self.config["playback_time"] = float(self.textboox5.text())
        self.config["speaker"] = self.textboox6.text()
        self.config["left_position"] = float(self.textboox7.text())
        self.config["middle_position"] = float(self.textboox8.text())
        self.config["gaze_window"] = float(self.textboox9.text())
        self.config["individual_gaze_event"] = float(self.textboox10.text())
        self.config["group_gaze_event"] = float(self.textboox11.text())
        self.config["gaze_beginning_buffer"] = float(self.textboox12.text())
        self.config["gaze_lookaway_buffer"] = float(self.textboox13.text())
        self.config["smooth_frame"] = int(self.textboox14.text())
        self.config["pose_window"] = float(self.textboox15.text())
        self.config["pose_positive_event"] = float(self.textboox16.text())
        self.config["pose_negative_event"] = float(self.textboox17.text())
        self.config["leanout_time"] = float(self.textboox18.text())
        self.config["update_check_interval"] = float(self.textboox19.text())
        self.config["gaze_positive_count_time"] = float(self.textboox20.text())
        self.config["gaze_negative_count_time"] = float(self.textboox21.text())
        self.config["posture_positive_count_time"] = float(self.textboox22.text())
        self.config["posture_negative_count_time"] = float(self.textboox23.text())
        self.config["draw_gaze_cone"] = self.checkbox1.isChecked()
        self.config["running_alive"] = self.checkbox2.isChecked()
        self.write_file()
        QMessageBox.about(self, "Parameters Saved", "Your parameters are successfully saved.")

    def initUI(self):
        self.read_file()

        self.setGeometry(200, 200, 800, 600)
        self.setWindowTitle("Experimental Setup")

        label1 = QLabel("Azure device frame rate (unit: frame/second)", self)
        self.textboox1 = QLineEdit(str(self.config["device_frame_rate"]), self)

        label2 = QLabel("MKV file path", self)
        self.textboox2 = QLineEdit(self.config["mkv_file_path"], self)

        label3 = QLabel("Frame rate of the MKV file (unit: frame/second)", self)
        self.textboox3 = QLineEdit(str(self.config["mkv_frame_rate"]), self)

        label4 = QLabel("Frame rate of reading the MKV file (unit: frame/second)", self)
        self.textboox4 = QLineEdit(str(self.config["playback_frame_rate"]), self)

        label5 = QLabel("How long to play the video from its beginning (unit: second)", self)
        self.textboox5 = QLineEdit(str(self.config["playback_time"]), self)

        label6 = QLabel("Pick out a participant who is speaking (from left to right, they are P1|P2|P3)", self)
        self.textboox6 = QLineEdit(self.config["speaker"], self)

        label7 = QLabel("Position of the divider that separates left participant and middle participant", self)
        self.textboox7 = QLineEdit(str(self.config["left_position"]), self)

        label8 = QLabel("Position of the divider that separates middle participant and right participant", self)
        self.textboox8 = QLineEdit(str(self.config["middle_position"]), self)

        label9 = QLabel("How long gaze history stored for each participant (unit: second)", self)
        self.textboox9 = QLineEdit(str(self.config["gaze_window"]), self)

        label10 = QLabel("Recorded as event when a participant did/didn't look at the speaker for ? seconds", self)
        self.textboox10 = QLineEdit(str(self.config["individual_gaze_event"]), self)

        label11 = QLabel("Recorded as event when two participants did/didn't look at the speaker for ? seconds", self)
        self.textboox11 = QLineEdit(str(self.config["group_gaze_event"]), self)

        label12 = QLabel("Buffer used when one participant switches from no gaze to gaze (unit: second)", self)
        self.textboox12 = QLineEdit(str(self.config["gaze_beginning_buffer"]), self)

        label13 = QLabel("Buffer used when one participant switches from gaze to no gaze (unit: second)", self)
        self.textboox13 = QLineEdit(str(self.config["gaze_lookaway_buffer"]), self)

        label14 = QLabel("How many frames one received feature can override the following None\nUsed to smooth features and reduce no information frame(unit: frame)", self)
        self.textboox14 = QLineEdit(str(self.config["smooth_frame"]), self)

        label15 = QLabel("How long posture history stored for each participant (unit: second)", self)
        self.textboox15 = QLineEdit(str(self.config["pose_window"]), self)

        label16 = QLabel("Report positive posture event ? seconds after last report", self)
        self.textboox16 = QLineEdit(str(self.config["pose_positive_event"]), self)

        label17 = QLabel("Report negative posture event ? seconds after last report", self)
        self.textboox17 = QLineEdit(str(self.config["pose_negative_event"]), self)

        label18 = QLabel("Recorded as a negative posture event if Leaning out for more than ? seconds", self)
        self.textboox18 = QLineEdit(str(self.config["leanout_time"]), self)

        label19 = QLabel("How frequently to update behavioral engagement level (unit: second)", self)
        self.textboox19 = QLineEdit(str(self.config["update_check_interval"]), self)

        label20 = QLabel("Gaze positive event count down time (unit: second)", self)
        self.textboox20 = QLineEdit(str(self.config["gaze_positive_count_time"]), self)

        label21 = QLabel("Gaze negative event count down time (unit: second)", self)
        self.textboox21 = QLineEdit(str(self.config["gaze_negative_count_time"]), self)

        label22 = QLabel("Posture positive event count down time (unit: second)", self)
        self.textboox22 = QLineEdit(str(self.config["posture_positive_count_time"]), self)

        label23 = QLabel("Posture negative event count down time (unit: second)", self)
        self.textboox23 = QLineEdit(str(self.config["posture_negative_count_time"]), self)

        self.checkbox1 = QCheckBox("Draw gaze cones on frame (after setting, please save)", self)
        self.checkbox1.setChecked(self.config["draw_gaze_cone"])

        self.checkbox2 = QCheckBox("If selected, run alive; If not selected, run play back video (after setting, please save)", self)
        self.checkbox2.setChecked(self.config["running_alive"])

        save_button = QPushButton("save", self)
        save_button.clicked.connect(self.save)

        run_button = QPushButton("run", self)
        run_button.clicked.connect(self.run_analysis)

        formlayout = QFormLayout()
        formlayout.addRow(label1, self.textboox1)
        formlayout.addRow(label2, self.textboox2)
        formlayout.addRow(label3, self.textboox3)
        formlayout.addRow(label4, self.textboox4)
        formlayout.addRow(label5, self.textboox5)
        formlayout.addRow(label6, self.textboox6)
        formlayout.addRow(label7, self.textboox7)
        formlayout.addRow(label8, self.textboox8)
        formlayout.addRow(label9, self.textboox9)
        formlayout.addRow(label10, self.textboox10)
        formlayout.addRow(label11, self.textboox11)
        formlayout.addRow(label12, self.textboox12)
        formlayout.addRow(label13, self.textboox13)
        formlayout.addRow(label14, self.textboox14)
        formlayout.addRow(label15, self.textboox15)
        formlayout.addRow(label16, self.textboox16)
        formlayout.addRow(label17, self.textboox17)
        formlayout.addRow(label18, self.textboox18)
        formlayout.addRow(label19, self.textboox19)
        formlayout.addRow(label20, self.textboox20)
        formlayout.addRow(label21, self.textboox21)
        formlayout.addRow(label22, self.textboox22)
        formlayout.addRow(label23, self.textboox23)
        formlayout.addRow(self.checkbox1)
        formlayout.addRow(self.checkbox2)

        hlayout = QHBoxLayout()
        hlayout.addWidget(save_button)
        hlayout.addWidget(run_button)

        vlayout = QVBoxLayout()
        vlayout.addLayout(formlayout)
        vlayout.addLayout(hlayout)

        #layout.addWidget(run_button)
        self.setLayout(vlayout)
    
    def run_analysis(self):
        reply = QMessageBox.information(self, "Confirmation", "Did you save your setting?", QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if reply == QMessageBox.Yes:
            exec(open(f"D:/multimodality/TRACE/examples/aaai_test.py").read())

if __name__ == "__main__":
    app = QApplication(sys.argv)
    menu = Menu()
    menu.show()
    sys.exit(app.exec_())