import cv2
import mediapipe as mp
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox
import threading
import os
from PIL import Image, ImageTk

class PushupDetector:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils
        
        # Pushup counting variables
        self.pushup_count = 0
        self.stage = None  # 'up' or 'down'
        self.angle_threshold_up = 160  # Angle threshold for 'up' position
        self.angle_threshold_down = 90  # Angle threshold for 'down' position
        self.prev_stage = None
        
        # Video processing variables
        self.cap = None
        self.is_processing = False
        self.video_path = None
        
    def calculate_angle(self, a, b, c):
        """Calculate angle between three points"""
        a = np.array(a)  # First point
        b = np.array(b)  # Mid point
        c = np.array(c)  # End point
        
        radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
        angle = np.abs(radians * 180.0 / np.pi)
        
        if angle > 180.0:
            angle = 360 - angle
            
        return angle
    
    def process_video(self, update_callback=None, completion_callback=None):
        """Process video file for pushup detection"""
        if not self.video_path or not os.path.exists(self.video_path):
            if completion_callback:
                completion_callback("Error: Video file not found")
            return
        
        self.cap = cv2.VideoCapture(self.video_path)
        
        # Check if video opened successfully
        if not self.cap.isOpened():
            if completion_callback:
                completion_callback("Error: Could not open video file")
            return
        
        self.is_processing = True
        self.pushup_count = 0
        self.stage = None
        
        frame_count = 0
        total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        while self.cap.isOpened() and self.is_processing:
            ret, frame = self.cap.read()
            
            if not ret:
                break
            
            frame_count += 1
            
            # Resize frame for better performance
            frame = cv2.resize(frame, (800, 600))
            
            # Convert to RGB
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            
            # Make detection
            results = self.pose.process(image)
            
            # Convert back to BGR
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
            # Extract landmarks
            try:
                landmarks = results.pose_landmarks.landmark
                
                # Get coordinates for pushup detection
                # Using left side body parts (can be changed to right side)
                shoulder = [
                    landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                    landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value].y
                ]
                elbow = [
                    landmarks[self.mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                    landmarks[self.mp_pose.PoseLandmark.LEFT_ELBOW.value].y
                ]
                wrist = [
                    landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                    landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST.value].y
                ]
                hip = [
                    landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value].x,
                    landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value].y
                ]
                knee = [
                    landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                    landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE.value].y
                ]
                
                # Calculate angles
                elbow_angle = self.calculate_angle(shoulder, elbow, wrist)
                hip_angle = self.calculate_angle(shoulder, hip, knee)
                
                # Pushup counting logic
                # Check if body is straight (hip angle should be around 180)
                if hip_angle > 160:  # Body is relatively straight
                    # Check elbow angle for pushup positions
                    if elbow_angle > self.angle_threshold_up:
                        self.stage = "up"
                    elif elbow_angle < self.angle_threshold_down and self.stage == "up":
                        self.stage = "down"
                        self.pushup_count += 1
                        print(f"Pushup Count: {self.pushup_count}")
                
                # Visualize angles and stage
                cv2.putText(image, f'Elbow Angle: {int(elbow_angle)}', 
                           (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                cv2.putText(image, f'Hip Angle: {int(hip_angle)}', 
                           (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                cv2.putText(image, f'Stage: {self.stage}', 
                           (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(image, f'Pushups: {self.pushup_count}', 
                           (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                
            except Exception as e:
                # If landmarks are not detected, continue without error
                pass
            
            # Draw landmarks
            self.mp_drawing.draw_landmarks(
                image, 
                results.pose_landmarks, 
                self.mp_pose.POSE_CONNECTIONS,
                self.mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                self.mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
            )
            
            # Display progress
            progress = int((frame_count / total_frames) * 100) if total_frames > 0 else 0
            cv2.putText(image, f'Progress: {progress}%', 
                       (50, 550), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Call update callback with the processed frame
            if update_callback:
                update_callback(image, self.pushup_count, progress)
        
        # Release resources
        if self.cap:
            self.cap.release()
        
        self.is_processing = False
        
        # Call completion callback
        if completion_callback:
            completion_callback(None, self.pushup_count)
    
    def stop_processing(self):
        """Stop video processing"""
        self.is_processing = False
        if self.cap:
            self.cap.release()

class PushupDetectorGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Pushup Detector")
        self.root.geometry("1000x700")
        
        self.detector = PushupDetector()
        self.video_thread = None
        
        self.create_widgets()
        
    def create_widgets(self):
        # Title
        title_label = tk.Label(self.root, text="Pushup Detector", font=("Arial", 24, "bold"))
        title_label.pack(pady=10)
        
        # File selection frame
        file_frame = tk.Frame(self.root)
        file_frame.pack(pady=10)
        
        tk.Label(file_frame, text="Select Video File:", font=("Arial", 12)).pack(side=tk.LEFT, padx=5)
        self.file_path_var = tk.StringVar()
        tk.Entry(file_frame, textvariable=self.file_path_var, width=50, state="readonly").pack(side=tk.LEFT, padx=5)
        tk.Button(file_frame, text="Browse", command=self.browse_file).pack(side=tk.LEFT, padx=5)
        
        # Control buttons frame
        button_frame = tk.Frame(self.root)
        button_frame.pack(pady=10)
        
        self.start_button = tk.Button(button_frame, text="Start Detection", command=self.start_detection, 
                                     bg="green", fg="white", font=("Arial", 12), padx=20)
        self.start_button.pack(side=tk.LEFT, padx=10)
        
        self.stop_button = tk.Button(button_frame, text="Stop", command=self.stop_detection, 
                                    bg="red", fg="white", font=("Arial", 12), padx=20, state="disabled")
        self.stop_button.pack(side=tk.LEFT, padx=10)
        
        # Video display canvas
        self.canvas = tk.Canvas(self.root, width=800, height=600, bg="black")
        self.canvas.pack(pady=10)
        
        # Results frame
        results_frame = tk.Frame(self.root)
        results_frame.pack(pady=10)
        
        tk.Label(results_frame, text="Pushup Count:", font=("Arial", 14)).pack(side=tk.LEFT, padx=5)
        self.count_var = tk.StringVar(value="0")
        tk.Label(results_frame, textvariable=self.count_var, font=("Arial", 16, "bold"), fg="blue").pack(side=tk.LEFT, padx=5)
        
        # Progress bar
        progress_frame = tk.Frame(self.root)
        progress_frame.pack(pady=10)
        
        tk.Label(progress_frame, text="Progress:", font=("Arial", 12)).pack(side=tk.LEFT, padx=5)
        self.progress_var = tk.StringVar(value="0%")
        tk.Label(progress_frame, textvariable=self.progress_var, font=("Arial", 12)).pack(side=tk.LEFT, padx=5)
        
        # Instructions
        instructions = tk.Label(self.root, text="Instructions:\n1. Click 'Browse' to select a video file\n2. Click 'Start Detection' to process the video\n3. View results in the pushup count display", 
                               font=("Arial", 10), justify=tk.LEFT, fg="gray")
        instructions.pack(pady=10)
        
        # Status
        self.status_var = tk.StringVar(value="Ready to process video")
        tk.Label(self.root, textvariable=self.status_var, font=("Arial", 10), fg="blue").pack(pady=5)
        
    def browse_file(self):
        """Open file dialog to select video file"""
        file_path = filedialog.askopenfilename(
            title="Select Video File",
            filetypes=[
                ("Video files", "*.mp4 *.avi *.mov *.mkv *.wmv"),
                ("All files", "*.*")
            ]
        )
        
        if file_path:
            self.file_path_var.set(file_path)
            self.detector.video_path = file_path
            self.status_var.set(f"Selected file: {os.path.basename(file_path)}")
            
    def start_detection(self):
        """Start pushup detection on selected video"""
        if not self.file_path_var.get():
            messagebox.showerror("Error", "Please select a video file first")
            return
            
        if not os.path.exists(self.file_path_var.get()):
            messagebox.showerror("Error", "Selected file does not exist")
            return
            
        # Disable start button and enable stop button
        self.start_button.config(state="disabled")
        self.stop_button.config(state="normal")
        self.status_var.set("Processing video...")
        
        # Start processing in a separate thread
        self.video_thread = threading.Thread(
            target=self.detector.process_video,
            args=(self.update_frame, self.processing_complete)
        )
        self.video_thread.daemon = True
        self.video_thread.start()
        
    def stop_detection(self):
        """Stop pushup detection"""
        self.detector.stop_processing()
        self.start_button.config(state="normal")
        self.stop_button.config(state="disabled")
        self.status_var.set("Processing stopped by user")
        
    def update_frame(self, frame, count, progress):
        """Update GUI with current frame and count"""
        # Update count display
        self.count_var.set(str(count))
        
        # Update progress display
        self.progress_var.set(f"{progress}%")
        
        # Convert frame to RGB format
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Convert to PIL Image
        pil_image = Image.fromarray(rgb_frame)
        
        # Resize image to fit canvas while maintaining aspect ratio
        canvas_width = 800
        canvas_height = 600
        img_width, img_height = pil_image.size
        
        # Calculate new dimensions to maintain aspect ratio
        scale_width = canvas_width / img_width
        scale_height = canvas_height / img_height
        scale = min(scale_width, scale_height)
        
        new_width = int(img_width * scale)
        new_height = int(img_height * scale)
        
        # Resize image
        pil_image = pil_image.resize((new_width, new_height), Image.LANCZOS)
        
        # Convert to ImageTk format
        self.photo = ImageTk.PhotoImage(pil_image)
        
        # Update canvas
        self.canvas.create_image(
            (canvas_width - new_width) // 2,  # Center horizontally
            (canvas_height - new_height) // 2,  # Center vertically
            anchor=tk.NW,
            image=self.photo
        )
        
        # Update status periodically
        if progress % 10 == 0:
            self.status_var.set(f"Processing... {progress}% complete")
            
    def processing_complete(self, error=None, count=None):
        """Callback when processing is complete"""
        self.start_button.config(state="normal")
        self.stop_button.config(state="disabled")
        
        if error:
            self.status_var.set(f"Error: {error}")
            messagebox.showerror("Error", error)
        else:
            self.count_var.set(str(count))
            self.status_var.set(f"Processing complete! Final count: {count} pushups")
            messagebox.showinfo("Complete", f"Video processing complete!\nTotal pushups detected: {count}")

def main():
    """Main function to run the GUI pushup detector"""
    root = tk.Tk()
    app = PushupDetectorGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()
