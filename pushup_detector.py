import cv2
import mediapipe as mp
import numpy as np
import time

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
    
    def detect_pushups(self):
        """Main function to detect pushups using webcam"""
        cap = cv2.VideoCapture(0)
        
        # Check if camera opened successfully
        if not cap.isOpened():
            print("Error: Could not open camera")
            return
        
        print("Pushup Detector Started!")
        print("Press 'q' to quit")
        print("Position yourself in front of the camera and start doing pushups...")
        
        while cap.isOpened():
            ret, frame = cap.read()
            
            if not ret:
                print("Error: Could not read frame")
                break
            
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
            
            # Display instructions
            cv2.putText(image, 'Press q to quit', 
                       (50, 550), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Display the frame
            cv2.imshow('Pushup Detector', image)
            
            # Break loop on 'q' key press
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
        
        # Release resources
        cap.release()
        cv2.destroyAllWindows()
        
        # Return the final pushup count
        return self.pushup_count

def main():
    """Main function to run the pushup detector"""
    detector = PushupDetector()
    
    print("Starting Pushup Detection...")
    print("Make sure you have good lighting and are visible to the camera")
    print("Position yourself so that your side profile is visible for better detection")
    
    # Run detection
    final_count = detector.detect_pushups()
    
    print(f"\nFinal Pushup Count: {final_count}")
    print("Program ended.")

if __name__ == "__main__":
    main()
