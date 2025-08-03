
import cv2
import numpy as np
from ultralytics import YOLO
import mediapipe as mp

def process_video_with_mediapipe(video_path, output_path):
    
    
    model = YOLO("yolov8m.pt")
    
   
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    mp_drawing = mp.solutions.drawing_utils

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    action_ball_id = None
    PROXIMITY_THRESHOLD = 50 # Max distance in pixels from foot to ball

    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        ball_tracking_results = model.track(frame, persist=True, tracker="bytetrack.yaml", classes=[32], verbose=False)
        
        boxes = ball_tracking_results[0].boxes.xyxy.cpu().numpy().astype(int)
        track_ids = ball_tracking_results[0].boxes.id
        
        
        ball_positions = {}
        if track_ids is not None:
            track_ids = track_ids.cpu().numpy().astype(int)
            for box, track_id in zip(boxes, track_ids):
                centroid = (int((box[0] + box[2]) / 2), int((box[1] + box[3]) / 2))
                ball_positions[track_id] = {'box': box, 'centroid': centroid}

       
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pose_results = pose.process(frame_rgb)

        foot_positions = []
        if pose_results.pose_landmarks:
            landmarks = pose_results.pose_landmarks.landmark
            
            left_heel = landmarks[mp_pose.PoseLandmark.LEFT_HEEL]
            right_heel = landmarks[mp_pose.PoseLandmark.RIGHT_HEEL]
            
            if left_heel.visibility > 0.5:
                foot_positions.append((int(left_heel.x * width), int(left_heel.y * height)))
            if right_heel.visibility > 0.5:
                foot_positions.append((int(right_heel.x * width), int(right_heel.y * height)))
            
            mp_drawing.draw_landmarks(frame, pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    
        action_ball_id = None
        min_dist = float('inf')

        if foot_positions and ball_positions:
            for track_id, ball_data in ball_positions.items():
                for foot_pos in foot_positions:
                    dist = np.linalg.norm(np.array(ball_data['centroid']) - np.array(foot_pos))
                    if dist < min_dist:
                        min_dist = dist
                        if min_dist < PROXIMITY_THRESHOLD:
                            action_ball_id = track_id
        
        
        for track_id, ball_data in ball_positions.items():
            box = ball_data['box']
            x1, y1, x2, y2 = box
            
            #  red dot for the action ball
            if track_id == action_ball_id:
                centroid = ball_data['centroid']
                cv2.circle(frame, centroid, 10, (0, 0, 255), -1)
                cv2.putText(frame, "Action Ball", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            #  green box for all other balls
            else:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

     
        out.write(frame)
    
 
 
    cap.release()
    out.release()
    pose.close()
    cv2.destroyAllWindows()
    print(f"Processing complete with Mediapipe. Output saved to {output_path}")

if __name__ == "__main__":
    
    process_video_with_mediapipe('test.mp4', 'output_mediapipe_1.mp4')
    process_video_with_mediapipe('test2.mp4', 'output_mediapipe_2.mp4')
