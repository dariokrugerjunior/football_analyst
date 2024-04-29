from ultralytics import YOLO
import supervision as sv
import pickle
import cv2
import os
import sys
sys.path.append('../')
from utils import get_center_of_bbox, get_bbox_width


class Tracker:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        self.tracker = sv.ByteTrack()

    def track(self, image):
        # Tracks objects in the given image using the YOLO model
        # Returns the result of object tracking using YOLO model
        return self.model(image)

    def detect_frames(self, frames):
        # Detects objects in a list of frames
        batch_size = 20  # Defines the batch size for processing frames
        detections = []  # Initializes an empty list to store detections
        for i in range(0, len(frames), batch_size):
            # Processes frames in batches of batch_size
            # Performs object detection on the batch of frames
            detections_batch = self.model.predict(
                frames[i:i + batch_size], conf=0.1)
            # Appends detections of the batch to the overall detections list
            detections += detections_batch
        return detections  # Returns the list of detections for all frames


    def get_object_tracking(self, frames, read_from_stub=False, stub_path=None):
        
        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            with open(stub_path, 'rb') as f:
                return pickle.load(f)
        
        # Detect objects in each frame of the video
        detections = self.detect_frames(frames)

        tracks = {
            "player": [],
            "referees": [],
            "ball": []
        }

        # Iterate over each returned detection
        for frame_num, detection in enumerate(detections):
            # Get the class names of the detected objects
            class_name = detection.names

            # Invert the class mapping to get a mapping of class ID to class name
            class_name_inv = {v: k for k, v in class_name.items()}

            # Convert the detections to a specific format using sv.Detections.from_ultralytics
            detection_supervion = sv.Detections.from_ultralytics(detection)

            # Check if any object is a "goalkeeper" and if so, change its class to "player"
            for object_ind, class_id in enumerate(detection_supervion.class_id):
                if class_name[class_id] == "goalkeeper":
                    detection_supervion.class_id[object_ind] = class_name_inv["player"]

            # Update the tracking of the detected objects
            detection_with_tracking = self.tracker.update_with_detections(
                detection_supervion)

            tracks["player"].append({})
            tracks["referees"].append({})
            tracks["ball"].append({})

            for frame_detection in detection_with_tracking:
                bbox = frame_detection[0].tolist()
                class_id = frame_detection[3]
                track_id = frame_detection[4]

                if class_id == class_name_inv["player"]:
                    tracks["player"][frame_num][track_id] = {"bbox": bbox}

                if class_id == class_name_inv["referee"]:
                    tracks["referees"][frame_num][track_id] = {"bbox": bbox}

            for frame_detection in detection_supervion:
                bbox = frame_detection[0].tolist()
                class_id = frame_detection[3]

                if class_id == class_name_inv["ball"]:
                    tracks["ball"][frame_num][1] = {"bbox": bbox}
                    
        if stub_path is not None:
            with open(stub_path, 'wb') as f:
                pickle.dump(tracks, f)
            
        return tracks

    def draw_ellipse(self, frame, bbox, color, track_id = None):
        print("track_id", track_id)
        y2 = int(bbox[3])

        x_center, _ = get_center_of_bbox(bbox)
        width = get_bbox_width(bbox)

        cv2.ellipse(frame,
                    center=(x_center, y2),
                    axes=(int(width), int(0.35*width)),
                    angle=0.0,
                    startAngle=-45,
                    endAngle=235,
                    color=color,
                    thickness=2,
                    lineType=cv2.LINE_4
                    )

        rectangle_width = 40
        rectangle_height = 20
        x1_rect = x_center - rectangle_width//2
        x2_rect = x_center + rectangle_width//2
        y1_rect = y2 - rectangle_height//2
        y2_rect = y2 + rectangle_height//2

        if track_id is not None:
            cv2.rectangle(frame,
                          (x1_rect, y1_rect),
                          (x2_rect, y2_rect),
                          color,
                          cv2.FILLED)

            x1_text = x1_rect + 12
            if track_id > 99:
                x1_text -=10

            cv2.putText(frame,
                        f"{track_id}",
                        (int(x1_text), int(y1_rect + 15)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 0, 0),
                        2
                        )



        return frame
        
    
    def draw_annotions(self, video_frames, tracks):
        output_video_frames = []
        for frame_num, frame in enumerate(video_frames):
            frame = frame.copy()
            
            player_dict = tracks["player"][frame_num]
            referees_dict = tracks["referees"][frame_num]
            ball_dict = tracks["ball"][frame_num]
            
            for track_id, player in player_dict.items():
                frame = self.draw_ellipse(frame, player["bbox"], (0, 0, 255), track_id)
                
            for _, referee in referees_dict.items():
                frame = self.draw_ellipse(frame, referee["bbox"], (0, 255, 0))
            
            for _, ball in ball_dict.items():
                frame = self.draw_ellipse(frame, ball["bbox"], (255, 0, 0))
                
                
                
            output_video_frames.append(frame)
            
        return output_video_frames