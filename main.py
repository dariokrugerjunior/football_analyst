from utils import *
from trackers import Tracker

def main(): 
    video_frames = read_video('input_videos/08fd33_0.mp4')
    
    tracker = Tracker('models/best.pt')
    
    tracks = tracker.get_object_tracking(video_frames, read_from_stub=True, stub_path='stubs/track_stubs.pkl')
    
    output_video_frame = tracker.draw_annotions(video_frames, tracks)
    
    save_video(output_video_frame, 'output_videos/output_video.avi')
    
if __name__ == '__main__':
    main()