from utils import *
from trackers import Tracker
from speed_and_distance_estimator import SpeedAndDistanceEstimator
from view_trasformer import ViewTransformer
from camera_movement_estimator import CameraMovementEstimator

def main():
    print("Starting...")
    # Lê os quadros de vídeo
    video_frames = read_video('input_videos/08fd33_0.mp4')

    # Instancia o objeto Tracker com um modelo pré-treinado especificado
    tracker = Tracker('models/best.pt')

    # Obtém o rastreamento dos objetos nos quadros do vídeo.
    # Caso não seja especificado para ler de um 'stub', processa o vídeo para detectar e rastrear objetos.
    tracks = tracker.get_object_tracking(video_frames, read_from_stub=True, stub_path='stubs/track_stubs.pkl')

    tracker.add_position_to_tracks(tracks)

    camera_movement_estimator = CameraMovementEstimator(video_frames[0])
    camera_movement_per_frame = camera_movement_estimator.get_camera_movement(video_frames,
                                                                                read_from_stub=True,
                                                                                stub_path='stubs/camera_movement_stub.pkl')
    camera_movement_estimator.add_adjust_positions_to_tracks(tracks, camera_movement_per_frame)

    view_transformer = ViewTransformer()
    view_transformer.add_transformed_position_to_tracks(tracks)

    # Velocidade e Distancia estimada
    speed_estimator = SpeedAndDistanceEstimator()
    speed_estimator.add_speed_and_distance_to_tracks(tracks)

    # Aplica anotações gráficas nos quadros do vídeo com base nos rastreamentos obtidos
    output_video_frames = tracker.draw_annotations(video_frames, tracks)

    # Aplica anotações para o movimento da camera
    output_video_frames = camera_movement_estimator.draw_camera_movement(output_video_frames, camera_movement_per_frame)

    # Aplica anotações da velocidade e distancia
    output_video_frame = speed_estimator.draw_speed_and_distance(output_video_frames, tracks)

    # Salva os quadros anotados como um novo arquivo de vídeo
    save_video(output_video_frame, 'output_videos/output_video.avi')
    print("Finishing...")
    
if __name__ == '__main__':
    main()