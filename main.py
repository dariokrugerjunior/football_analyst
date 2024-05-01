from utils import *
from trackers import Tracker
from speed_and_distance_estimator import SpeedAndDistanceEstimator

def main():
    # Lê os quadros de vídeo
    video_frames = read_video('input_videos/08fd33_0.mp4')

    # Instancia o objeto Tracker com um modelo pré-treinado especificado
    tracker = Tracker('models/best.pt')

    # Obtém o rastreamento dos objetos nos quadros do vídeo.
    # Caso não seja especificado para ler de um 'stub', processa o vídeo para detectar e rastrear objetos.
    tracks = tracker.get_object_tracking(video_frames, read_from_stub=True, stub_path='stubs/track_stubs.pkl')

    # Velocidade e Distancia estimada
    speed_estimator = SpeedAndDistanceEstimator()
    speed_estimator.add_speed_and_distance_to_tracks(tracks)

    # Aplica anotações gráficas nos quadros do vídeo com base nos rastreamentos obtidos
    output_video_frame = tracker.draw_annotations(video_frames, tracks)

    # Aplica anotações da velocidade e distancia
    speed_estimator.draw_speed_and_distance(output_video_frame, tracks)

    # Salva os quadros anotados como um novo arquivo de vídeo
    save_video(output_video_frame, 'output_videos/output_video.avi')
    
if __name__ == '__main__':
    main()