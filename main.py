from utils import *
from trackers import Tracker

def main():
    # Lê os quadros de vídeo
    video_frames = read_video('input_videos/08fd33_0.mp4')

    # Instancia o objeto Tracker com um modelo pré-treinado especificado
    tracker = Tracker('models/best.pt')

    # Obtém o rastreamento dos objetos nos quadros do vídeo.
    # Caso não seja especificado para ler de um 'stub', processa o vídeo para detectar e rastrear objetos.
    tracks = tracker.get_object_tracking(video_frames, read_from_stub=True, stub_path='stubs/track_stubs.pkl')

    # Aplica anotações gráficas nos quadros do vídeo com base nos rastreamentos obtidos
    output_video_frame = tracker.draw_annotations(video_frames, tracks)

    # Salva os quadros anotados como um novo arquivo de vídeo
    save_video(output_video_frame, 'output_videos/output_video-2.avi')
    
if __name__ == '__main__':
    main()