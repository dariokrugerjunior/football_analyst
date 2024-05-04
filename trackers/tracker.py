from ultralytics import YOLO
import supervision as sv
import pickle
import cv2
import os
import sys
sys.path.append('../')
from utils import get_center_of_bbox, get_bbox_width, get_foot_position


class Tracker:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        self.tracker = sv.ByteTrack()

    def track(self, image):
        # Rastreia objetos na imagem usando o modelo YOLO
        return self.model(image)

    def add_position_to_tracks(self,tracks):
        print("Adding position to tracks...")
        for object, object_tracks in tracks.items():
            for frame_num, track in enumerate(object_tracks):
                for track_id, track_info in track.items():
                    bbox = track_info['bbox']
                    if object == 'ball':
                        position = get_center_of_bbox(bbox)
                    else:
                        position = get_foot_position(bbox)
                    tracks[object][frame_num][track_id]['position'] = position
        print("Position added to tracks!")

    def detect_frames(self, frames):
        # Detecta objetos em uma lista de frames
        batch_size = 20  # Define o tamanho do lote para processamento de frames
        detections = []  # Inicializa uma lista vazia para armazenar detecções
        for i in range(0, len(frames), batch_size):
            detections_batch = self.model.predict(
                frames[i:i + batch_size], conf=0.1)
            detections += detections_batch
        return detections  # Retorna a lista de detecções para todos os frames

    def get_object_tracking(self, frames, read_from_stub=False, stub_path=None):
        print("Getting object tracking...")
        # Obtém o rastreamento de objetos para os frames fornecidos
        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            # Se a flag read_from_stub for True e um caminho de stub for fornecido e o arquivo stub existir
            with open(stub_path, 'rb') as f:
                return pickle.load(f)  # Carrega os dados do stub e retorna

        detections = self.detect_frames(frames)  # Detecta objetos em cada frame

        tracks = {
            "player": [],
            "referees": [],
            "ball": []
        }

        for frame_num, detection in enumerate(detections):
            class_name = detection.names  # Obtém os nomes das classes detectadas

            class_name_inv = {v: k for k, v in class_name.items()}  # Inverte o mapeamento de classes

            detection_supervision = sv.Detections.from_ultralytics(
                detection)  # Converte as detecções para o formato supervision

            for object_ind, class_id in enumerate(detection_supervision.class_id):
                # Verifica se algum objeto é um "goalkeeper" e, se for, muda sua classe para "player"
                if class_name[class_id] == "goalkeeper":
                    detection_supervision.class_id[object_ind] = class_name_inv["player"]

            detection_with_tracking = self.tracker.update_with_detections(
                detection_supervision)  # Atualiza o rastreamento dos objetos detectados

            tracks["player"].append({})
            tracks["referees"].append({})
            tracks["ball"].append({})

            for frame_detection in detection_with_tracking:
                if frame_detection is None:
                    continue
                if frame_detection[0] is None:
                    continue
                bbox = frame_detection[0].tolist()  # Converte as coordenadas do retângulo delimitador para uma lista
                class_id = frame_detection[3]  # Obtém o ID da classe do objeto detectado
                track_id = frame_detection[4]  # Obtém o ID de rastreamento do objeto detectado

                if class_id == class_name_inv["player"]:
                    # Se a classe for um jogador, adiciona as informações do rastreamento ao dicionário de jogadores
                    tracks["player"][frame_num][track_id] = {"bbox": bbox}

                if class_id == class_name_inv["referee"]:
                    # Se a classe for um árbitro, adiciona as informações do rastreamento ao dicionário de árbitros
                    tracks["referees"][frame_num][track_id] = {"bbox": bbox}

            for frame_detection in detection_supervision:
                bbox = frame_detection[0].tolist()  # Converte as coordenadas do retângulo delimitador para uma lista
                class_id = frame_detection[3]

                if class_id == class_name_inv["ball"]:
                    # Se a classe for uma bola, adiciona as informações do rastreamento ao dicionário de bola
                    tracks["ball"][frame_num][1] = {"bbox": bbox}

        if stub_path is not None:
            # Se um caminho de stub for fornecido, salva os dados de rastreamento em um arquivo stub
            with open(stub_path, 'wb') as f:
                pickle.dump(tracks, f)
                
        print("Object tracking obtained!")
        return tracks  # Retorna os dados de rastreamento

    def draw_ellipse(self, frame, bbox, color, track_id=None):
        # Exibe o ID do rastreamento, útil para debugging
        # print("track_id", track_id)
        # Coordenada y do canto inferior do retângulo delimitador (bbox)
        y2 = int(bbox[3])

        # Obtém o centro x do bbox e ignora a coordenada y retornada
        x_center, _ = get_center_of_bbox(bbox)
        # Calcula a largura do bbox
        width = get_bbox_width(bbox)

        # Desenha uma elipse na imagem de entrada
        cv2.ellipse(frame,
                    center=(x_center, y2),  # Define o centro da elipse
                    axes=(int(width), int(0.35 * width)),  # Define os eixos maior e menor da elipse
                    angle=0.0,  # Sem rotação
                    startAngle=-45,  # Ângulo inicial para desenhar a elipse
                    endAngle=235,  # Ângulo final para desenhar a elipse
                    color=color,  # Cor da elipse
                    thickness=2,  # Espessura da linha
                    lineType=cv2.LINE_4  # Tipo de linha
                    )

        # Define as dimensões do retângulo para o texto do ID
        rectangle_width = 40
        rectangle_height = 20
        x1_rect = x_center - rectangle_width // 2  # Esquerda do retângulo
        x2_rect = x_center + rectangle_width // 2  # Direita do retângulo
        y1_rect = y2 - rectangle_height // 2  # Topo do retângulo
        y2_rect = y2 + rectangle_height // 2  # Base do retângulo

        # if track_id is not None:
        #     # Se um ID de rastreamento foi fornecido, desenha um retângulo preenchido
        #     cv2.rectangle(frame,
        #                   (x1_rect, y1_rect),  # Canto superior esquerdo
        #                   (x2_rect, y2_rect),  # Canto inferior direito
        #                   color,  # Cor do retângulo
        #                   cv2.FILLED)  # Preenchimento do retângulo

        #     # Ajusta a posição x do texto baseado no tamanho do ID do rastreamento
        #     x1_text = x1_rect + 12
        #     if track_id > 99:
        #         x1_text -= 10  # Ajusta para esquerda se o ID é de três dígitos

        #     # Coloca o texto do ID de rastreamento dentro do retângulo
        #     cv2.putText(frame,
        #                 f"{track_id}",
        #                 (int(x1_text), int(y1_rect + 15)),  # Posição do texto
        #                 cv2.FONT_HERSHEY_SIMPLEX,  # Fonte do texto
        #                 0.6,  # Tamanho da fonte
        #                 (0, 0, 0),  # Cor do texto (preto)
        #                 2  # Espessura do texto
        #                 )

        return frame  # Retorna o frame com as modificações aplicadas

    def draw_annotations(self, video_frames, tracks):
        print("Drawing annotations...")
        # Inicializa uma lista para armazenar os quadros de vídeo com anotações
        output_video_frames = []

        # Itera sobre cada quadro e seu índice do vídeo original
        for frame_num, frame in enumerate(video_frames):
            # Faz uma cópia do quadro atual para evitar modificar o original
            frame = frame.copy()

            # Acessa os dicionários de rastreamentos dos jogadores, árbitros e bola no quadro atual
            player_dict = tracks["player"][frame_num]
            referees_dict = tracks["referees"][frame_num]
            ball_dict = tracks["ball"][frame_num]

            # Para cada jogador rastreado no quadro atual, desenha uma elipse azul
            for track_id, player in player_dict.items():
                frame = self.draw_ellipse(frame, player["bbox"], (0, 0, 255), track_id)

            # Para cada árbitro rastreado, desenha uma elipse verde
            for _, referee in referees_dict.items():
                frame = self.draw_ellipse(frame, referee["bbox"], (0, 255, 0))

            # Para a bola rastreada, desenha uma elipse vermelha
            for _, ball in ball_dict.items():
                frame = self.draw_ellipse(frame, ball["bbox"], (255, 0, 0))

            # Adiciona o quadro anotado à lista de quadros de saída
            output_video_frames.append(frame)

        # Retorna a lista de quadros de vídeo com anotações
        print("Annotations drawn!")
        return output_video_frames