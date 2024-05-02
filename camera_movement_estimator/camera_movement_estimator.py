import pickle
import cv2
import numpy as np
import os
import sys

sys.path.append('../')
from utils import measure_distance, measure_xy_distance


class CameraMovementEstimator():
    def __init__(self, frame):
        self.minimum_distance = 5

        self.lk_params = dict(
            winSize=(15, 15),
            maxLevel=2,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
        )

        first_frame_grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mask_features = np.zeros_like(first_frame_grayscale)
        mask_features[:, 0:20] = 1
        mask_features[:, 900:1050] = 1

        self.features = dict(
            maxCorners=100,
            qualityLevel=0.3,
            minDistance=3,
            blockSize=7,
            mask=mask_features
        )

    def add_adjust_positions_to_tracks(self, tracks, camera_movement_per_frame):
        # Itera sobre cada objeto rastreado e suas trilhas no conjunto de dados.
        for object, object_tracks in tracks.items():
            # Enumera cada quadro e suas respectivas informações de rastreamento.
            for frame_num, track in enumerate(object_tracks):
                # Itera sobre cada identificador de trilha e seus detalhes no quadro atual.
                for track_id, track_info in track.items():
                    # Recupera a posição atual do objeto rastreado.
                    position = track_info['position']
                    # Recupera o movimento da câmera para o quadro atual.
                    camera_movement = camera_movement_per_frame[frame_num]
                    # Ajusta a posição do objeto subtraindo o movimento da câmera.
                    position_adjusted = (position[0] - camera_movement[0], position[1] - camera_movement[1])
                    # Armazena a posição ajustada de volta na estrutura de dados de rastreamento.
                    tracks[object][frame_num][track_id]['position_adjusted'] = position_adjusted

    def get_camera_movement(self, frames, read_from_stub=False, stub_path=None):
        # Verifica se deve ler o movimento da câmera de um arquivo stub.
        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            # Abre o arquivo stub e carrega os dados de movimento da câmera.
            with open(stub_path, 'rb') as f:
                return pickle.load(f)

        # Inicializa a lista de movimentos da câmera com valores zero para cada quadro.
        camera_movement = [[0, 0]] * len(frames)

        # Converte o primeiro quadro para escala de cinza.
        old_gray = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
        # Detecta pontos característicos no primeiro quadro.
        old_features = cv2.goodFeaturesToTrack(old_gray, **self.features)

        # Processa cada quadro, começando do segundo até o último.
        for frame_num in range(1, len(frames)):
            # Converte o quadro atual para escala de cinza.
            frame_gray = cv2.cvtColor(frames[frame_num], cv2.COLOR_BGR2GRAY)
            # Calcula o fluxo óptico para encontrar novos pontos característicos.
            new_features, _, _ = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, old_features, None, **self.lk_params)

            # Inicializa a variável para rastrear a maior distância encontrada.
            max_distance = 0
            camera_movement_x, camera_movement_y = 0, 0

            # Itera sobre cada par de pontos antigos e novos.
            for i, (new, old) in enumerate(zip(new_features, old_features)):
                # Achata as coordenadas dos pontos para simplificar o acesso.
                new_features_point = new.ravel()
                old_features_point = old.ravel()

                # Calcula a distância entre os pontos antigos e novos.
                distance = measure_distance(new_features_point, old_features_point)
                # Atualiza o movimento da câmera se a distância for a maior encontrada.
                if distance > max_distance:
                    max_distance = distance
                    camera_movement_x, camera_movement_y = measure_xy_distance(old_features_point, new_features_point)

            # Se a maior distância encontrada for significativa, atualiza o registro de movimento para esse quadro.
            if max_distance > self.minimum_distance:
                camera_movement[frame_num] = [camera_movement_x, camera_movement_y]
                # Detecta novos pontos característicos no quadro atual para usar no próximo cálculo.
                old_features = cv2.goodFeaturesToTrack(frame_gray, **self.features)

            # Atualiza o quadro antigo para ser o atual para o próximo loop.
            old_gray = frame_gray.copy()

        # Se um caminho para o stub foi fornecido, salva os dados de movimento da câmera no arquivo.
        if stub_path is not None:
            with open(stub_path, 'wb') as f:
                pickle.dump(camera_movement, f)

        # Retorna a lista completa de movimentos da câmera.
        return camera_movement

    def draw_camera_movement(self, frames, camera_movement_per_frame):
        output_frames = []

        # Itera sobre cada quadro e o índice correspondente na lista de quadros.
        for frame_num, frame in enumerate(frames):
            # Cria uma cópia do quadro atual para evitar alterações no quadro original.
            frame = frame.copy()

            # Cria uma cópia do quadro para usar como overlay (camada sobreposta).
            overlay = frame.copy()
            # Adiciona um retângulo branco ao overlay para servir como fundo para o texto.
            cv2.rectangle(overlay, (0, 0), (500, 100), (255, 255, 255), -1)
            # Define a transparência do overlay.
            alpha = 0.6
            # Combina o overlay com o quadro original usando a transparência definida.
            cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

            # Extrai o movimento da câmera em X e Y para o quadro atual.
            x_movement, y_movement = camera_movement_per_frame[frame_num]
            # Adiciona texto ao quadro para mostrar o movimento da câmera em X.
            frame = cv2.putText(frame, f"Camera Movimento X: {x_movement:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                (0, 0, 0), 3)
            # Adiciona texto ao quadro para mostrar o movimento da câmera em Y.
            frame = cv2.putText(frame, f"Camera Movimento Y: {y_movement:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                (0, 0, 0), 3)

            # Adiciona o quadro modificado à lista de quadros de saída.
            output_frames.append(frame)

        # Retorna a lista de quadros modificados.
        return output_frames
