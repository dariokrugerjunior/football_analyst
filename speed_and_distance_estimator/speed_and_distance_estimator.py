import sys
import cv2
sys.path.append("../")
from utils import measure_distance, get_foot_position

class SpeedAndDistanceEstimator():
    def __init__(self):
        self.frame_window = 5
        self.frame_rate = 24

    def add_speed_and_distance_to_tracks(self, tracks):
        # Dicionário para armazenar a distância total percorrida por cada objeto e trilha.
        total_distance = {}

        # Itera sobre cada objeto e suas trilhas.
        for object, object_tracks in tracks.items():
            # Ignora objetos específicos como "ball" ou "referees".
            if object == "ball" or object == "referees":
                continue
            # Determina o número de quadros das trilhas do objeto.
            number_of_frames = len(object_tracks)
            # Itera sobre os quadros em janelas definidas por self.frame_window.
            for frame_num in range(0, number_of_frames, self.frame_window):
                # Calcula o último quadro da janela atual.
                last_frame = min(frame_num + self.frame_window, number_of_frames - 1)

                # Itera sobre cada identificador de trilha e suas informações no quadro inicial da janela.
                for track_id, _ in object_tracks[frame_num].items():
                    # Verifica se a trilha existe no último quadro da janela.
                    if track_id not in object_tracks[last_frame]:
                        continue

                    # Obtém a posição transformada nos quadros inicial e final da janela.
                    start_position = object_tracks[frame_num][track_id]['position_transformed']
                    end_position = object_tracks[last_frame][track_id]['position_transformed']

                    # Se as posições inicial ou final não estiverem definidas, ignora este cálculo.
                    if start_position is None or end_position is None:
                        continue

                    # Calcula a distância percorrida entre as duas posições.
                    distance_covered = measure_distance(start_position, end_position)
                    # Calcula o tempo decorrido entre os quadros em segundos.
                    time_elapsed = (last_frame - frame_num) / self.frame_rate
                    # Calcula a velocidade em metros por segundo.
                    speed_meteres_per_second = distance_covered / time_elapsed
                    # Converte a velocidade para quilômetros por hora.
                    speed_km_per_hour = speed_meteres_per_second * 3.6

                    # Registra a distância total percorrida para cada trilha.
                    if object not in total_distance:
                        total_distance[object] = {}
                    if track_id not in total_distance[object]:
                        total_distance[object][track_id] = 0
                    total_distance[object][track_id] += distance_covered

                    # Atualiza a velocidade e a distância total em cada quadro da janela para a trilha atual.
                    for frame_num_batch in range(frame_num, last_frame):
                        if track_id not in tracks[object][frame_num_batch]:
                            continue
                        tracks[object][frame_num_batch][track_id]['speed'] = speed_km_per_hour
                        tracks[object][frame_num_batch][track_id]['distance'] = total_distance[object][track_id]

    def draw_speed_and_distance(self, frames, tracks):
        # Lista para armazenar os quadros com os desenhos de velocidade e distância.
        output_frames = []
        # Itera sobre cada quadro e o respectivo número do quadro.
        for frame_num, frame in enumerate(frames):
            # Itera sobre cada objeto e suas trilhas nos dados de rastreamento.
            for object, object_tracks in tracks.items():
                # Ignora objetos que não devem ser exibidos (ex: "ball" ou "referees").
                if object == "ball" or object == "referees":
                    continue
                # Itera sobre as informações de trilha de cada objeto no quadro atual.
                for _, track_info in object_tracks[frame_num].items():
                    # Verifica se as informações de velocidade estão disponíveis na trilha.
                    if "speed" in track_info:
                        speed = track_info.get('speed', None)
                        distance = track_info.get('distance', None)
                        # Se a velocidade ou distância estiverem ausentes, ignora esta trilha.
                        if speed is None or distance is None:
                            continue

                        # Obtém a caixa delimitadora do objeto rastreado.
                        bbox = track_info['bbox']
                        # Calcula a posição dos pés com base na caixa delimitadora.
                        position = get_foot_position(bbox)
                        # Converte a posição para uma lista para ajustes.
                        position = list(position)
                        # Ajusta a posição vertical para não sobrepor o texto ao objeto.
                        position[1] += 40

                        # Converte a posição de volta para tupla e arredonda para inteiro.
                        position = tuple(map(int, position))
                        # Desenha o texto da velocidade no quadro.
                        cv2.putText(frame, f"{speed:.2f} km/h", position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
                        # Desenha o texto da distância logo abaixo do texto da velocidade.
                        cv2.putText(frame, f"{distance:.2f} m", (position[0], position[1] + 20),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
            # Adiciona o quadro modificado à lista de saída.
            output_frames.append(frame)

        # Retorna a lista de quadros com os desenhos de velocidade e distância.
        return output_frames



