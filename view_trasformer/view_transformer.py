import numpy as np
import cv2


class ViewTransformer():
    def __init__(self):
        # Definição das dimensões da quadra de basquete.
        court_width = 68
        court_length = 23.32

        # Definição dos vértices da quadra na imagem pixelada.
        self.pixel_vertices = np.array([[110, 1035],
                                        [265, 275],
                                        [910, 260],
                                        [1640, 915]])

        # Definição dos vértices correspondentes na quadra real.
        self.target_vertices = np.array([
            [0, court_width],
            [0, 0],
            [court_length, 0],
            [court_length, court_width]
        ])

        # Conversão dos vértices para o tipo de dados apropriado.
        self.pixel_vertices = self.pixel_vertices.astype(np.float32)
        self.target_vertices = self.target_vertices.astype(np.float32)

        # Cálculo da matriz de transformação de perspectiva.
        self.perspective_transformer = cv2.getPerspectiveTransform(self.pixel_vertices, self.target_vertices)

    def transform_point(self, point):
        # Converte as coordenadas do ponto para inteiros.
        p = (int(point[0]), int(point[1]))
        # Verifica se o ponto está dentro da região delimitada pelo polígono.
        is_inside = cv2.pointPolygonTest(self.pixel_vertices, p, False) >= 0
        # Se o ponto estiver fora da região delimitada, retorna None.
        if not is_inside:
            return None

        # Redimensiona o ponto para o formato adequado para a transformação.
        reshaped_point = point.reshape(-1, 1, 2).astype(np.float32)
        # Aplica a transformação de perspectiva ao ponto.
        transformed_point = cv2.perspectiveTransform(reshaped_point, self.perspective_transformer)
        # Redimensiona o ponto transformado para o formato padrão.
        return transformed_point.reshape(-1, 2)

    def add_transformed_position_to_tracks(self, tracks):
        # Itera sobre cada objeto e suas trilhas dentro do dicionário de trilhas.
        for object, object_tracks in tracks.items():
            # Itera sobre cada quadro e sua respectiva trilha para o objeto.
            for frame_num, track in enumerate(object_tracks):
                # Itera sobre cada identificador de trilha e suas informações no quadro atual.
                for track_id, track_info in track.items():
                    # Obtém a posição ajustada armazenada na trilha.
                    position = track_info['position_adjusted']
                    # Converte a posição para um array NumPy para facilitar transformações.
                    position = np.array(position)
                    # Transforma a posição utilizando uma função definida pelo usuário.
                    position_transformed = self.transform_point(position)
                    # Verifica se a posição transformada não é nula.
                    if position_transformed is not None:
                        # Reduz a dimensão do array e converte para lista.
                        position_transformed = position_transformed.squeeze().tolist()
                    # Atualiza a trilha com a nova posição transformada.
                    tracks[object][frame_num][track_id]['position_transformed'] = position_transformed
