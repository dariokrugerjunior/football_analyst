# Football Analyts

## Overview

`football_analyts` é um projeto de estudo desenvolvido para a análise de vídeos de jogos de futebol utilizando a tecnologia de detecção de objetos YOLOv5. O sistema identifica e rastreia jogadores, arbitros e a bola automaticamente..

##  

## Pre requisites

Antes de iniciar, você precisará das seguintes ferramentas:
- Python 3.8+
- Pip

## Installation

Siga os passos abaixo para configurar o ambiente de desenvolvimento:

```bash
# Instalar as dependências
pip install -r requirements.txt
````

## Project structure
```bash
football_analyts/
├── input_videos/   # Coloque aqui os vídeos de entrada para análise
├── models/         # Pasta para armazenar modelos treinados, incluindo best.pt
├── output_videos/  # Vídeos processados são salvos nesta pasta
├── runs/           # Resultados de execuções de treino/teste
├── stubs/          # Stubs para tipagem estática e linting
├── training/       # Scripts e notebooks de treinamento de modelo
├── utils/          # Utilitários diversos para apoio às funcionalidades do projeto
````

## Video Processing
Execute o script principal para iniciar a análise dos vídeos:
```bash
python main.py
````

## Training the Model
Para treinar o modelo utilizando o notebook Jupyter, siga os passos abaixo:

Inicie o Jupyter Notebook:
````bash
jupyter notebook
````

Isso abrirá a interface do Jupyter no seu navegador.

Navegue até o notebook: Localize e abra o arquivo football_training_yolo_v5.ipynb dentro da pasta training.

Execute as células do notebook: Siga as instruções dentro do notebook para treinar o modelo. Assegure-se de que o modelo best.pt seja salvo na pasta models/ após o treinamento.

## Contributing
Interessados em contribuir para o projeto podem fazê-lo através de pull requests ou abrindo issues para discussão de melhorias.


