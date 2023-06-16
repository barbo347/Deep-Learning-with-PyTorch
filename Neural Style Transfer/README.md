# Neural Style Transfer

Para acessar os dados utilizados: !git clone https://github.com/parth1620/Project-NST.git

O código a seguir implementa a técnica de Transferência de Estilo Neural (Neural Style Transfer) utilizando o modelo VGG-19 pré-treinado. A Transferência de Estilo Neural permite combinar o conteúdo de uma imagem com o estilo de outra imagem, criando uma nova imagem que preserva o conteúdo da imagem de entrada, mas é estilizada com os padrões e características visuais da imagem de estilo.

## Objetivos do projeto:

1. Carregar o modelo VGG-19 pré-treinado.
2. Extrair recursos de conteúdo e estilo das imagens.
3. Criar funções para calcular a perda de estilo e conteúdo.
4. Minimizar a perda total para gerar uma nova imagem com o estilo desejado.

## Passos do código:

1. Importar as bibliotecas necessárias, incluindo PyTorch, torchvision, PIL, numpy e matplotlib.
2. Carregar o modelo pré-treinado VGG-19.
3. Definir o dispositivo de execução (CPU ou GPU) com base na disponibilidade do CUDA.
4. Definir a função para pré-processamento de imagens, redimensionando-as e normalizando-as.
5. Carregar e pré-processar as imagens de conteúdo e estilo.
6. Definir a função para pós-processamento de imagens, revertendo as transformações aplicadas.
7. Exibir as imagens de conteúdo e estilo pré-processadas.
8. Extrair recursos de conteúdo e estilo das imagens usando o modelo VGG-19.
9. Calcular as matrizes gram das características de estilo.
10. Definir a função de perda de conteúdo para medir a diferença entre as características de conteúdo da imagem de destino e da imagem de conteúdo original.
11. Definir a função de perda de estilo para medir a diferença entre as matrizes gram das características de estilo da imagem de destino e da imagem de estilo original.
12. Definir a função de perda total que combina a perda de conteúdo e a perda de estilo ponderadas por fatores de escala.
13. Inicializar a imagem de destino como uma cópia da imagem de conteúdo e habilitar o cálculo de gradientes para otimização.
14. Configurar o otimizador Adam para ajustar a imagem de destino.
15. Executar o loop de treinamento para minimizar a perda total.
16. A cada intervalo de iterações definido, exibir a perda total atual e adicionar a imagem de destino desestilizada à lista de resultados.
17. Exibir as imagens resultantes em uma grade.

Este projeto permite explorar diferentes combinações de imagens de conteúdo e estilo, bem como ajustar os hiperparâmetros, como os pesos da perda de estilo e conteúdo, para criar imagens estilizadas personalizadas.


