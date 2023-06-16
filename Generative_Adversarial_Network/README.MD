# Projeto GAN - README

Este projeto utiliza uma Rede Generativa Adversarial (GAN) para gerar imagens do conjunto de dados MNIST. 

## Objetivos do projeto:

1. Carregando o conjunto de dados MNIST em lotes usando o DataLoader.
    - Definir o tamanho do lote (batch size).
    - Utilizando a transformação `train_augs` para realizar o pré-processamento dos dados.
    - Verificar o número total de imagens presentes no conjunto de treinamento.

2. Criação da Rede Discriminadora.
    - Definir a função `get_disc_block` para criar um bloco da Rede Discriminadora.
    - Criar a classe `Discriminator` que herda de `nn.Module` e implemente sua estrutura.
    - Verificar um resumo da arquitetura da Rede Discriminadora.

3. Criação da Rede Geradora.
    - Definir a função `get_gen_block` para criar um bloco da Rede Geradora.
    - Criar a classe `Generator` que herda de `nn.Module` e implemente sua estrutura.
    - Verificar um resumo da arquitetura da Rede Geradora.

4. Inicialização dos pesos da Rede Discriminadora e da Rede Geradora.
    - Definir a função `weights_init` para inicializar os pesos das camadas.
    - Aplicar a inicialização dos pesos para as redes Discriminadora e Geradora.

5. Definição das funções de perda.
    - Definir as funções `real_loss` e `fake_loss` para calcular as perdas da Discriminadora.
    
6. Treinamento da GAN.
    - Utilizar um loop de treinamento para treinar o modelo GAN.
    - Definir o otimizador para a Discriminadora e a Geradora.
    - A cada iteração do loop de treinamento, realizar os seguintes passos:
      - Gerar ruído aleatório.
      - Calcular as perdas e atualize os pesos da Discriminadora.
      - Calcular as perdas e atualize os pesos da Geradora.
    - Ao final do treinamento, serão exibidos a época atual, a perda média da Discriminadora e a perda média da Geradora.

7. Visualização das imagens geradas.
    - Utilizando a função `show_tensor_images` para exibir algumas das imagens geradas pela GAN.

