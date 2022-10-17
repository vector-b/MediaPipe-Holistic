BLESSED README FILE


Esse projeto tem como objetivo criar e aplicar um modelo preditivo em ações feitas por pessoas.
Aqui estaremos utilizando a biblioteca [MediaPipe](https://google.github.io/mediapipe/) focando na ferramenta [Holistic](https://google.github.io/mediapipe/solutions/holistic)
Com isso podemos ter a percepção em tempo real e simultânea da posição humana, marcas do rosto e rastreio das mãos.

![alt text](holistic_example.gif "Holistic Example")

Com isso, o objetivo desse projeto é classificar as posições em tempo real utilizando uma fonte de vídeo, sendo essa a webcam ou um video disponibilizado ao modelo.

Podemos utilizar **n** classes nesse modelo, cada classe será uma pasta contendo arquivos (imagens ou videos) dentro do diretório */gestures*

Para a execução do software é necessário instalar os pacotes básicos disponíveis em *requirements.txt*, para isso execute na raíz do projeto:
<pre><code>pip3 install -r requirements</code></pre>

Estando tudo em ordem, temos alguns arquivos que podem ser utilizados:

*   *holisticTrain.py* - Extração das features (pontos da face e do corpo) para um arquivo csv, o software irá extrair as Features de **TODOS** os arquivos da pasta escolhida, isso pode levar um tempo... 
    *   Opções de configurações suportadas:
        *    **path** - Endereço para onde devem ser encontradas as classes, o padrão é o diretórios */gestures*
        *    **filename** - Nome ou endereço do arquivo no qual o csv será gerado, o padrão é o arquivo *coords.csv*
*   *holisticRun.py* - Processo de treinamento utilizando um modelo csv gerado após o treinamento das classes e inicialização da classificação em tempo real utilizando um video de teste
    *   Opções de configuração suportadas:
        *   **filename** - Nome ou endereço do arquivo csv que será utilizado para o treinamento do modelo, esse arquivo deve ser previamente gerado utilizando *holisticTrain.py*
        *   **mode** - Tipo de entrada - 0 para WEBCAM e 1 para .MP4, caso seja definido como 0, videopath será ignorado.
        *   **videopath** - Endereço do vídeo em formato .mp4 que será utilizado para classificação em tempo real do modelo
        

**Exemplo de execução:**
<pre><code>python holisticTrain.py gestures coords.csv
python holisticRun.py coords.csv test_files/taxi_driver.mp4
</code></pre>

**IMPORTANTE**

**PRESSIONE ESC PARA SAIR DA EXECUÇÃO DO VÍDEO**

Neste projeto também está disponível uma DEMO, correspondente ao arquivo *holisticDemo.py*, basta executá-lo que um teste pré-definido será realizado:

<pre><code>python holisticDemo.py</code></pre>

A Demo tem dependência das pastas contidas neste repositório, portanto pode apresentar problemas no caso de remoção. Os arquivos atuais na pasta Gestures são exemplos (com 5 classes) retirados de videos no YouTube e podem ser removidos ou acrescentados de acordo com a preferência do usuário.

**DIVIRTA-SE!!**

![alt text](dancing.gif "DANCE")


TO-DO by priority

*  TSP01
*  TSP05
*  TSP03
*  TSP04
*  TSP02
