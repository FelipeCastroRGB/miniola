# SPEC-005: Dashboard Web e Controle Interativo (Flask)

| Metadado | Valor |
| :--- | :--- |
| **ID da Especificação** | `SPEC-005` |
| **Status** | `Completed` |
| **Autor** | Equipe Miniola |
| **Data de Criação** | 2026-07-19 |
| **Última Atualização** | 2026-07-19 |

---

## 1. Contexto e Objetivo
Para possibilitar a inspeção e calibração de películas 35mm sem a necessidade de monitores físicos conectados diretamente ao hardware (modo *headless* em Raspberry Pi ou MiniPCs em rack/bancada), o Miniola fornece um dashboard web interativo servido via Flask (`miniola.py`). 

O dashboard permite visualização ao vivo com telemetria (perfurações detectadas, linhas de gatilho, pista de áudio em escala de cinza, histograma de luminância e alerta de exposição *Zebra*), bem como o ajuste visual da geometria de corte (Crop/ROI) e monitoramento térmico/sistema via rotas REST HTTP.

## 2. Requisitos Funcionais
- `[RF-01]`: O servidor Flask deve rodar no processo principal (`app = Flask(__name__)`), desativando logs verbosos de requisição do `werkzeug` (`log.setLevel(logging.ERROR)`) para não poluir o terminal de controle.
- `[RF-02]`: A rota de streaming ao vivo (`@app.route('/video_feed')` ou equivalente consumida pelo template) deve chamar `generate_dashboard()` em generator de Multipart MJPEG (`--frame\r\nContent-Type: image/jpeg\r\n\r\n`), emitindo quadros compostos a aproximadamente 15-20 FPS (`time.sleep(0.06)`).
- `[RF-03]`: O quadro do dashboard deve compor os seguintes painéis virtuais:
  - **Live View (Superior Esquerdo)**: Imagem redimensionada da câmera com retângulos da ROI de visão, ROI de áudio (amarelo) e linha de gatilho de perfuração (`LINHA_GATILHO_Y +- MARGEM_GATILHO`).
  - **Perfurações Binárias (Painel Esquerdo)**: Exibição da imagem binarizada gerada pelo motor C++ (`ultimo_frame_binario`).
  - **Pista de Áudio (Painel Direito)**: Visualização em escala de cinza da fenda de som (`AUDIO_READ_W`), permitindo conferir o foco óptico e o alinhamento da fenda.
  - **Zebra & Histograma (Inferior)**: Visualização de superexposição (vermelho se luma > 245) e subexposição (azul se luma < 10) em uma prévia do crop final, acompanhada de histograma em tempo real de 256 níveis.
- `[RF-04]`: A rota REST `@app.route('/set_crop')` deve receber parâmetros `x, y, w, h` via query string ou JSON do navegador, calcular as coordenadas relativas ao centro do furo de ancoragem (`OFFSET_X`, `OFFSET_Y_CROP`) e impor que `CROP_W` e `CROP_H` sejam sempre **números pares** para evitar falhas no subsampling de chroma `YUV420p` do codificador FFmpeg H.264.
- `[RF-05]`: A rota REST `@app.route('/status')` deve retornar JSON com uso de CPU, uso de RAM, temperatura da CPU, quantidade de quadros capturados no diretório de armazenamento e espaço livre em disco em MB.

## 3. Requisitos Não-Funcionais e Performance
- `[RNF-01]`: A geração do painel composto em `generate_dashboard()` deve redimensionar e converter a cor do último quadro bruto (`ultimo_frame_bruto`) em uma cópia independente ou usando `cv2.imencode('.jpg', dashboard, [IMWRITE_JPEG_QUALITY, 70])` sem bloquear o thread principal do `logica_scanner`.
- `[RNF-02]`: O consumo de banda de rede do MJPEG stream deve ser otimizado (qualidade JPEG 70 e resolução de exibição adaptada).

---

## 4. Matriz de Impacto Multi-Plataforma

| Plataforma | Comportamento Esperado / Restrições Específicas |
| :--- | :--- |
| **Raspberry Pi 5/4 (`arm64`)** | A leitura de temperatura do processador na rota `/status` acessa `/sys/class/thermal/thermal_zone0/temp`. |
| **Mac Mini / MiniPCs (`x86_64`)** | Em sistemas Linux x86_64 genéricos, a leitura térmica em `/sys/class/thermal/` pode não existir ou usar arquivos diferentes (ex.: `thermal_zone` da CPU Intel vs acpitz). A rota `/status` deve capturar graciosamente qualquer exceção (`try/except`) e retornar `0.0` em vez de gerar erro 500. |

---

## 5. Arquitetura e Design Técnico

### 5.1. Componentes e Arquivos Modificados
- `miniola.py`: Rotas `@app.route` (`/set_crop`, `/status`, `/video_feed`) e gerador `generate_dashboard()`.
- `templates/`: Arquivos HTML/JS que consomem o MJPEG feed e enviam requisições assíncronas para calibração de corte e controle do scanner.

### 5.2. Contrato da Rota `/status` (JSON)
```json
{
  "cpu_percent": 18.4,
  "ram_percent": 32.1,
  "cpu_temp_c": 54.2,
  "frames_captured": 1420,
  "disk_free_mb": 850.5
}
```

---

## 6. Critérios de Aceitação e Plano de Verificação

### 6.1. Verificação Automatizada (`tests/`)
- [x] O script de checagem de especificações (`check_specs.py`) confirma que a SPEC-005 cumpre todos os requisitos do template.

### 6.2. Verificação Manual / Operacional
- [x] Acessar o IP do host na porta configurada exibe o dashboard atualizando a cerca de 15 FPS, com telemetria visual das perfurações em verde/vermelho.
- [x] Arrastar o seletor de crop no navegador aciona a rota `/set_crop`, atualizando o console do terminal e forçando que largura e altura sejam pares.
