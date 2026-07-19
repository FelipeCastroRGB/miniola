# REF-002: FIAF Digital Statement Parte I - Práticas de Digitalização e Sensores

| Metadado | Valor |
| :--- | :--- |
| **Código/Instituição** | FIAF Technical Commission (Digital Statement Part I) |
| **Documento Original** | `pdfs_originais/The Digital Statement FIAF` |
| **Destilado por** | Equipe Miniola / Antigravity |
| **Data de Resumo** | 2026-07-19 |

---

## 1. Escopo e Relevância para o Miniola
A Parte I do *Digital Statement* da FIAF define as recomendações éticas e técnicas para a digitalização, restauração e acesso a coleções cinematográficas, em referência aos artigos 1.4, 1.5 e 1.7 do Código de Ética da FIAF. O objetivo central é que a reprodução digital seja o mais "sem perdas" possível, mantendo as propriedades fotográficas originais da película sem introduzir distorções mecânicas ou interpolações.

Para o **Miniola**, este documento orienta a escolha e justificação da arquitetura de sensores (câmera **Ximea USB 3.0** e **Picamera2**), evidenciando os compromissos entre velocidade, custo e fidelidade na reconstrução de cores (filtro mosaico Bayer vs. sensores monocromáticos com iluminação sequencial).

---

## 2. Parâmetros Críticos e Categorização de Sensores

A FIAF classifica as arquiteturas de captura dos scanners comerciais e protótipos de arquivo em três categorias principais:

| Tipo de Sensor | Funcionamento e Transporte | Vantagens | Desvantagens / Compromissos | Relação com o Miniola |
| :--- | :--- | :--- | :--- | :--- |
| **1. Area-based Sensors (Sensores de Quadro Completo Monocromáticos)** | Capturam a área total do fotograma parado (transporte intermitente, estilo *step printing*). O sensor é nativamente "cego para cores" e realiza 3 exposições sequenciais em R, G e B (ou flashes múltiplos HDR). | Fidelidade de cor perfeita, resolução espacial total real em R, G e B, alinhamento geométrico ideal e alto alcance dinâmico sem interpolação. | Transporte intermitente mais lento; custo de hardware muito mais elevado. | Representa o padrão ouro de preservação master. A Miniola pode no futuro buscar uma implementação com transporte por motores e uma camera monocromática. |
| **2. Line Sensors (Sensores de Linha / Varredura Contínua)** | Capturam linhas horizontais individuais enquanto o filme se move continuamente, remontando a imagem 2D por software (*continuous printing*). | Alta velocidade (tempo real ou superior); elimina o tempo de parada mecânica por quadro. | Sensível a vibrações, emendas grossas (*splice bumps*) e variações na tração mecânica, que causam ondulações horizontais na imagem se o encoder não for perfeito. | A tração contínua da Miniola exige sincronia precisa via encoder e fase PLL no OpenCV para evitar deformações verticais/horizontais na linha. |
| **3. Area Color Chips (Sensores de Cor com Filtro Bayer / Mosaico)** | Captura simultânea das cores R, G e B em um único disparo usando um mosaico de microfiltros de cor (Padrão Bayer: 50% G, 25% R, 25% B). | Captura muito rápida, excelente custo-benefício e mecânica simplificada (um único disparo por fotograma). | Subamostragem espacial de cor: um sensor 2K/4K Bayer captura apenas metade da resolução em verde e um quarto em vermelho e azul. A imagem final requer algoritmos de interpolação (*demosaicing/debayering*). | É o modo padrão das nossas câmeras `ximea` (módulo colorido) e `pi` (CSI). Por isso, salvamos o RAW ou efetuamos o debayering com máxima qualidade de interpolação. |

---

## 3. Diretrizes de Supressão de Riscos e Avarias
A FIAF recomenda técnicas físicas/ópticas para suprimir riscos superficiais da base do filme durante a digitalização, evitando que se tornem parte da informação da imagem, o que é superior à remoção por software post-facto:

1. **Wet Gates (Janela Úmida ou Aquário)**:
   - Preenchimento dos riscos da película com um líquido de índice de refração compatível ao do suporte fotográfico, eliminando o espalhamento de luz (*light scattering*) que torna os riscos visíveis.
   - Eficaz também contra manchas de mofo e decomposição. Requer espaçamento mecânico rigoroso na fenda para não degradar a nitidez da lente.
   - Será difícil conseguir um sistema de *wet gate* para a miniola, apenas a longo prazo e com desenvolvimento avançado.
2. **Luz Difusa (*Diffuse Illumination*)**:
   - Uso de fontes de luz não-especulares (espalhadas e difusas). Reduz drasticamente a visibilidade de riscos superficiais na base sem a complexidade e os riscos químicos dos líquidos de *wet gate*.
   - **Na Miniola**: Adotamos painéis de iluminação LED difusos de alto CRI na mesa/fenda ótica para minimizar o espalhamento de luz em riscos sem molhar a película (`[PRATICA-02]`).
3. **Mapeamento Infravermelho (*IR / Dirt-map Scanning*)**:
   - Para películas coloridas integrais cromogênicos com corantes transparentes ao IR (Eastmancolor/Fujicolor), um canal infravermelho extra detecta poeira e riscos superficiais para gerar uma máscara de erro (*dirt map*), permitindo que softwares de restauro isolem a intervenção às avarias.

---

## 4. Granulação Fotográfica vs. Ruído de Captura
- A película analógica possui dois tipos de "informação": o detalhe visual da cena e a textura dos grãos fotográficos de haleto de prata ou nuvens de corante.
- O escaneamento converte essa distribuição aleatória tridimensional em uma grade estática de pixels 2D. 
- A Miniola deve evitar compressão destrutiva temporal no momento da captura para preservar a textura aleatória autêntica do grão (*graininess*), em vez de borrá-lo ou transformá-lo em artefatos de compressão de bloco.

---

## 5. Especificações de SDD Vinculadas (`specs/`)
- `[SPEC-002](../specs/002-camera-abstraction.md)` - Abstração de câmeras (`ximea`, `pi`, `uvc`) e processamento de formatos RAW/Bayer em conformidade com as diretrizes da FIAF para sensores *Area Color Chips*.
- `[SPEC-004](../specs/004-multiprocessing-capture-pipeline.md)` - Loop de captura assíncrona que preserva os buffers de imagem originais sem compressão intermediária com perda.
