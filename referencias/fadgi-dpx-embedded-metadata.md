# REF-005: FADGI DPX Embedded Metadata - Diretrizes para Cabeçalhos (`SMPTE ST 268`)

| Metadado | Valor |
| :--- | :--- |
| **Código/Instituição** | FADGI Audio-Visual Working Group (23 de Abril de 2019) / SMPTE ST 268-1 |
| **Documento Original** | `pdfs_originais/DPX_Embed_Guideline_20190423.pdf` |
| **Destilado por** | Equipe Miniola / Antigravity |
| **Data de Resumo** | 2026-07-19 |

---

## 1. Escopo e Relevância para o Miniola
O formato **DPX (*Digital Moving-Picture Exchange*)**, padronizado pela `SMPTE ST 268-1`, é o formato de arquivo padrão mundial para digitalização e preservação master de fotogramas de película cinematográfica sem compressão. As diretrizes da FADGI padronizam o preenchimento dos blocos de metadados binários fixos (*Generic Section* e *Motion Picture Section*) dentro do cabeçalho de cada arquivo DPX (os primeiros `2048 bytes`).

Para o **Miniola**, este documento fornece o mapa exato de quais campos binários e metadados ASCII devem ser populados quando exportarmos quadros individuais ou sequências DPX a partir da nossa fila assíncrona (`fila_gravacao`), garantindo interoperabilidade com arquivos nacionais e internacionais de cinema.

---

## 2. Blocos de Cabeçalho e Campos Críticos FADGI

A FADGI divide o cabeçalho DPX em duas grandes seções obrigatórias para preservação de patrimônio:

### 2.1. Seção Genérica do Arquivo (*File Information & Image Information*)
| Offset / Campo DPX | Tipo / Tamanho | Especificação FADGI / Prática Miniola |
| :--- | :--- | :--- |
| `Magic Number` (0x00) | 4 bytes (`SDPX` ou `XPDS`) | Define o *Byte Order* (`Big-Endian` ou `Little-Endian`). Miniola exporta em `Big-Endian` (`SDPX`). |
| `Image Offset` (0x04) | 4 bytes inteiro | Padrão fixado em `2048 bytes` (ou `8192 bytes` se incluir *User Data* longo). |
| `Version Header` (0x08) | 8 bytes ASCII | Deve ser preenchido como `V1.0` ou `V2.0` (sem espaços extras truncados com nulos). |
| `File Name` (0x24) | 100 bytes ASCII | Nome exato do arquivo (ex.: `SDK01473_20260719_0001024.dpx`). |
| `Creation Date/Time` (0x88) | 24 bytes ASCII | Formato ISO `YYYY:MM:DD:HH:MM:SS:LTZ` representando o instante real da captura no scanner. |
| `Creator` (0xA0) | 100 bytes ASCII | Nome da instituição e do sistema (ex.: `Miniola Scanner / Cinemateca`). |
| `Project Name` (0x104) | 200 bytes ASCII | Identificador do projeto de preservação ou do título do filme (`film_metadata.film_title`). |
| `Bit Depth` | Inteiro | Para preservação, padronizado em **10-bit Log** (`Cineon`) ou **12/16-bit Linear**. |

### 2.2. Seção Específica de Cinema (*Motion Picture Film Information Section*)
Esta seção do cabeçalho DPX armazena a proveniência fotoquímica e mecânica do fotograma:

| Campo DPX (Motion Picture) | Tamanho ASCII | Diretriz de Preenchimento do Miniola |
| :--- | :--- | :--- |
| `Film Mfr ID` | 2 bytes | Código de dois caracteres do fabricante da película (ex.: `01` para Kodak, `02` para Fuji, `03` para Agfa/Gevaert). Extraído via leitura do *Edge Code*. |
| `Film Type` | 2 bytes | Identificador da emulsão (ex.: `5247` ou `5222`). |
| `Offset in Perfs` | 2 bytes inteiro | Número do furo em relação ao quadro (ex.: `1` a `4` em 35mm). |
| `Prefix / Count` | 6 + 4 bytes | O número de borda (*Keykode / Edge Number*) legível na lateral do filme. |
| `Frame Rate of Original` | Float (4 bytes) | Taxa de quadros nominal da película de origem (ex.: `16.0` ou `24.0` FPS). |
| `Shutter Angle` | Float (4 bytes) | Em scanners digitais contínuos como o Miniola, preenchido como `360.0` ou `0.0`. |
| `Frame Identification` | 32 bytes ASCII | Identificador único serial da sequência ou rolo digitalizado (`rolo_01_reel`). |

---

## 3. Diretrizes de Empacotamento de Pixels (*Packing & Alignment*)
1. **Alinhamento de Palavra de 32 bits (*Method A vs. Method B*)**:
   - Para arquivos **10-bit log**, 3 canais R, G, B consomem 30 bits. Os 2 bits restantes em cada palavra de 32 bits (*DWORD*) não devem conter dados de imagem (*Padding bits* em zero).
   - O Miniola deve certificar que o empacotamento em C++ via `miniola_cv.cpp` ou bibliotecas Python de exportação DPX não divida amostras de um mesmo pixel através dos limites da palavra de 32 bits (`Packing Method 1`).
2. **Razão de Aspecto de Pixel (`Pixel Aspect Ratio`)**:
   - Deve ser explicitamente registrada como `1:1` (`Horizontal=1`, `Vertical=1`) para sensores digitais quadrados, evitando distorções anamórficas não intencionais na exibição.

---

## 4. Especificações de SDD Vinculadas (`specs/`)
- `[SPEC-004](../specs/004-multiprocessing-capture-pipeline.md)` - Estrutura de exportação dos quadros e metadados capturados na fila assíncrona.
- `[SPEC-007](../specs/007-ground-truth-annotator.md)` - Coleta interativa de metadados de bitola, encolhimento e fabricante, que alimentam diretamente os campos da *Motion Picture Section* do DPX.
