# REF-006: FADGI Digitizing Motion Picture Film - Fluxogramas de Captação e SOW

| Metadado | Valor |
| :--- | :--- |
| **Código/Instituição** | FADGI Audio-Visual Working Group (18 de Abril de 2016) |
| **Documento Original** | `pdfs_originais/FilmScan_PWS-SOW_20160418.pdf` |
| **Destilado por** | Equipe Miniola / Antigravity |
| **Data de Resumo** | 2026-07-19 |

---

## 1. Escopo e Relevância para o Miniola
O guia da FADGI explora a complexidade de digitalizar elementos fílmicos em arquivos federais dos EUA (como a Biblioteca do Congresso e o NARA), estabelecendo um modelo de **Declaração de Trabalho (*Statement of Work - SOW*)** para definir entradas, ações recomendadas e saídas digitais em três cenários arquetípicos: películas positivas com som óptico, películas com som magnético e películas silenciosas.

Para o **Miniola**, este documento orienta a nossa arquitetura de extração de som (`SPEC-003` e `SPEC-005`), estabelecendo como o áudio óptico deve ser amostrado em sincronia com o quadro de imagem e quais são as resoluções e arquivos de entrega esperados para pacotes master de preservação (*Preservation Master Packages*).

---

## 2. Fluxos Arquetípicos de Digitalização (*Three Examples*)

A FADGI padroniza os pacotes de saída (*Deliverables*) de acordo com a natureza do elemento físico de entrada:

### Exemplo 1: Película Positiva com Som Óptico (*Positive Film, Optical Sound*)
- **Entrada**: Cópia de exibição 16mm ou 35mm (*Projection Print*) com pista de som óptico de área variável (*Variable Area*) ou densidade variável (*Variable Density*).
- **Ação Recomendada de Captura**: 
  - Varredura da imagem e da pista de som simultaneamente ou em passes dedicados.
  - A imagem deve ser capturada em toda a largura útil, preferencialmente incluindo a pista de som em alta resolução geométrica para que a decodificação digital por software (*Optical Sound Decoding via Image*) seja viável e reversível.
- **Saídas Digitais (*Outputs*)**:
  1. **Preservation Master Image**: Sequência de arquivos DPX 10-bit Log (`SMPTE ST 268`) em resolução `2K` ou `4K`.
  2. **Preservation Master Audio**: Arquivo WAV não comprimido (`Broadcast WAVE - BWF`), `24-bit / 96 kHz` (ou `48 kHz` mínimo), sincronizado com o código de tempo da imagem.
  3. **Mezzanine / Intermediário**: Arquivo Apple ProRes 422 HQ ou 4444 unificando imagem e áudio sincronizado para edição/restauração.
  4. **Access Copy**: MP4 (H.264 / AAC) leve para visualização web.

### Exemplo 2: Película com Som Magnético (*Double-System or Composite Mag Sound*)
- **Entrada**: Película com pista magnética colada na borda (*Stripe*) ou som magnético em rolo separado sincronizado (*Sepmag / Fullcoat*).
- **Ação Recomendada**: Varredura por cabeçote magnético calibrado conforme curva de equalização da época, gerando BWF de `24-bit / 96 kHz` isolado de ruídos mecânicos de tração.
> **Nota de Escopo do Miniola**: A extração de som magnético é descrita aqui para completude da norma FADGI. No entanto, o **Miniola foca exclusivamente na extração de som óptico via visão computacional (`SPEC-003`)**, sem suporte nativo ou cabeçote magnético a princípio.

### Exemplo 3: Película Positiva Silenciosa (*Positive Film, Silent*)
- **Entrada**: Película 16mm, 35mm ou 8mm sem pista sonora (ex.: cinema mudo ou tomadas de arquivo/rushes).
- **Ação Recomendada**: Captura no modo *Overscan* de borda a borda para preservar anotações de montagem, tintura (*tinting/toning*) e marcações de cadência da câmera original na taxa nominal de quadros histórica (`14 FPS` a `24 FPS`).

---

## 3. Diretrizes para Extração de Áudio Óptico no Miniola (`SPEC-003`)
1. **Varredura Óptica 1D / 2D**:
   - Se o scanner realiza a leitura do áudio óptico através de uma câmera de área (como a Ximea do Miniola iluminando a coluna da pista sonora), a resolução horizontal da pista de som (`audio_w`) e a uniformidade de iluminação da fenda são críticas para evitar distorção harmônica (*THD*).
2. **Sincronia Imagem-Áudio (*Lip-Sync Advance*)**:
   - Em projetores e leitores analógicos 35mm, o cabeçote de som óptico fica posicionado **20 ou 21 quadros à frente** da janela de projeção da imagem (e **26 quadros à frente** em 16mm).
   - O nosso motor C++ (`miniola_cv.cpp`) ou pipeline assíncrono em Python deve registrar com precisão matemática o deslocamento temporal entre a fatia de áudio lida (`audio_chunk`) e o quadro visual correspondente na fila `fila_gravacao`.

---

## 4. Especificações de SDD Vinculadas (`specs/`)
- `[SPEC-003](../specs/003-optical-audio-extraction.md)` - Extração computacional de som óptico com costura contínua e compensação de velocidade do filme.
- `[SPEC-005](../specs/005-web-dashboard-routing.md)` - Roteamento dos pacotes de saída (*Deliverables*) para download na interface do usuário (RAW vs ProRes/MP4).
