# REF-007: FADGI Audio ADC Performance - Especificações de Conversores Analógico-Digitais

| Metadado | Valor |
| :--- | :--- |
| **Código/Instituição** | FADGI Audio-Visual Working Group (2016 v1.1 & Setembro 2017 - Low Cost) |
| **Documentos Originais** | `pdfs_originais/ADC_performGuide_v1-1_20160216.pdf` e `ADC_Low_Cost_performGuide_2017-09-30.pdf` |
| **Destilado por** | Equipe Miniola / Antigravity |
| **Data de Resumo** | 2026-07-19 |

---

## 1. Escopo e Relevância para o Miniola
A FADGI estabelece métricas e métodos de ensaio rigorosos para conversores analógico-digitais (ADC) utilizados na preservação de áudio patrimonial, dividindo as especificações em duas categorias: **High Level Performance** (para estúdios e arquivos de referência) e **Low Cost Guideline** (para instituições com orçamento limitado, mas que exigem fidelidade de preservação sem distorções severas).

Para a **Miniola**, este documento orienta a nossa extração de som (`SPEC-003`) nas tolerâncias equivalentes exigidas para o processamento de som óptico por visão computacional em C++ (`miniola_cv.cpp`).

---

## 2. Parâmetros Críticos e Tabelas Comparativas (High Level vs. Low Cost)

A tabela abaixo sintetiza os limites de desempenho exigidos em testes padronizados para conversões em `24-bit / 96 kHz` (e mínimas em `48 kHz`):

| Parâmetro / Ensaio | FADGI High Level Performance (v1.1) | FADGI Low Cost Guideline (2017) | Relevância e Aplicação no Miniola |
| :--- | :--- | :--- | :--- |
| **1. Resposta em Frequência (*Frequency Response*)** | `20 Hz - 20 kHz`: `±0.1 dB`<br>`20 Hz - 40 kHz`: `±0.3 dB` (para `96 kHz`) | `20 Hz - 20 kHz`: `±0.5 dB`<br>`20 Hz - 40 kHz`: `±1.0 dB` | A extração óptica por câmera no Miniola é limitada pela largura da fenda em pixels (`slit_y`) e pela velocidade do filme. O nosso filtro óptico não deve atenuar frequências de voz (`100 Hz - 5 kHz`) além de `±0.5 dB`. |
| **2. Distorção Harmônica Total + Ruído (*THD+N*)** | `≤ -105 dB` (`0.00056%`) a 1 kHz (`-1 dBFS`) | `≤ -90 dB` (`0.00316%`) a 1 kHz (`-1 dBFS`) | A costura de quadros da pista óptica em C++ deve manter a distorção harmônica no limiar *Low Cost* (`≤ -90 dB`), garantindo varredura limpa sem zumbidos ou artefatos de junção. |
| **3. Alcance Dinâmico / Sinal-Ruído (*Dynamic Range / SNR*)** | `≥ 115 dB` (ponderado A ou não ponderado a `20-20 kHz`) | `≥ 100 dB` (ponderado A ou não ponderado a `20-20 kHz`) | Para a captação de som óptico (área variável ou densidade variável), a uniformidade de luz na fenda e a binarização do sensor devem garantir um piso de ruído de pelo menos `100 dB` abaixo do pico. |
| **4. Diafonia (*Cross-Talk*)** | `≤ -110 dB` entre canais (`20 Hz - 20 kHz`) | `≤ -85 dB` entre canais (`20 Hz - 20 kHz`) | Ao digitalizar trilhas ópticas duplas/bilaterais ou estéreo óptico, o vazamento de amostragem de pixels de um canal para o outro no sensor não pode ultrapassar `-85 dB`. |
| **5. Rejeição de Interferência Digital / Ruído** | `≥ 80 dB` (`20 Hz - 20 kHz`) | `≥ 60 dB` (`20 Hz - 20 kHz`) | Essencial para garantir que flutuações de luminosidade ou interferências eletromagnéticas dos motores de passo do Miniola não interfiram no sensor da câmera que lê a pista óptica. |

---

## 3. Diretrizes de Teste e Calibração para Som Óptico
- **Sinais de Teste (*Test Signals*)**: Os ensaios de conformidade da FADGI utilizam ondas senoidais puras de `1 kHz` a `-1 dBFS` para THD+N e varreduras (*sweeps*) logarítmicas de `20 Hz a 40 kHz` para resposta em frequência, que servem como padrão ouro para calibrar a decodificação da nossa câmera.
- **Iluminação Uniforme da Fenda**: Como o Miniola extrai o som exclusivamente por via **óptica** computacional (`SPEC-003`), a fenda luminosa LED sobre a trilha sonora deve ter brilho rigorosamente constante e sem cintilação (*flicker-free*), evitando que o ruído de alimentação USB/PWM se transforme em ruído na conversão de áudio.

---

## 4. Especificações de SDD Vinculadas (`specs/`)
- `[SPEC-003](../specs/003-optical-audio-extraction.md)` - Algoritmo de extração 1D/2D em C++ para garantir que a costura (*stitching*) de áudio óptico não introduza ruído harmônico ou saltos de fase que violem os limites de *THD+N*.
