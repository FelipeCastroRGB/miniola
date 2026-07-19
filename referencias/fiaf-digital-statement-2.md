# REF-003: FIAF Digital Statement Parte II - Digitalização para Preservação, Overscan e Resolução

| Metadado | Valor |
| :--- | :--- |
| **Código/Instituição** | FIAF Technical Commission (Digital Statement Part II - Maio 2022) |
| **Documento Original** | `pdfs_originais/The Digital Statement II FIAF` |
| **Destilado por** | Equipe Miniola / Antigravity |
| **Data de Resumo** | 2026-07-19 |

---

## 1. Escopo e Relevância para o Miniola
A Parte II do *Digital Statement* aprofunda a distinção entre **Digitalização para Preservação** (*Preservation Scans*) e **Digitalização para Acesso** (*Access Scans*). Ela enfatiza que nenhum scan digital é um "clone" exato da película fotoquímica original, e reforça a regra de ouro dos arquivos: **sempre preservar o elemento físico original (`Keep the Originals!`)**.

No ecossistema do **Miniola**, este documento fundamenta:
1. A necessidade de suporte a **Overscan de borda a borda (*Edge-to-Edge Overscan*)** para inspecionar perfurações, códigos de borda e estabilizar o registro por visão computacional (`miniola_cv.cpp`).
2. A clareza conceitual na separação entre o fluxo de preservação RAW/Log não comprimido e o fluxo de acesso (MP4/H.264 no painel web).

---

## 2. Parâmetros Críticos e Tabelas de Referência

| Parâmetro / Conceito | Recomendação da FIAF | Implementação / Impacto no Miniola |
| :--- | :--- | :--- |
| **1. Digitalização de Preservação vs. Acesso** | *Preservation Scans*: capturados do elemento original (OCN ou melhor cópia), em máxima resolução não comprimida (mínimo 2K/4K 10-bit log), exigindo trabalho posterior de cor/estabilização.<br>*Access Scans*: capturados com resolução e codificação otimizadas para consumo imediato (ex.: cópia já temporizada/graded). | O Miniola opera no modo de **Preservação RAW** capturando fotogramas inteiros com bordas na fila assíncrona (`fila_gravacao`), gerando simultaneamente um sub-fluxo de visualização rápida no dashboard SPA. |
| **2. Overscan (Varredura Além da Imagem)** | Recomendação enfática do uso de *Overscan* (abertura total incluindo janela da câmera, linha de quadro e bordas internas) ou *Edge-to-Edge Overscan* (cobrindo perfurações e marcações de fabricante/edge codes). | É a **espinha dorsal da detecção mecânica do Miniola**: nosso algoritmo C++ (*Phase-Locked Loop*) exige enxergar as perfurações nas bordas laterais para medir o pitch instantâneo e acionar o gatilho. |
| **3. Resolução Espacial para Overscan** | Se um filme é capturado com *overscan edge-to-edge* em 4K (4096px de largura total), a área da imagem interna terá apenas ~3600px. Para garantir 4K reais na imagem de um filme mudo/Standard, o scanner ideal deve ter resolução no sensor em torno de **4300 x 3956 px** ou superior. | Ao utilizar o módulo Ximea de `1420x880` ou Picamera2 em `1920x1080` para inspeção em tempo real, mantemos a ciência de que o crop útil da imagem (`cropBox` no Anotador) terá resolução ligeiramente menor que o frame total. A miniola, com as cameras atuais ainda não consegue atender resoluções em 2K ou 4K, mas gostaríamos de deixar a arquitetura pronta caso seja utilizada uma camera com tal capacidade de resolução de alta velocidade.|
| **4. Densidade (Curvas Sensitométricas)** | Adoção do fluxo **Cineon 10-bit log**, onde o preto base é posicionado em 10% acima do zero digital. Cada incremento de 1 código 10-bit equivale a `0.002 ND` (Densidade Neutra), totalizando uma faixa de `2.046 ND`. Para películas com alcance dinâmico extremo (ex.: Kodachrome ou reversíveis), recomenda-se dupla exposição combinada em arquivos 16-bit. | O pipeline nativo do Miniola deve evitar o corte bruto (*clipping*) dos valores de sombra em 8-bit linear durante a captura de negativos, dando preferência a buffers RAW/10-bit. |
| **5. Resolução e o Limite de 4K** | Há consenso de que **4K matematicamente não é suficiente** para extrair 100% da informação contida nos grãos de um Negativo Original de Câmera (NOX) de 35mm, sendo preferíveis sensores 6K/8K. No entanto, para gerações posteriores (dupe positives/negatives ou prints), 4K é amplamente aceito como suficiente. | A Miniola, sendo uma plataforma acessível de inspeção e preservação, almeja sensores 2K/4K como padrão de excelente equilíbrio entre fidelidade e taxa de transferência USB 3.0 em tempo real (`120 FPS+`). |

---

## 3. Diretrizes Teóricas: Granulação Fotográfica vs. Aliasing Digital

### 3.1. O que é o "Grão Fotográfico"?
A comissão técnica lembra (via John F. Hamilton, 1972) que o termo "grão" designa a percepção visual (*graininess*) das flutuações de densidade formadas pela distribuição aleatória e tridimensional dos cristais de prata ou nuvens de corante na emulsão.

### 3.2. O Problema do *Aliasing* (Serrilhado / Dança de Grão)
- A película fotoquímica, sendo um meio de amostragem não-uniforme e aleatório, **nunca sofre de *aliasing*** nativamente.
- O *aliasing* surge no momento em que a película é amostrada por uma grade espacial fixa e uniforme de pixels (o sensor da câmera digital).
- Se a frequência dos grãos finos da película for superior à metade da frequência de amostragem do sensor (Teorema de Nyquist), ocorre **Aliasing Espacial e Temporal** (*boiling grain* ou grãos que parecem "ferver", "dançar" e criar linhas coloridas falsas em projeção DCP).
- **Diretriz de Mitigação**: A resolução de amostragem deve ser a mais alta possível e os algoritmos de nitidez (*sharpening*) artificiais não devem ser aplicados na etapa de captura bruta (*raw scan*).

---

## 4. O que é um "Raw Scan" Realmente?
A FIAF destaca que o termo *"Raw Scan"* é impreciso: todo scanner já aplica internamente suposições, ganhos analógicos, tabelas de demosaicing (em sensores Bayer) ou reconstrução de linhas. Portanto, o arquivo bruto é simplesmente a versão digital **mais próxima possível do elemento físico, sem intervenções ativas de restauração ou color grading**, e deve ser preservado intacto como master de segurança antes de qualquer processamento posterior.

---

## 5. Especificações de SDD Vinculadas (`specs/`)
- `[SPEC-001](../specs/001-vision-engine-cpp.md)` - Uso do *Overscan* na detecção de contornos das perfurações laterais no C++.
- `[SPEC-007](../specs/007-ground-truth-annotator.md)` - Anotação explícita da ROI da área de perfurações (`roiBox`) vs. ROI da imagem limpa (`cropBox`).
