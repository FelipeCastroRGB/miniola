# REF-XXX: [Título Oficial da Norma / Manual / Boas Práticas]

| Metadado | Valor |
| :--- | :--- |
| **Código/Instituição** | Ex.: FIAF Technical Commission / SMPTE ST 139 / Manual Ximea |
| **Ano / Edição** | Ex.: 2019 (2ª Edição) |
| **Documento Original** | Ex.: `pdfs_originais/FIAF_Digital_Preservation_v2.pdf` (ou Link Oficial) |
| **Destilado por** | Nome do Colaborador ou Agente IA |
| **Data de Resumo** | AAAA-MM-DD |

---

## 1. Escopo e Relevância para o Miniola
*(Explique brevemente por que este documento é importante para o scanner. Ex.: "Este documento define os parâmetros recomendados de resolução espacial, profundidade de cor e tolerância a engasgos mecânicos na digitalização de películas de 35mm para preservação em arquivos públicos...")*

---

## 2. Parâmetros Críticos e Tabelas de Referência
*(Extraia e formate as tabelas, números e diretrizes que impactam o código, os sensores ou a visão computacional)*

| Parâmetro / Norma | Recomendação da Instituição | Aplicação Prática no Miniola |
| :--- | :--- | :--- |
| **Resolução de Captura (35mm)** | Mínimo de 2K (2048px largura) para inspeção/acesso; 4K para masterização | O sensor Ximea em corte (`RES_W=1420, RES_H=880`) atende à faixa de inspeção em tempo real. |
| **Espaço de Cor / Bit Depth** | 10-bit ou 12-bit RAW em espaço nativo do sensor para preservação | Captura RAW8/RAW12 decodificada sem interpolação destrutiva (`subsampling=0`). |
| **Estabilidade de Pitch** | Variação de registro vertical inferior a ±0.5 linha de varredura | O cálculo instantâneo em `miniola_cv.cpp` compensa variações dinâmicas de encolhimento. |

---

## 3. Diretrizes Algorítmicas e Fórmulas
*(Se a norma contiver fórmulas matemáticas ou lógicas de inspeção de danos, documente-as aqui)*

```text
Fórmula de Encolhimento (%):
Encolhimento (%) = ((Pitch_Nominal_mm - Pitch_Medido_mm) / Pitch_Nominal_mm) * 100
```

---

## 4. Práticas de Conservação e Manuseio (Reflexo no Hardware)
- `[PRATICA-01]`: A tração mecânica sobre as perfurações de filmes de nitrato ou acetato empenado deve ser minimizada. O Miniola não utiliza pinos de registro mecânicos furos-a-dentado, confiando no registro **óptico/computacional via OpenCV**.
- `[PRATICA-02]`: A iluminação (LED na mesa de luz) deve ser fria e de espectro uniforme, evitando aquecimento termal da película na fenda de captura.

---

## 5. Especificações de SDD Vinculadas (`specs/`)
Listagem das nossas especificações técnicas que implementam ou obedecem a esta referência:
- `[SPEC-001](../specs/001-vision-engine-cpp.md)` - Compensação de encolhimento e binarização.
- `[SPEC-003](../specs/003-optical-audio-extraction.md)` - Extração de áudio óptico em conformidade com tolerâncias de densidade/área.
