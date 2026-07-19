# REF-004: FIAF Film Digitisation - Diretrizes e Melhores Práticas (2021)

| Metadado | Valor |
| :--- | :--- |
| **Código/Instituição** | FIAF Technical Commission (Film Digitisation Best Practices 2021) |
| **Documento Original** | `pdfs_originais/Film_Digitisation_2021.pdf` |
| **Destilado por** | Equipe Miniola / Antigravity |
| **Data de Resumo** | 2026-07-19 |

---

## 1. Escopo e Relevância para o Miniola
O documento *Film Digitisation* (2021) consolida as melhores práticas práticas da FIAF na montagem de laboratórios de digitalização arquivística. Ele abrange a escolha do elemento fílmico, controle de temperatura/umidade, inspeção física pré-varredura, mecânica de transporte, iluminação e o equilíbrio entre qualidade bruta e velocidade de captura.

Para o **Miniola**, este documento valida nossas decisões de engenharia de baixo custo focada em preservação: a tração sem pinos dentados (*sprocketless continuous capstan*), a iluminação LED difusa fria, a captura de toda a largura do suporte (*overscan* com perfurações) e a geração de metadados sidecar/DPX auditáveis.

---

## 2. Parâmetros Críticos e Tabelas de Referência

| Área / Parâmetro | Recomendação da FIAF (2021) | Aplicação e Alinhamento no Miniola |
| :--- | :--- | :--- |
| **1. Transporte Mecânico (*Film Transport*)** | Transporte suave com tensão controlada (*gentle tension control*) ou tracionamento contínuo sem pinos (*sprocketless/pinless*). É mandatório para filmes danificados, abaulados, com emendas antigas ou encolhimento superior a 1.0%. | O Miniola adota tração contínua macia e sincronização por **Visão Computacional em C++ (`miniola_cv.cpp`)**, eliminando qualquer engrenagem dentada que possa rasgar perfurações fragilizadas. **É importante ressaltar que o estado atual da Miniola usa um sistema de transporte manual da película com total controle do operador sobre a tensão, porém temos o plano de implementar uso de motores passo no futuro e criar um híbrido transporte manual/motoriazado*. Para filmes em bom estado poderemos usar o sistema motorizado, em caso de filmes muito deteriorados seá utilizado o transporte manual (por manivela em uma mesa de revisão). |
| **2. Fonte de Luz (*Illumination System*)** | Fonte de luz difusa e fria (LED de espectro contínuo com alto CRI/TLCI), regulada termicamente para evitar sobreaquecimento da película no gate. | Utilização de matriz LED difusa acionada na mesa, sem emissão térmica perigosa na faixa infravermelha ou ultravioleta (`[PRATICA-02]`). |
| **3. Inspeção e Preparação Pré-Scan** | Toda película deve passar por inspeção física prévia na mesa de revisão (*winding table*) para reparo de emendas quebradas, limpeza ultrassônica/manual de poeira e anotação de encolhimento e danos mecânicos. | Criamos o **Anotador Visual de Ground Truth (`tools/templates/anotador.html`)** para permitir que o arquivista documente exatamente o encolhimento, bitola e avarias da amostra antes ou durante a captura. |
| **4. Varredura Total (*Overscan to Edges*)** | Captura completa das bordas (*full aperture / edge-to-edge*) para registrar códigos de borda (*edge numbers*), carimbos de censura, marcas de emenda, marcas de som e a geometria da perfuração. | O crop da ROI de visão (`roiBox`) no nosso software é intencionalmente dimensionado para abranger as perfurações e bordas, preservando a integridade histórica do artefato fílmico. |
| **5. Metadados e Rastreabilidade** | Documentação técnica rigorosa associada a cada arquivo capturado (dados do scanner, lente, velocidade em FPS, sensor, data e operador). | O Miniola exporta arquivos sidecar `.json` e embutirá metadados nos cabeçalhos DPX em conformidade com as normas FADGI/SMPTE (`REF-005`). |

---

## 3. Diretrizes de Manuseio e Segurança do Suporte
- **Películas de Nitrato e Acetato Degradado (*Vinegar Syndrome*)**: Exigem tração mecânica com o mínimo de atrito no *gate*. Se a película apresentar ondulações severas (*buckling/fluting*), guias laterais com molas suaves devem manter o foco da lente sem esmagar o filme.
- **Velocidade de Varredura**: Embora scanners lineares modernos operem em altas velocidades (`24 FPS` a `120 FPS+`), a velocidade de digitalização deve ser reduzida imediatamente se a película apresentar fragilidade extrema, rasgos repetidos nas perfurações ou emendas soltas.

---

## 4. Especificações de SDD Vinculadas (`specs/`)
- `[SPEC-001](../specs/001-vision-engine-cpp.md)` - Rastreamento por fase circular tolerante a emendas e perfurações rasgadas.
- `[SPEC-004](../specs/004-multiprocessing-capture-pipeline.md)` - Desacoplamento da fila de gravação para garantir que o loop de tração mecânica não sofra solavancos por bloqueios de I/O em disco.
