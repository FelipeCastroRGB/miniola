# REF-001: Padrões e Dimensões Geométricas da Bitola 35mm (SMPTE / ANSI)

| Metadado | Valor |
| :--- | :--- |
| **Código/Instituição** | SMPTE ST 139 (35mm Film - Perforated KS) & SMPTE ST 93 (BH) |
| **Documento Original** | Padrões Internacionais SMPTE / ISO Cinematografia |
| **Destilado por** | Equipe Miniola / Antigravity |
| **Data de Resumo** | 2026-07-19 |

---

## 1. Escopo e Relevância para o Miniola
O algoritmo de visão computacional em C++ (`miniola_cv.cpp`) detecta as perfurações na película 35mm em movimento para disparar o gatilho eletrônico (`capturar = true`) e medir o encolhimento do filme em tempo real. 

Conhecer as dimensões nominais padronizadas pela SMPTE/ANSI para as variações de perfuração (Kodak Standard - KS, Bell & Howell - BH e CinemaScope - CS) permite que o filtro geométrico de contornos do OpenCV (`0.2 < w/h < 2.5` e área em `200 < area < 10000`) isole perfeitamente um furo verdadeiro e ignore poeira, emendas de fita ou rasgos na borda da película.

---

## 2. Parâmetros Críticos e Tabelas de Referência

| Tipo de Perfuração | Largura (W) | Altura (H) | Proporção (W/H) | Aplicação Principal / História |
| :--- | :--- | :--- | :--- | :--- |
| **KS (Kodak Standard / Posi)** | 2.79 mm (`0.110"`) | 1.98 mm (`0.078"`) | **1.41** | Padrão mundial em cópias de exibição (positivos) de 35mm desde a década de 1920 até o fim do filme fotográfico. Cantos arredondados. |
| **BH (Bell & Howell / Neg)** | 2.79 mm (`0.110"`) | 1.85 mm (`0.073"`) | **1.51** | Usado principalmente em negativos de câmera e masters de laboratório pela alta precisão de pinos mecânicos. Cantos planos/ovais. |
| **CS (CinemaScope / Fox)** | 1.98 mm (`0.078"`) | 1.85 mm (`0.073"`) | **1.07** | Perfurações estreitas ("Fox holes") criadas nos anos 1950 para acomodar 4 pistas magnéticas de som surround ao lado da imagem anamórfica. |

### 2.1. Pitch Nominal (Distância Centro a Centro entre Furos)
O pitch é a distância percorrida na vertical entre o centro de uma perfuração e o centro da perfuração seguinte:

| Norma / Tipo de Estoque | Pitch Nominal em Milímetros | Pitch Nominal em Polegadas |
| :--- | :--- | :--- |
| **KS Long Pitch (Positivo Padrão)** | **4.750 mm** | `0.1870"` |
| **BH Short Pitch (Negativo Padrão)** | **4.740 mm** | `0.1866"` |
| **Filmes Antigos / Nitrato (Encolhidos)** | **4.600 mm a 4.700 mm** | Encolhimento de 1.0% a 3.0% |

---

## 3. Diretrizes Algorítmicas no Motor OpenCV (`miniola_cv.cpp`)
1. **Passo Padrão no Fotograma 35mm**: Exatamente **4 perfurações por fotograma de imagem** em formato Standard/Academy (razão pela qual `contador_perfs_ciclo >= 4` aciona a captura). Em formatos Techniscope (Super35 2-perf), o passo é de 2 perfurações.
2. **Razão de Aspecto (`w/h`) no Sensor**:
   - Para um furo KS (proporção nominal 1.41), se a câmera estiver perpendicular ao filme, a caixa delimitadora (`boundingRect`) em pixels terá largura superior à altura.
   - O filtro do nosso código aceita `0.2 < w/h < 2.5` para acomodar inclinações ou perfurações rasgadas/avariadas sem rejeitar o quadro.
3. **Calibração Dinâmica de Pitch (`pitch_padrao_px`)**:
   - Se 4.75 mm de película equivalem a `195.0 pixels` no sensor (com a lente em foco nominal `foco_atual = 14.5`), então **1 milímetro na película corresponde a exatamente 41.05 pixels**.
   - Qualquer desvio em que a média dos últimos pitches detectados seja inferior ao pitch nominal (`PITCH_PADRAO_PX`) reflete o encolhimento físico da película em tempo real:

```text
Fórmula de Encolhimento (%):
Encolhimento (%) = ((Pitch_Nominal_mm - Pitch_Medido_mm) / Pitch_Nominal_mm) * 100
Ou no espaço do sensor (pixels):
Encolhimento (%) = (1.0 - (pitch_medio / PITCH_PADRAO_PX)) * 100.0
```

---

## 4. Práticas de Conservação e Manuseio (Reflexo no Hardware)
- `[PRATICA-01]`: Películas com mais de 1.5% de encolhimento não podem ser projetadas ou digitalizadas em equipamentos com pinos mecânicos dentados (*sprockets*), pois rasgarão as perfurações. O Miniola elimina os pinos dentados através do tração contínua macia por cabrestante (*capstan*) e registro computacional via OpenCV.
- `[PRATICA-02]`: A iluminação na fenda deve ser fria (LED Diffuse Light), evitando dilatação térmica e expansão do pitch durante a inspeção.

---

## 5. Especificações de SDD Vinculadas (`specs/`)
- `[SPEC-001](../specs/001-vision-engine-cpp.md)` - Binarização, contornos de perfurações e cálculo do pitch e encolhimento.
- `[SPEC-007](../specs/007-ground-truth-annotator.md)` - Anotador visual para marcação manual das 4 perfurações e cálculo de pitch em pixels.
