# REF-008: Kodak Essential Reference Guide for Filmmakers - Estrutura da Película e Emulsão

| Metadado | Valor |
| :--- | :--- |
| **Código/Instituição** | Eastman Kodak Company (Motion Picture Film Group) |
| **Documento Original** | `pdfs_originais/kodak-essential-reference-guide-for-filmmakers.pdf` |
| **Destilado por** | Equipe Miniola / Antigravity |
| **Data de Resumo** | 2026-07-19 |

---

## 1. Escopo e Relevância para o Miniola
O *Essential Reference Guide for Filmmakers* da Kodak é a Bíblia técnica da fabricação e estrutura de filmes cinematográficos. Ele detalha a composição física do suporte (Acetato vs. Poliéster), o comportamento dos cristais de haleto de prata na emulsão (*T-Grain*), curvas sensitométricas de densidade característica (curva H&D) e o sistema de identificação por códigos de borda (*Keykode*).

Para o **Miniola**, compreender a física das camadas de emulsão e o índice de refração dos suportes Kodak orienta a calibração de exposição e limiar (*Threshold*) da nossa binarização em C++ (`miniola_cv.cpp`), além de explicar por que certos filmes coloridos envelhecidos exigem ajuste de densidade diferencial ou luz difusa na mesa.

---

## 2. Parâmetros Críticos e Estrutura Física do Suporte

A Kodak categoriza as bases físicas das películas cinematográficas em dois grandes grupos:

| Propriedade / Camada | Triacetato de Celulose (*Acetate Base*) | ESTAR / Poliéster (*Polyester Base*) | Impacto e Tratamento no Miniola |
| :--- | :--- | :--- | :--- |
| **1. Espessura da Base (*Base Thickness*)** | Negativos e Reversíveis de Câmera: `~0.13 mm` (`5.2 mils`).<br>Cópias de Exibição: `~0.14 mm` (`5.5 mils`). | Negativos/Intermediários e Cópias ESTAR: `~0.10 mm` (`4.0 mils`) a `0.12 mm` (`4.7 mils`). | A variação de espessura de `0.10 mm` a `0.14 mm` altera sutilmente o plano focal. O nosso suporte de lente com rosca fina permite foco micrométrico ajustável (`foco_atual`). |
| **2. Resistência Mecânica e Tração** | Suporte que rasga com facilidade sob tensão. Ao envelhecer em ambientes úmidos, sofre hidrólise ácido-catalisada (*Síndrome do Vinagre*), causando encolhimento severo e empenamento (*buckling*). | Suporte extremamente resistente a rasgos (*High Tensile Strength*). Não sofre Síndrome do Vinagre nem encolhimento significativo com o tempo. | **Alerta Crítico**: Filmes ESTAR nunca devem passar por projetores ou scanners com pinos dentados travados, pois, em caso de engasgo, arrebentarão o próprio mecanismo em vez de rasgar o filme. O Miniola é ideal para ESTAR e Acetato por usar roletes lisos (*sprocketless*). |
| **3. Camada Antialo (*Rem-Jet / Anti-Halation Undercoat*)** | Camada preta de carbono no verso (*Rem-Jet*) em negativos de cor para evitar reflexos da luz na base durante a exposição na câmera, removida no banho alcalino inicial do laboratório. | Camada anti-halo e anti-estática transparente ou acinzentada embutida ou no verso, otimizada para transporte limpo em laboratórios de alta velocidade. | Se um filme não processado (ou mal lavado) contiver resíduos de *Rem-Jet* no verso, a leitura de luz por transparência na fenda do Miniola será obstruída. |
| **4. Códigos de Borda (*Keykode / Edge Numbers*)** | Código de barras latente e numeração humanamente legível gravados a cada 0.5 pé (`64 perfurações` em 35mm / `20 perfurações` em 16mm) com informações de lote, rolo e ano. | Idem, com sufixo ou marcação de suporte ESTAR (`E`). | Nosso Anotador e exportador DPX (`REF-005`) devem capturar e registrar essa numeração no campo `Prefix / Count` para rastreabilidade de montagem. |

---

## 3. Curvas Sensitométricas e Camadas de Emulsão
1. **Curva Característica (Hurter & Driffield - H&D)**:
   - Relaciona o Logaritmo da Exposição (`Log E`) no eixo X com a Densidade Óptica (`D`) no eixo Y.
   - Divide-se em 3 regiões: **Pé (*Toe*)** (sombras, menor contraste), **Trecho Linear (*Straight-Line Portion*)** (gama de contraste proporcional da imagem) e **Ombro (*Shoulder*)** (altas luzes, saturação dos cristais).
   - O nosso scanner deve capturar desde o piso do *Toe* (`Dmin` da máscara de cor laranja em filmes Kodak `5219/5207`) até o topo do *Shoulder* (`Dmax`), sem clipar o canal azul (que possui maior densidade na máscara integral).
2. **Cristais T-Grain (Kodak Vision/Vision3)**:
   - Cristais planos em formato tabular (*Tabular Grains*) que oferecem maior captura de luz por volume de prata, reduzindo a granulação em emulsões de alta sensibilidade (`500T`).

---

## 4. Especificações de SDD Vinculadas (`specs/`)
- `[SPEC-001](../specs/001-vision-engine-cpp.md)` - Algoritmo de binarização com limiar dinâmico que se adapta à densidade base (`Dmin`) do acetato ou poliéster.
- `[SPEC-007](../specs/007-ground-truth-annotator.md)` - Documentação da bitola, encolhimento e estado física (Síndrome do Vinagre / empenamento) com base nos parâmetros metrológicos da Kodak.
