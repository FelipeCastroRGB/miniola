# Amostras e Banco de Dados Visual (`amostras/`)

Este diretório abriga o **banco de dados de imagens e fotogramas reais de película 35mm** que serve como *Ground Truth Visual* e massa de dados para testes unitários, calibração algorítmica e simulação do scanner em PC/Mac sem hardware conectado (`--camera mock`).

---

## Estrutura do Diretório

```text
amostras/
├── README.md                   # Este catálogo e guia
├── 000-template-metadata.json  # Modelo de metadados sidecar para cada imagem (.json)
├── fotogramas/                 # Quadros isolados de imagem + perfuração (PNG/JPG)
└── audio_optico/               # Amostras focadas em pistas de áudio (Densidade/Área Variável)
```

---

## O Poder dos Metadados Sidecar (`.json`) no SDD

Para que uma imagem em `amostras/fotogramas/filmcolors_1935_technicolor_01.png` seja mais do que apenas uma foto, ela deve ser acompanhada de um arquivo sidecar com o **mesmo nome e extensão `.json`** (ex.: `amostras/fotogramas/filmcolors_1935_technicolor_01.json`).

Esse JSON descreve as propriedades físicas da película original (bitola, encolhimento, tipo de áudio, estado de conservação) e, principalmente, as **coordenadas de Ground Truth (`cv_ground_truth`) para verificação automática no nosso motor C++ (`miniola_cv.cpp`) e nos testes (`tests/`)**!

---

## Catálogo de Amostras Ativas

| ID | Arquivo de Imagem | Origem / Acervo | Bitola / Furos | Áudio Óptico | Encolhimento |
| :--- | :--- | :--- | :--- | :--- | :--- |
| *(Exemplo)* `SMP-001` | `fotogramas/kodak_ks_padrao_01.png` | Teste Bancada Miniola | 35mm KS (4 furos) | Densidade Variável | 0.0% |

> **Como Alimentar**: Ao baixar exemplos de sites de referência (como o [FilmColors](https://filmcolors.org/)), escaneamentos próprios ou fotogramas de acervos parceiros, salve a imagem em `fotogramas/` e crie o `.json` correspondente usando `000-template-metadata.json`.

---

## Integração com o Driver Mock e CI
Quando o desenvolvedor executa no Mac Mini ou PC Linux:
```bash
python3 miniola.py --camera mock --sample fotogramas/filmcolors_1935_technicolor_01.png
```
O provedor `MockCameraProvider` carrega a imagem, lê o sidecar `.json`, ajusta automaticamente a ROI e transmite o fotograma em loop no dashboard Flask, permitindo validar o comportamento de detecção de perfurações e alertas *Zebra* em condições idênticas a um rolo real rodando no scanner!
