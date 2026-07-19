# SPEC-007: Ferramenta Interativa de Anotação Visual e Geração de Ground Truth (`tools/anotador_amostras.py`)

| Metadado | Valor |
| :--- | :--- |
| **ID da Especificação** | `SPEC-007` |
| **Status** | `Completed` |
| **Autor** | Equipe Miniola |
| **Data de Criação** | 2026-07-19 |
| **Última Atualização** | 2026-07-19 |

---

## 1. Contexto e Objetivo
O fluxo de **Spec-Driven Development (SDD)** do Miniola depende do banco de amostras (`amostras/fotogramas/` e `amostras/audio_optico/`) como *Ground Truth* para testes unitários do motor C++ (`miniola_cv.cpp`) e simulação visual no painel (`--camera mock`).

Entretanto, criar manualmente os arquivos Sidecar JSON (`.json`) digitando coordenadas de pixel `[x, y, w, h]` ou calculando a distância média entre perfurações (`pitch`) na mão é um processo tedioso, demorado e sujeito a erros humanos. 

Esta especificação formaliza a criação do **Anotador Visual Interativo (`tools/anotador_amostras.py`)**, um servidor Flask independente rodando na porta **`5001`** que serve uma interface HTML5 Canvas moderna (`tools/templates/anotador.html`). O usuário seleciona a imagem, desenha a ROI e o corte visualmente com o mouse, marca os 4 furos para cálculo automático do pitch e salva o sidecar `.json` com um clique.

## 2. Requisitos Funcionais
- `[RF-01]`: O servidor Flask em `tools/anotador_amostras.py` deve rodar em porta dedicada (`5001` por padrão ou configurável via `--port`) sem interferir no scanner principal (`miniola.py` na porta 5000).
- `[RF-02]`: O servidor deve escanear as pastas `amostras/fotogramas` e `amostras/audio_optico` e disponibilizar a rota REST `GET /api/images` retornando JSON com todas as imagens encontradas (`.png`, `.jpg`, `.jpeg`) e um indicador booleano se cada imagem já possui seu respectivo `.json` gerado.
- `[RF-03]`: A rota `GET /api/image/<path:filepath>` deve entregar o arquivo de imagem e, opcionalmente, os dados do sidecar JSON existente para permitir re-editar anotações antigas.
- `[RF-04]`: A interface web em `tools/templates/anotador.html` deve renderizar a imagem sobre um elemento `<canvas>` interativo, suportando 3 ferramentas de desenho com feedback visual em tempo real:
  - **Ferramenta ROI de Visão (Retângulo Verde)**: Define onde as 4 perfurações se encontram no quadro (`caixa_delimitadora_roi_recomendada: [x, y, w, h]`).
  - **Ferramenta de Crop do Fotograma (Retângulo Azul)**: Define a área recomendada de corte para o vídeo final (`crop_roi: [x, y, w, h]`), forçando largura e altura serem números pares se desejado ou sugerindo ajuste.
  - **Ferramenta de 4 Perfurações (Pontos Amarelos com Linhas Conectadas)**: O usuário clica sequencialmente no centro dos 4 furos. A distância vertical média em pixels `(d1 + d2 + d3) / 3` deve ser calculada instantaneamente no JavaScript e inserida no campo `pitch_px_esperado`.
- `[RF-05]`: O painel lateral deve apresentar um formulário completo sincronizado com o schema `amostras/000-template-metadata.json`, permitindo preencher `origem`, `ano_estimado`, `processo_cor`, `estoque_filme`, `bitola`, `tipo_perfuracao` e `tipo_audio`.
- `[RF-06]`: Se valores como `pitch_nominal_mm` e `encolhimento_estimado_pct` forem deixados em branco no formulário, o payload JSON os registrará explicitamente como `null` conforme as convenções do projeto para imagens empíricas (como do FilmColors).
- `[RF-07]`: A rota `POST /api/save_json` deve receber o payload formatado e gravá-lo no disco com indentação (`indent=2`) no exato caminho `amostras/<filepath_sem_extensao>.json`.

## 3. Requisitos Não-Funcionais e Performance
- `[RNF-01]`: A interface de desenho HTML5 Canvas deve responder suavemente aos eventos `mousedown`, `mousemove` e `mouseup` sem engasgos visualizando imagens de alta resolução (2K a 4K).
- `[RNF-02]`: O código de backend Flask não deve alocar recursos pesados de OpenCV no startup, mantendo o consumo de memória abaixo de 50 MB para rodar tranquilamente em background.

---

## 4. Matriz de Impacto Multi-Plataforma

| Plataforma | Comportamento Esperado / Restrições Específicas |
| :--- | :--- |
| **Mac Mini / MiniPCs (`x86_64`)** | Ambiente ideal para a etapa de curadoria e anotação. O desenvolvedor ou arquivista roda `python3 tools/anotador_amostras.py`, abre o navegador no Mac/PC local, anota imagens de alta velocidade e faz o commit dos arquivos `.json` gerados diretamente no Git. |
| **Raspberry Pi 5/4 (`arm64`)** | A ferramenta também funciona perfeitamente rodando no próprio Raspberry Pi sem interface gráfica local (modo *headless*), permitindo que o usuário acesse o IP do Pi na porta `5001` a partir de qualquer computador ou tablet na mesma rede Wi-Fi para anotar imagens capturadas pelo próprio scanner. |

---

## 5. Arquitetura e Design Técnico

### 5.1. Componentes e Arquivos Modificados
- `tools/anotador_amostras.py`: Servidor Flask leve com 4 rotas (`/`, `/api/images`, `/api/image/<path>`, `/api/save_json`).
- `tools/templates/anotador.html`: Single-page application com HTML5 Canvas e CSS puro (com tema escuro *Dark Mode* ergonômico).

### 5.2. Contrato da Rota `POST /api/save_json`
**Payload Enviado:**
```json
{
  "filepath": "fotogramas/filmcolors_1935_technicolor_01.png",
  "data": {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "id": "SAMPLE-001",
    "titulo": "Anotação FilmColors 1935",
    "provenance": { ... },
    "geometry": { "pitch_nominal_mm": null, ... },
    "cv_ground_truth": {
      "caixa_delimitadora_roi_recomendada": [200, 15, 84, 830],
      "pitch_px_esperado": 194.33
    }
  }
}
```
**Resposta do Servidor (`200 OK`):**
```json
{
  "status": "success",
  "saved_path": "amostras/fotogramas/filmcolors_1935_technicolor_01.json"
}
```

---

## 6. Critérios de Aceitação e Plano de Verificação

### 6.1. Verificação Automatizada (`tests/` / `scripts/`)
- [x] O validador SDD `python3 scripts/check_specs.py` confirma a validade da `SPEC-007`.
- [x] O script `tools/anotador_amostras.py` passa por compilação sintática (`python3 -m py_compile tools/anotador_amostras.py`) sem erros de sintaxe ou dependência ausente.

### 6.2. Verificação Operacional
- [x] Ao acessar a rota `/api/images`, o servidor lista arquivos de imagem das subpastas em `amostras/` sem falhas.
- [x] Ao desenhar no canvas, calcular o pitch e enviar o formulário via POST `/api/save_json`, o arquivo `.json` é gerado fisicamente no disco no caminho especificado e com indentação limpa.
