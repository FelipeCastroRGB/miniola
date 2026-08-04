# Regras para o Assistente Gemini / Antigravity - Projeto Miniola

Ao atuar neste repositório (`miniola`), você DEVE seguir estritamente as diretrizes abaixo e consultar o arquivo principal `AGENTS.md` para referência completa.

## 1. Spec-Driven Development (SDD) Obrigatório
- **Consultar Normas e Referências (`referencias/`)**: Antes de desenhar algoritmos de visão ou propor novas especificações, consulte sempre os documentos em `referencias/` (padrões FIAF, SMPTE, manuais de sensores) para garantir alinhamento com as melhores práticas de preservação audiovisual.
- **Nunca implementar ou alterar funcionalidades no código C++/Python** sem antes verificar se existe uma especificação na pasta `specs/`.
- Se a funcionalidade for nova ou representar uma mudança estrutural/arquitetural, **crie ou edite primeiro o arquivo `specs/XXX-nome.md`** usando o template `specs/000-template.md`.
- Siga sempre o ciclo: Consulte (`referencias/`) -> Especifique (`specs/`) -> Revise Impacto Multi-Plataforma -> Implemente -> Valide (`check_specs.py` e `tests/`).

## 2. Arquitetura Multi-Plataforma (Raspberry Pi vs. MiniPCs x86_64)
- O Miniola roda tanto em Raspberry Pi 5 (`arm64`) quanto em MiniPCs / Mac Mini Late 2012 (`x86_64` Linux).
- Evite hardcoding exclusivo de Raspberry Pi (`pykms`, `/sys/class/thermal/`, caminhos `/home/felipe/miniola/capturas`). Use abstrações de plataforma ou detecção `platform.machine()`.
- Respeite o contrato modular em `cameras/` (`ximea`, `pi`, `uvc`, `mock`).

## 3. Restrições de Hardware e Performance
- Mantenha todo o processamento pesado de visão computacional de alto FPS (binarização, contornos de perfuração, pitch e gatilho) em **C++** via `pybind11` (`src/miniola_cv.cpp`).
- Nunca bloqueie o loop de captura de imagem com gravações em disco ou chamadas de codificação de vídeo (`ffmpeg`); utilize sempre a fila assíncrona `multiprocessing.Queue`.

## 4. Testes e Validação
Sempre verifique seu trabalho executando:
```bash
python3 scripts/check_specs.py
python3 -m unittest discover -s tests
```
