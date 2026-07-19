# Miniola - Diretrizes do Projeto e Regras para Agentes (AGENTS.md)

Este documento estabelece as regras de arquitetura, desenvolvimento e convenções para todos os colaboradores trabalhando no repositório **Miniola**.

---

## 1. O Mandamento Central: Spec-Driven Development (SDD)

O **Miniola** é um sistema de preservação audiovisual de precisão em tempo real. Qualquer alteração não planejada pode degradar a performance (taxa de quadros de captura, saturação de USB ou desvio no cálculo do pitch).

> **REGRA DE OURO**: Nenhuma nova feature, refatoração estrutural, alteração de algoritmo em C++ ou mudança em provedores de câmera/áudio pode ser implementada sem **antes criar ou atualizar a Especificação (`Spec`) correspondente na pasta `specs/`**.

### O Fluxo Obrigatório de Trabalho:
1. **Consultar Normas e Referências (`referencias/`)**: Antes de desenhar ou alterar algoritmos, verifique os documentos técnicos (FIAF, SMPTE, manuais de sensores e scanners) na pasta `referencias/` para garantir alinhamento com as boas práticas de preservação audiovisual.
2. **Verificar Specs Existentes (`specs/`)**: Antes de tocar em qualquer código, consulte `specs/` para entender o comportamento atual da funcionalidade.
3. **Especificar (`specs/XXX-nome.md`)**: Se for uma nova funcionalidade ou alteração arquitetural, crie a especificação usando `specs/000-template.md` ou atualize uma especificação existente.
4. **Revisar Impacto Multi-Plataforma**: Verifique como a mudança se comporta em Raspberry Pi (`arm64`) e em MiniPCs/Mac Mini (`x86_64`).
5. **Implementar**: Escreva o código C++/Python seguindo estritamente a especificação.
6. **Validar & Certificar**: Execute o script `check_specs.py` e os testes automatizados em `tests/` antes de concluir.

---

## 2. Arquitetura Multi-Plataforma (RPi 5/4 vs. MiniPCs x86_64)

O Miniola foi projetado para operar sem modificação manual em duas classes distintas de hardware:

| Aspecto | Raspberry Pi 5/4 (`arm64`) | Mac Mini Late 2012 / MiniPCs (`x86_64`) |
| :--- | :--- | :--- |
| **Câmeras Suportadas** | `pi` (picamera2 via CSI) e `ximea` (USB 3.0) | `ximea` (USB 3.0), `uvc` (Webcam/Industrial USB) e `mock` |
| **Armazenamento / RAM Drive** | Montagem `tmpfs` obrigatória (`/home/felipe/miniola/capturas`, 1GB max) devido a I/O lento do MicroSD | Armazenamento flexível (`tmpfs` ou direto em SSD NVMe/SATA de alta velocidade) |
| **Motor C++ (`miniola_cv`)** | Compilado via `pybind11` com flags otimizadas para ARM | Compilado via `pybind11` com otimizações nativas x86_64 (SSE/AVX) |
| **Ajuste USB / Display** | Requer `usbfs_memory_mb=1000` em `/boot/firmware/cmdline.txt` e mock de display (`pykms`) | Gerenciamento USB padrão do kernel Linux; sem dependência de `pykms` |

### Diretrizes de Código Multi-Plataforma:
- **Desacoplamento de SO/Hardware**: Nunca insira caminhos ou chamadas exclusivas de Raspberry Pi no fluxo principal sem checar a arquitetura ou usar isolamento de plataforma (ex.: checagem de temperatura em `/sys/class/thermal/thermal_zone0/temp` deve falhar silenciosamente ou retornar `0.0` se não existir em x86_64).
- **Provedores de Câmera (`cameras/`)**: Toda nova interface de captura deve herdar ou respeitar o contrato de `cameras.base.CameraProvider`.

---

## 3. Restrições de Performance e Hardware

1. **Visão Computacional em C++ (`src/miniola_cv.cpp`)**:
   - O processamento de quadros em 120 FPS+ (binarização, busca de contornos de perfurações, cálculo de pitch instantâneo/médio e gatilho) **deve** ser mantido no módulo nativo C++ via `pybind11`.
   - O fallback em Python nativo (`miniola.py`) existe apenas para depuração ou emergência em sistemas onde a compilação C++ falhou.
2. **I/O de Quadros Assíncrono (`multiprocessing.Queue`)**:
   - O loop principal de captura de câmera nunca pode gravar arquivos diretamente em disco ou realizar chamadas bloqueantes de codificação.
   - Todo quadro deve ser enfileirado em `fila_gravacao`, que é consumida por um processo trabalhador isolado em outro núcleo da CPU.
3. **Gerenciamento de Memória**:
   - Evite alocação contínua de novos arrays NumPy ou conversões redundantes de espaço de cor no loop crítico do `miniola.py`.
   - Reutilize buffers sempre que possível e faça o debayering (Bayer -> BGR -> RGB) apenas no processo de gravação ou na exibição do dashboard.

---

## 4. Estrutura e Convenções de Commit

Ao realizar commits no Git, relacione sempre o ID da Especificação principal que motivou a mudança:

- `feat(platform): [SPEC-006] adiciona detecção automática de arquitetura x86_64`
- `fix(vision): [SPEC-001] corrige cálculo de pitch instantâneo quando furos estão na borda`
- `docs(specs): [SPEC-003] documenta filtro de áudio óptico no template`
- `refactor(web): [SPEC-005] otimiza consumo de memória das rotas do painel`

---

## 5. Comandos Rápidos para Verificação local

```bash
# Ativar ambiente virtual
source venv/bin/activate

# Compilar motor C++ via pybind11
python3 setup.py build_ext --inplace

# Validar conformidade e status das especificações (SDD)
python3 scripts/check_specs.py

# Executar testes unitários e de bancada (Mock frames)
python3 -m unittest discover -s tests
```
