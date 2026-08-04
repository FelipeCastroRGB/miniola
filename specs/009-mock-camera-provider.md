# SPEC-009: Provedor Mock de Câmera (`mock`)

| Metadado | Valor |
| :--- | :--- |
| **ID da Especificação** | `SPEC-009` |
| **Status** | `Completed` |
| **Autor** | Equipe Miniola |
| **Data de Criação** | 2026-07-23 |
| **Última Atualização** | 2026-07-23 |

---

## 1. Contexto e Objetivo
Para possibilitar o desenvolvimento, teste e validação de rotas, gravação assíncrona e gatilhos de visão em computadores de desenvolvimento (x86_64) sem a necessidade de hardware conectado, o sistema necessita de um provedor de câmera simulado (`mock`). 
Este provedor atende ao princípio de desenvolvimento multi-plataforma e reduz o atrito para testar a interface e a lógica central do motor.

## 2. Requisitos Funcionais
- `[RF-01]`: O provedor deve herdar de `CameraProvider`.
- `[RF-02]`: O modo sintético deve gerar frames contínuos a 120 FPS que simulem uma película em movimento, contendo perfurações animadas (retângulos brancos se deslocando na vertical) e uma fenda de áudio, permitindo o engajamento do algoritmo C++ (`miniola_cv`).
- `[RF-03]`: Se fornecido um caminho de vídeo (playback) na inicialização ou via variável de ambiente, o provedor deve extrair os frames desse vídeo iterativamente, fazendo loop ao final.
- `[RF-04]`: O provedor **não deve operar em sistemas ARM64** (Raspberry Pi), para evitar consumo desnecessário de CPU com geração de frames falsos em um ambiente de produção/captura. O instanciamento deve lançar erro ou ser prevenido.

## 3. Requisitos Não-Funcionais e Performance
- `[RNF-01]`: A geração de quadros sintéticos deve ser otimizada (usando arrays numpy pré-alocados ou operações rápidas) para manter os 120 FPS na máquina de desenvolvimento.
- `[RNF-02]`: O mock deve implementar todos os métodos da interface (ex: `set_exposure`, `set_gain`, etc) de forma "no-op" (sem fazer nada, ou apenas registrando o valor).

---

## 4. Matriz de Impacto Multi-Plataforma

| Plataforma | Comportamento Esperado / Restrições Específicas |
| :--- | :--- |
| **Raspberry Pi 5/4 (`arm64`)** | Não suportado. O instanciamento falhará intencionalmente com `RuntimeError` ou similar. |
| **Mac Mini / MiniPCs (`x86_64`)** | Suportado nativamente. Funciona como principal ferramenta de simulação de hardware (SDD). |

---

## 5. Arquitetura e Design Técnico

### 5.1. Componentes e Arquivos Modificados
- `cameras/mock.py`: Implementa `MockCameraProvider`. Gera quadros pretos com elementos brancos dinâmicos usando `numpy` e `cv2`. Mantém o tempo com `time.sleep` ou controle de loop para não ultrapassar 120 FPS.
- `cameras/__init__.py`: Importa e retorna a instância de `MockCameraProvider` se a plataforma não for `arm64` / `aarch64`.

### 5.2. Contratos e Estruturas de Dados
```python
class MockCameraProvider(CameraProvider):
    def __init__(self, video_path: str = None):
        ...
```

---

## 6. Critérios de Aceitação e Plano de Verificação

### 6.1. Verificação Automatizada / Bancada (`tests/`)
- [ ] O script `check_specs.py` não aponta erros.

### 6.2. Verificação Manual / Hardware
- [ ] Ao iniciar com `--camera mock` num x86_64, a UI mostra quadros em movimento e mantém ~120 FPS.
- [ ] Se tentar iniciar com `--camera mock` num RPi, ocorre um log/falha clara sobre arquitetura não suportada.
