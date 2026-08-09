# SPEC-002: Camada de Abstração de Câmeras (CameraProvider)

| Metadado | Valor |
| :--- | :--- |
| **ID da Especificação** | `SPEC-002` |
| **Status** | `Completed` |
| **Autor** | Equipe Miniola |
| **Data de Criação** | 2026-07-19 |
| **Última Atualização** | 2026-07-19 |

---

## 1. Contexto e Objetivo
Para democratizar a preservação audiovisual, o Miniola não pode ficar preso a um único modelo de sensor. Projetos de baixo custo podem utilizar a câmera padrão do ecossistema Raspberry Pi (`Camera Module 3` via `picamera2`), enquanto digitalizações profissionais/industriais utilizam a câmera de alta velocidade Ximea (`MQ042MG-CM` com `ximea_api`). Além disso, para rodar em MiniPCs ou no Mac Mini x86_64, o sistema precisa suportar câmeras industriais genéricas USB/UVC e provedores virtuais de simulação (Mock/Playback).

Esta especificação formaliza a interface abstrata `CameraProvider` (`cameras/base.py`) e o contrato modular dos drivers implementados.

## 2. Requisitos Funcionais
- `[RF-01]`: O sistema deve prover uma função fábrica `get_camera_provider(provider_name: str) -> CameraProvider` em `cameras/__init__.py`.
- `[RF-02]`: Todos os drivers de câmera devem herdar de `CameraProvider` e implementar obrigatoriamente os métodos: `start(...)`, `get_frame() -> np.ndarray`, `set_exposure(...)`, `set_gain(...)`, `set_fps(...)` e `stop()`.
- `[RF-03]`: O método `get_frame()` deve retornar rapidamente (`non-blocking` ou com timeout curto) um quadro RAW ou BGR do tamanho configurado no `start()`.
- `[RF-04]`: O driver `ximea` (`cameras/ximea.py`) deve aplicar crop por hardware (`CAM_OFFSET_X`, `CAM_OFFSET_Y`) diretamente nos registradores do sensor via `xiAPI` para maximizar o frame rate no barramento USB 3.0, ou utilizar negociação automática (`auto_bandwidth_calculation = 1`) com fallback `FREE_RUN` e `timeout=2000` em barramentos xHCI.
- `[RF-05]`: O driver `pi` (`cameras/pi.py`) deve instanciar `Picamera2`, configurar controles manuais (exposição, ganho) e capturar frames contínuos em memória.
- `[RF-06]`: A arquitetura deve permitir registrar novos provedores (como `uvc` e `mock`) dinamicamente para rodar em hardware x86_64 que não possui CSI de Raspberry Pi.

## 3. Requisitos Não-Funcionais e Performance
- `[RNF-01]`: A obtenção do quadro em `get_frame()` deve retornar sem alocar cópias redundantes de memória, EXCETO quando o provedor retornar um ponteiro sobre um buffer C mutável externo (como em `XimeaAdapter`, onde `get_image_data_numpy().copy()` é obrigatório para evitar colisão de concorrência com threads consumidoras do OpenCV/Flask).
- `[RNF-02]`: O tempo entre chamadas sucessivas de `get_frame()` deve ser estável para suportar 120 FPS (`< 8.33 ms` por frame).

---

## 4. Matriz de Impacto Multi-Plataforma

| Plataforma | Provedores Suportados | Restrições / Notas de Operação |
| :--- | :--- | :--- |
| **Raspberry Pi 5/4 (`arm64`)** | `pi`, `ximea`, `mock` | O driver `pi` exige `picamera2` instalado via apt/pip do SO Bookworm. O driver `ximea` exige o aumento permanente do buffer USB (`usbfs_memory_mb=1000`) em `/boot/firmware/cmdline.txt`. |
| **Mac Mini / MiniPCs (`x86_64`)** | `ximea`, `uvc`, `mock` | O driver `pi` não está disponível em sistemas x86_64 sem hardware CSI (o import deve ser graciosamente evitado em `__init__.py` ou retornar erro explicativo se solicitado). A câmera Ximea conecta via USB 3.0 no Linux x86_64 com o pacote `ximea_linux_arm_sp_beta.tgz` (ou versão x86). |

---

## 5. Arquitetura e Design Técnico

### 5.1. Componentes e Arquivos Modificados
- `cameras/base.py`: Define a classe abstrata `CameraProvider`.
- `cameras/ximea.py`: Implementação para câmeras industriais Ximea usando `from ximea import xiapi`.
- `cameras/pi.py`: Implementação para módulos Raspberry Pi usando `from picamera2 import Picamera2`.
- `cameras/__init__.py`: Seletor de provedores `get_camera_provider(provider_name)`.

### 5.2. Contrato da Classe Abstrata (`cameras/base.py`)
```python
class CameraProvider:
    def start(self, width: int, height: int, fps: int, shutter_us: int, gain: float, focus: float, offset_x: int, offset_y: int) -> None:
        raise NotImplementedError
        
    def get_frame(self) -> np.ndarray:
        raise NotImplementedError
        
    def set_exposure(self, shutter_us: int) -> None:
        raise NotImplementedError
        
    def set_gain(self, gain: float) -> None:
        raise NotImplementedError
        
    def set_fps(self, fps: int) -> None:
        raise NotImplementedError
        
    def set_focus(self, focus: float) -> None:
        pass
        
    def set_white_balance(self, kr: float, kg: float, kb: float) -> None:
        pass
        
    def set_gamma(self, gamma_y: float, gamma_c: float) -> None:
        pass
        
    def set_contrast(self, value: float) -> None:
        pass
        
    def set_sharpness(self, value: float) -> None:
        pass
        
    def stop(self) -> None:
        raise NotImplementedError
```

---

## 6. Critérios de Aceitação e Plano de Verificação

### 6.1. Verificação Automatizada (`tests/`)
- [x] O teste de verificação de especificações (`check_specs.py`) confirma a existência e validade do contrato `CameraProvider`.
- [x] A instanciação de provedores mock/sintéticos em `tests/` verifica se `get_frame()` retorna matrizes com as dimensões especificadas (`RES_W, RES_H`).

### 6.2. Verificação Manual / Hardware
- [x] Ao iniciar com `python3 miniola.py --camera ximea` no Raspberry Pi ou MiniPC, a inicialização imprime o modelo da câmera e começa a transmitir quadros no dashboard sem engasgos de buffer USB.
