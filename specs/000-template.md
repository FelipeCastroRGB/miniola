# SPEC-000: [Título Descritivo da Funcionalidade / Evolução]

| Metadado | Valor |
| :--- | :--- |
| **ID da Especificação** | `SPEC-XXX` |
| **Status** | `Draft` | `Approved` | `In Progress` | `Completed` | `Deprecated` |
| **Autor** | Nome / IA |
| **Data de Criação** | AAAA-MM-DD |
| **Última Atualização** | AAAA-MM-DD |

---

## 1. Contexto e Objetivo
*(Descreva o problema no mundo real da preservação audiovisual ou o motivo técnico que torna esta alteração necessária. Ex.: "Películas 35mm antigas costumam sofrer encolhimento de até 3%, fazendo com que a distância fixa entre perfurações cause desvio na captura...")*

## 2. Requisitos Funcionais (O que o sistema deve fazer)
- `[RF-01]`: O sistema deve...
- `[RF-02]`: O usuário no dashboard web deve ser capaz de...

## 3. Requisitos Não-Funcionais e Performance
- `[RNF-01]`: A taxa de captura do sensor (`fps_cam`) não deve cair abaixo de X FPS.
- `[RNF-02]`: A latência de processamento no motor C++ (`tempo_ms_ciclo`) deve permanecer inferior a X ms por quadro.

---

## 4. Matriz de Impacto Multi-Plataforma

Descreva o comportamento e as restrições em cada perfil de hardware suportado pelo Miniola:

| Plataforma | Comportamento Esperado / Restrições Específicas |
| :--- | :--- |
| **Raspberry Pi 5/4 (`arm64`)** | *(Ex.: Gravar quadros obrigatoriamente no RAM drive `tmpfs` em `/home/felipe/miniola/capturas` com limite de 1GB; compilação C++ com flags ARM).* |
| **Mac Mini / MiniPCs (`x86_64`)** | *(Ex.: Gravar quadros em `tmpfs` ou SSD NVMe local sem limite de 1GB; usar vetorização C++ x86_64; suporte a câmeras UVC/Mock via USB 3.0).* |

---

## 5. Arquitetura e Design Técnico

### 5.1. Componentes e Arquivos Modificados
- `src/miniola_cv.cpp`: *(Descrição das classes, métodos ou algoritmos adicionados/alterados)*
- `miniola.py`: *(Descrição de como o loop principal ou o dashboard interagem)*
- `cameras/`: *(Se houver alterações no contrato ou drivers)*

### 5.2. Contratos e Estruturas de Dados
*(Defina assinaturas de funções C++/Python, formato de pacotes JSON, filas ou sidecars de áudio `.f32`)*

```python
# Exemplo de assinatura ou contrato
def process_frame(frame: np.ndarray, ...) -> dict:
    ...
```

---

## 6. Critérios de Aceitação e Plano de Verificação

### 6.1. Verificação Automatizada / Bancada (`tests/`)
- [ ] O teste unitário `python3 -m unittest discover -s tests` deve passar 100%.
- [ ] A checagem `python3 scripts/check_specs.py` não deve apontar erros nesta spec.

### 6.2. Verificação Manual / Hardware
- [ ] Executar com o provedor de câmera alvo (`--camera ximea` ou `--camera uvc`) por pelo menos 5 minutos sem perda de quadros ou vazamento de memória (`/status`).
- [ ] Verificar integridade da saída gerada em `output/` após rodar `process.py`.
