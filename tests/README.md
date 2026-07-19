# Framework de Testes de Bancada e Validação SDD (`tests/`)

Este diretório contém a suíte de testes automatizados e sintéticos do **Miniola**, desenvolvida para permitir a validação do fluxo de **Spec-Driven Development (SDD)** no seu PC Linux local (como Mac Mini Late 2012 x86_64 ou MiniPCs) sem a necessidade de hardware físico (câmera ou Raspberry Pi acoplados).

## Objetivos
- Verificação rápida do motor C++ (`miniola_cv`) usando quadros sintéticos (`mock`).
- Garantir conformidade com as especificações [SPEC-001](../specs/001-vision-engine-cpp.md) e [SPEC-006](../specs/006-multiplatform-minipc-support.md).
- Execução em Integração Contínua (CI) no GitHub Actions (`.github/workflows/sdd-validate.yml`).

## Como Executar

```bash
# A partir da raiz do projeto, com o ambiente virtual ativo:
python3 -m unittest discover -s tests -v
```
