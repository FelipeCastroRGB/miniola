# Referências Normativas e Arquivísticas (`referencias/`)

Esta pasta armazena o **conhecimento arquivístico, normas técnicas, manuais de hardware e diretrizes de preservação** que servem como fundamentação (*Ground Truth Documental*) para o **Spec-Driven Development (SDD)** do Miniola.

---

## Por que converter PDFs para Markdown (`.md`)?

Para garantir a máxima eficiência na colaboração entre desenvolvedores humanos e **Agentes de IA (Gemini/Antigravity)**, adotamos a prática de **destilar e converter documentos em PDF (como diretrizes da FIAF e manuais) para arquivos Markdown (`.md`) estruturados**:

1. **Leitura Instantânea por Agentes**: Arquivos `.md` são lidos nativamente e com altíssima velocidade por ferramentas de busca semântica (`grep_search`) e IDEs agentizadas, sem gastar tokens excessivos com diagramação de páginas ou ruído de leitura de PDF.
2. **Citação Exata nas Especificações (`specs/`)**: Cada especificação técnica em `specs/` pode citar linhas exatas ou seções de normas em `referencias/` (ex.: `[referencias/fiaf-preservacao-digital.md#L45-L60](file:///home/felipe/Projetos/miniola/referencias/fiaf-preservacao-digital.md#L45-L60)`).
3. **Padrão e Limpeza**: Tabelas de tolerância geométrica, fórmulas matemáticas de pitch e parâmetros de digitalização são mantidos limpos e prontos para verificação em código.

> **Dica**: Se desejar manter o PDF original baixado da FIAF ou do fabricante, guarde-o na subpasta `referencias/pdfs_originais/`, e crie o arquivo `.md` correspondente na raiz de `referencias/` usando o nosso template `000-template-referencia.md`.

---

## Índice de Referências Ativas

| ID | Documento / Norma | Tema Principal | Arquivo MD |
| :--- | :--- | :--- | :--- |
| **REF-000** | Template de Referência Normativa | Estrutura padrão para converter normas/manuais em MD | `000-template-referencia.md` |
| **REF-001** | Padrões e Dimensões da Bitola 35mm (SMPTE/ANSI) | Geometria de perfurações BH, KS e CS, pitch nominal e tolerâncias | `smpte-35mm-dimensoes.md` |
| **REF-002** | FIAF Digital Statement I - Práticas de Digitalização | Sensores de área, linha e Bayer/mosaico; supressão de riscos por Wet Gate/difusão | `fiaf-digital-statement-1.md` |
| **REF-003** | FIAF Digital Statement II - Digitalização e Preservação | Overscan, resolução (2K/4K/6K+), densidade Cineon 10-bit log e aliasing | `fiaf-digital-statement-2.md` |
| **REF-004** | FIAF Film Digitisation - Diretrizes e Práticas (2021) | Transporte mecânico suave sem pinos, iluminação fria e inspeção pré-scan | `fiaf-film-digitisation-2021.md` |
| **REF-005** | FADGI DPX Embedded Metadata (`SMPTE ST 268`) | Metadados obrigatórios no cabeçalho binário/ASCII de arquivos DPX | `fadgi-dpx-embedded-metadata.md` |
| **REF-006** | FADGI Digitizing Motion Picture Film (SOW) | Fluxogramas de captação para som óptico, magnético e filmes silenciosos | `fadgi-digitizing-motion-picture-film.md` |
| **REF-007** | FADGI Audio ADC Performance Specifications | Métricas de ensaio e tolerâncias (High Level & Low Cost) para conversores de áudio | `fadgi-audio-adc-performance.md` |
| **REF-008** | Kodak Essential Reference Guide for Filmmakers | Bases de acetato/ESTAR, cristais T-Grain, curvas sensitométricas H&D e códigos | `kodak-filmmakers-reference-guide.md` |

---

## Como Adicionar uma Nova Referência

1. Copie o arquivo `000-template-referencia.md` para o nome do documento (ex.: `fiaf-audio-preservation.md`).
2. Extraia os pontos essenciais, tabelas e parâmetros de preservação do PDF ou documento original.
3. Registre o novo documento no índice acima e vincule-o às especificações de `specs/` relevantes.
