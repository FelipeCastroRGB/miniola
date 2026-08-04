#!/usr/bin/env python3
"""
Validador de Especificações (SDD - Spec-Driven Development) para o Miniola.
Verifica se todas as especificações em specs/ possuem os cabeçalhos, status válidos,
matriz de impacto Multi-Plataforma e seções obrigatórias.

Uso:
    python3 scripts/check_specs.py
"""

import os
import re
import sys
from pathlib import Path

# Diretório base do projeto
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SPECS_DIR = PROJECT_ROOT / "specs"

STATUS_VALIDOS = {"Draft", "Approved", "In Progress", "Completed", "Deprecated"}

SECOES_OBRIGATORIAS = [
    "Contexto e Objetivo",
    "Requisitos Funcionais",
    "Matriz de Impacto Multi-Plataforma",
    "Arquitetura e Design Técnico",
    "Critérios de Aceitação e Plano de Verificação",
]


def check_spec_file(filepath: Path) -> dict:
    content = filepath.read_text(encoding="utf-8")
    lines = content.splitlines()

    resultado = {
        "filename": filepath.name,
        "title": "Desconhecido",
        "spec_id": "N/A",
        "status": "N/A",
        "missing_sections": [],
        "errors": [],
        "valid": True,
    }

    # Extrair título principal H1
    for line in lines:
        if line.startswith("# "):
            resultado["title"] = line[2:].strip()
            break

    # Extrair ID da Spec do cabeçalho da tabela ou do título
    id_match = re.search(r"(SPEC-\d{3})", content)
    if id_match:
        resultado["spec_id"] = id_match.group(1)
    else:
        resultado["errors"].append("ID SPEC-XXX não encontrado")
        resultado["valid"] = False

    # Extrair Status da tabela de metadados
    status_match = re.search(r"\|\s*\*\*Status\*\*\s*\|\s*`?([A-Za-z\s]+)`?\s*\|", content)
    if status_match:
        status_raw = status_match.group(1).strip()
        # Se contiver múltiplos status (ex.: no template), checar se há um status único válido ou se é o template
        if filepath.name == "000-template.md":
            resultado["status"] = "Template"
        elif status_raw in STATUS_VALIDOS:
            resultado["status"] = status_raw
        else:
            # Tentar pegar palavra única
            for s in STATUS_VALIDOS:
                if s.lower() == status_raw.lower():
                    resultado["status"] = s
                    break
            else:
                resultado["errors"].append(f"Status inválido: '{status_raw}'")
                resultado["valid"] = False
    elif filepath.name == "000-template.md":
        resultado["status"] = "Template"
    else:
        resultado["errors"].append("Metadado '**Status**' não encontrado na tabela de metadados")
        resultado["valid"] = False

    # Verificar Seções Obrigatórias
    for secao in SECOES_OBRIGATORIAS:
        found = False
        for line in lines:
            if line.strip().startswith("## ") and secao.lower() in line.lower():
                found = True
                break
        if not found:
            resultado["missing_sections"].append(secao)
            resultado["valid"] = False

    return resultado


def main():
    if not SPECS_DIR.exists():
        print(f"[ERRO] Diretório de especificações não encontrado: {SPECS_DIR}")
        sys.exit(1)

    spec_files = sorted([f for f in SPECS_DIR.glob("*.md") if f.is_file()])
    if not spec_files:
        print(f"[AVISO] Nenhuma especificação encontrada em {SPECS_DIR}")
        sys.exit(0)

    print("═" * 90)
    print("   MINIOLA - VALIDADOR DE SPEC-DRIVEN DEVELOPMENT (SDD)")
    print("═" * 90)
    print(f"{'ID':<10} {'Arquivo':<32} {'Status':<14} {'Seções / Erros'}")
    print("─" * 90)

    total = len(spec_files)
    valid_count = 0
    all_errors = []

    for f in spec_files:
        res = check_spec_file(f)
        if res["valid"]:
            valid_count += 1
            info = "✓ Ok"
        else:
            err_msgs = res["errors"] + [f"Falta: {s}" for s in res["missing_sections"]]
            info = "✗ " + "; ".join(err_msgs)
            all_errors.append((res["filename"], err_msgs))

        print(f"{res['spec_id']:<10} {res['filename']:<32} {res['status']:<14} {info}")

    print("═" * 90)
    print(f"Resumo: {valid_count}/{total} especificações válidas.")

    if all_errors:
        print("\n[ERRO] Detalhamento das falhas encontradas:")
        for fname, errs in all_errors:
            print(f"  -> {fname}:")
            for e in errs:
                print(f"       * {e}")
        sys.exit(1)
    else:
        print("\n[SUCESSO] Todas as especificações estão em conformidade com o SDD e Multi-Plataforma!")
        sys.exit(0)


if __name__ == "__main__":
    main()
