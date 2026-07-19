#!/usr/bin/env python3
"""
Anotador Visual de Amostras e Geração de Ground Truth (SPEC-007) - Miniola

Servidor Flask independente rodando na porta 5001 (ou especificada) que fornece
uma interface HTML5 Canvas para anotação visual interativa de fotogramas 35mm,
gerando automaticamente arquivos Sidecar JSON na pasta amostras/.

Uso:
    python3 tools/anotador_amostras.py [--port 5001]
"""

import os
import json
import argparse
from pathlib import Path
from flask import Flask, render_template, request, jsonify, send_from_directory

PROJECT_ROOT = Path(__file__).resolve().parent.parent
AMOSTRAS_DIR = PROJECT_ROOT / "amostras"
TEMPLATES_DIR = Path(__file__).resolve().parent / "templates"

app = Flask(__name__, template_folder=str(TEMPLATES_DIR))


@app.route("/")
def index():
    """Renderiza a interface web interativa (HTML5 Canvas)."""
    return render_template("anotador.html")


@app.route("/api/images", methods=["GET"])
def get_images():
    """Vasculha amostras/fotogramas e amostras/audio_optico retornando lista de imagens."""
    if not AMOSTRAS_DIR.exists():
        return jsonify({"images": [], "error": "Pasta amostras/ não encontrada"}), 404

    image_exts = {".png", ".jpg", ".jpeg"}
    images_list = []

    # Procura recursivamente em subpastas ou na raiz de amostras/
    for root, _, files in os.walk(AMOSTRAS_DIR):
        for file in sorted(files):
            ext = Path(file).suffix.lower()
            if ext in image_exts:
                abs_path = Path(root) / file
                rel_path = abs_path.relative_to(AMOSTRAS_DIR).as_posix()
                
                # Checa se o sidecar .json existe
                sidecar_path = abs_path.with_suffix(".json")
                has_json = sidecar_path.exists()

                images_list.append({
                    "filepath": rel_path,
                    "filename": file,
                    "has_json": has_json
                })

    return jsonify({"images": images_list, "count": len(images_list)})


@app.route("/api/image_data/<path:filepath>", methods=["GET"])
def get_image_data(filepath):
    """Entrega o arquivo binário da imagem selecionada para o canvas."""
    return send_from_directory(str(AMOSTRAS_DIR), filepath)


@app.route("/api/sidecar/<path:filepath>", methods=["GET"])
def get_sidecar(filepath):
    """Retorna o JSON sidecar existente se houver."""
    sidecar_path = (AMOSTRAS_DIR / filepath).with_suffix(".json")
    if sidecar_path.exists():
        try:
            with open(sidecar_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return jsonify(data), 200
        except Exception as e:
            return jsonify({"error": f"Falha ao ler JSON: {e}"}), 500
    return jsonify({"error": "Sidecar não encontrado"}), 404


@app.route("/api/save_json", methods=["POST"])
def save_json():
    """Recebe o payload da interface e grava o sidecar .json na pasta amostras/."""
    try:
        payload = request.get_json()
        if not payload or "filepath" not in payload or "data" not in payload:
            return jsonify({"status": "error", "message": "Payload mal formatado ou incompleto"}), 400

        filepath = payload["filepath"]
        json_data = payload["data"]

        target_path = (AMOSTRAS_DIR / filepath).with_suffix(".json")
        target_path.parent.mkdir(parents=True, exist_ok=True)

        with open(target_path, "w", encoding="utf-8") as f:
            json.dump(json_data, f, indent=2, ensure_ascii=False)

        rel_saved = target_path.relative_to(PROJECT_ROOT).as_posix()
        print(f"[ANOTADOR] Sidecar salvo com sucesso em: {rel_saved}")
        return jsonify({"status": "success", "saved_path": rel_saved}), 200

    except Exception as e:
        print(f"[ANOTADOR] Erro ao salvar sidecar: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500


def main():
    parser = argparse.ArgumentParser(description="Servidor do Anotador Visual de Amostras do Miniola")
    parser.add_argument("--host", default="0.0.0.0", help="Endereço de escuta (padrão 0.0.0.0)")
    parser.add_argument("--port", type=int, default=5001, help="Porta HTTP (padrão 5001)")
    parser.add_argument("--debug", action="store_true", help="Ativar modo de depuração do Flask")
    args = parser.parse_args()

    print("═" * 70)
    print("   MINIOLA - ANOTADOR VISUAL INTERATIVO DE GROUND TRUTH (SPEC-007)")
    print("═" * 70)
    print(f"Diretório de amostras: {AMOSTRAS_DIR}")
    print(f"Acesse a interface em: http://localhost:{args.port}")
    print("═" * 70)

    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()
