import unittest
import numpy as np
import cv2
import sys
import os

# Adiciona o diretório raiz ao path para importar módulos do projeto e o binário compiled C++
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


class TestVisionEngine(unittest.TestCase):
    """
    Testes de bancada para o motor de visão (SPEC-001) e suporte multi-plataforma (SPEC-006).
    Utiliza quadros sintéticos com perfurações em padrão 35mm simuladas.
    """

    @classmethod
    def setUpClass(cls):
        try:
            import miniola_cv
            cls.has_cpp_module = True
            cls.scanner_cv = miniola_cv.ScannerVision()
        except ImportError:
            cls.has_cpp_module = False
            cls.scanner_cv = None

    def test_01_cpp_module_loaded(self):
        """Verifica se a extensão C++ foi importada com sucesso via pybind11."""
        if not self.has_cpp_module:
            self.skipTest("Extensão C++ miniola_cv não está compilada no ambiente atual. Execute `python3 setup.py build_ext --inplace`.")
        self.assertIsNotNone(self.scanner_cv, "Instância de ScannerVision falhou")

    def create_synthetic_frame(self, w=1420, h=880, perf_y=110):
        """
        Gera um quadro BGR escuro com um retângulo branco simulando o furo da perfuração
        na posição perf_y dentro da ROI de visão.
        ROI padrão: ROI_X=200, ROI_Y=10, ROI_W=80, ROI_H=840.
        """
        frame = np.zeros((h, w, 3), dtype=np.uint8)
        # Desenha retângulo de perfuração (branco brilhante > threshold 239)
        roi_x, roi_y = 200, 10
        # Coordenadas globais da perfuração
        px1 = roi_x + 10
        px2 = roi_x + 70
        py1 = roi_y + perf_y - 15
        py2 = roi_y + perf_y + 15
        cv2.rectangle(frame, (px1, py1), (px2, py2), (255, 255, 255), -1)
        return frame

    def test_02_perforation_trigger_cycle(self):
        """
        Simula a passagem de 4 perfurações consecutivas sobre a linha de gatilho
        e verifica se o ciclo dispara gravação (capturar=True) exatamente no 4º furo.
        """
        if not self.has_cpp_module:
            self.skipTest("Módulo C++ ausente")

        self.scanner_cv.reset_ciclo()

        roi_x, roi_y, roi_w, roi_h = 200, 10, 80, 840
        thresh_val = 239
        linha_gatilho_y = 110  # Perfuração simulada na linha de gatilho
        margem_gatilho = 23
        pitch_padrao_px = 195.0
        capturar_audio = False
        audio_x, audio_w, slit_y = 280, 96, 430

        # Envia quadro sem perfuração na linha primeiro para estabilizar
        frame_vazio = np.zeros((880, 1420, 3), dtype=np.uint8)
        res_vazio = self.scanner_cv.process_frame(
            frame_vazio, roi_x, roi_y, roi_w, roi_h,
            thresh_val, linha_gatilho_y, margem_gatilho,
            pitch_padrao_px, capturar_audio, audio_x, audio_w, slit_y
        )
        self.assertFalse(res_vazio["capturar"], "Não deve capturar em quadro vazio")

        # Simula avanço de 4 perfurações que se aproximam e cruzam a linha de gatilho
        capturas_disparadas = 0
        
        for i in range(1, 5):
            # 1. Quadro com furo ACIMA da linha (arma o gatilho de fase PLL via histerese > 15%)
            frame_aproxima = self.create_synthetic_frame(perf_y=linha_gatilho_y - 35)
            self.scanner_cv.process_frame(
                frame_aproxima, roi_x, roi_y, roi_w, roi_h,
                thresh_val, linha_gatilho_y, margem_gatilho,
                pitch_padrao_px, capturar_audio, audio_x, audio_w, slit_y
            )

            # 2. Quadro com furo CRUZANDO a linha (dispara o contador de ciclo PLL)
            frame_cruza = self.create_synthetic_frame(perf_y=linha_gatilho_y + 2)
            res = self.scanner_cv.process_frame(
                frame_cruza, roi_x, roi_y, roi_w, roi_h,
                thresh_val, linha_gatilho_y, margem_gatilho,
                pitch_padrao_px, capturar_audio, audio_x, audio_w, slit_y
            )
            if res["capturar"]:
                capturas_disparadas += 1

            # 3. Entre cada furo, passamos um quadro vazio simulando avanço entre perfurações
            self.scanner_cv.process_frame(
                frame_vazio, roi_x, roi_y, roi_w, roi_h,
                thresh_val, linha_gatilho_y, margem_gatilho,
                pitch_padrao_px, capturar_audio, audio_x, audio_w, slit_y
            )

        self.assertEqual(capturas_disparadas, 1, "O motor de visão deve disparar exatamente 1 captura ao completar o ciclo de 4 perfurações")


if __name__ == "__main__":
    unittest.main()
