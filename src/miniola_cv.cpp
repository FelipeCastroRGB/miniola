#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <opencv2/opencv.hpp>
#include <vector>
#include <algorithm>
#include <cmath>

namespace py = pybind11;

class ScannerVision {
private:
    int contador_perfs_ciclo = 0;
    bool perfuracao_na_linha = false;
    std::vector<double> buffer_pitches;
    double ultimo_pitch_medio = 0.0;
    double encolhimento_atual_pct = 0.0;
    
    // Tracking para o Virtual Rotary Encoder (Audio)
    double last_perf_y = -1.0;

public:
    ScannerVision() {}

    py::dict process_frame(py::array_t<uint8_t> input_array,
                           int roi_x, int roi_y, int roi_w, int roi_h,
                           int thresh_val, int linha_gatilho_y, int margem_gatilho,
                           double pitch_padrao,
                           bool audio_enabled, int audio_x, int audio_w, int audio_slit_y,
                           int audio_thresh_val = 100) {
        
        py::buffer_info buf = input_array.request();
        int rows = buf.shape[0];
        int cols = buf.shape[1];
        
        cv::Mat frame(rows, cols, CV_8UC3, buf.ptr);
        
        cv::Rect roi_rect(std::max(0, roi_x), std::max(0, roi_y),
                          std::min(roi_w, cols - roi_x), std::min(roi_h, rows - roi_y));
                          
        if (roi_rect.width <= 0 || roi_rect.height <= 0) {
            py::dict err; err["capturar"] = false; return err;
        }

        cv::Mat roi_color = frame(roi_rect);
        cv::Mat roi_gray, binary_small;
        cv::cvtColor(roi_color, roi_gray, cv::COLOR_RGB2GRAY);
        
        // Removido o cv::resize(0.5). Analisamos em resolução nativa para fluidez de tracking perfeita e síncrona
        cv::threshold(roi_gray, binary_small, thresh_val, 255, cv::THRESH_BINARY);
        
        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(binary_small, contours, cv::RETR_LIST, cv::CHAIN_APPROX_SIMPLE);
        
        int limite_superior = linha_gatilho_y - margem_gatilho;
        int limite_inferior = linha_gatilho_y + margem_gatilho;
        
        struct Furo {
            int cy_roi;
            int cx_g;
            int cy_g;
            bool acionou;
            cv::Rect rect;
        };
        
        std::vector<Furo> furos_validos;
        py::list debug_visual;
        
        for(size_t i = 0; i < contours.size(); i++) {
            cv::Rect rect = cv::boundingRect(contours[i]);
            double w_s = rect.width;
            double h_s = rect.height;
            double area_aprox = w_s * h_s; // Multiplicador 4.0 varrido
            
            if(area_aprox > 200 && area_aprox < 10000 && (w_s/h_s) > 0.2 && (w_s/h_s) < 2.5) {
                // Cálculo exato no pixel original (resolução 1:1)
                int cy_roi = rect.y + (rect.height / 2);
                int cx_global = rect.x + (rect.width / 2) + roi_rect.x;
                int cy_global = cy_roi + roi_rect.y;
                
                bool acionou = (cy_roi >= limite_superior && cy_roi <= limite_inferior);
                furos_validos.push_back({cy_roi, cx_global, cy_global, acionou, rect});
                
                py::dict debug_item;
                // Coordenadas diretas para o renderizador de tela
                debug_item["rect"] = py::make_tuple(rect.x + roi_rect.x, rect.y + roi_rect.y, rect.width, rect.height);
                debug_item["color"] = acionou ? py::make_tuple(0, 0, 255) : py::make_tuple(0, 255, 0); 
                debug_visual.append(debug_item);
            }
        }
        
        std::sort(furos_validos.begin(), furos_validos.end(), [](const Furo& a, const Furo& b) {
            return a.cy_roi < b.cy_roi;
        });
        
        // --- INÍCIO DO AUDIO LINE-SCANNER ---
        std::vector<float> audio_samples;
        
        double real_pitch = (ultimo_pitch_medio > 0) ? ultimo_pitch_medio : pitch_padrao;
        
        if (audio_enabled && !furos_validos.empty() && real_pitch > 0) {
            double curr_perf_y = furos_validos[0].cy_g;
            
            if (last_perf_y < 0) {
                last_perf_y = curr_perf_y;
            } else {
                double dy = last_perf_y - curr_perf_y; // y diminui quando o filme sobe
                
                // Embrulho de Fase para saltos de furo (Wrapped Phase Tracking)
                while (dy < -(real_pitch * 0.5)) dy += real_pitch;
                while (dy >  (real_pitch * 0.5)) dy -= real_pitch;
                
                int pixels_movidos = std::round(dy);
                
                if (pixels_movidos > 0) {
                    int safe_x = std::max(0, std::min(audio_x, cols - 1));
                    int safe_w = std::max(1, std::min(audio_w, cols - safe_x));
                    
                    // --- JIGSAW POINTER ANCHOR ---
                    // Como o filme trafega para CIMA (Y diminui), os frames novos vêm de baixo.
                    // Para que os recortes de áudio de diferentes tamanhos (devido à variação de velocidade)
                    // se conectem perfeitamente sem deixar vãos cegos "engolindo" a onda, 
                    // temos que manter a BASE fixa e deixar o retângulo crescer pra cima!
                    int base_y = std::min(audio_slit_y + 150, rows - 1); // Ponto de colisão mecânico cravado
                    int read_h = std::min(pixels_movidos, base_y);
                    int safe_y = base_y - read_h;
                    
                    if (read_h > 0) {
                        cv::Rect audio_rect(safe_x, safe_y, safe_w, read_h);
                        cv::Mat audio_slice, audio_slice_gray;
                        cv::cvtColor(frame(audio_rect), audio_slice_gray, cv::COLOR_RGB2GRAY);
                        
                        // Achatar linhas para 1D array com compensação de desfoque óptico
                        for (int r = 0; r < read_h; ++r) {
                            cv::Mat row_mat = audio_slice_gray.row(r);
                            
                            // === LEITURA DE AMPLITUDE POR MODO DE PISTA ===
                            // Área Variável  (audio_thresh_val > 0): binariza a linha antes
                            //   da média. O sinal está na LARGURA da faixa branca — bordas
                            //   desfocadas criariam pixels cinza falsos que distorceriam a
                            //   largura medida. O threshold elimina esse gradiente e entrega
                            //   uma borda digital precisa.
                            //
                            // Densidade Variável (audio_thresh_val == 0): usa a média direta
                            //   dos tons de cinza. O sinal está na LUMINÂNCIA de toda a faixa
                            //   — binarizar destruiria a informação analógica do sinal.
                            double row_mean;
                            if (audio_thresh_val > 0) {
                                cv::Mat row_bin;
                                cv::threshold(row_mat, row_bin, audio_thresh_val, 255, cv::THRESH_BINARY);
                                row_mean = cv::mean(row_bin)[0];
                            } else {
                                row_mean = cv::mean(row_mat)[0];
                            }
                            
                            // Normalização (255 - mean) / 255 convertida de -1.0 a 1.0
                            float val = (float)((255.0 - row_mean) / 255.0);
                            val = (val * 2.0f) - 1.0f;
                            
                            audio_samples.push_back(val);
                        }
                    }
                }
                
                // Acumulador Perfeito de Fase Espacial! 
                // Se movemos 10.4 pixels e limamos para 10, armazenamos o 0.4 para a próxima volta,
                // impedindo a amputação submilimétrica que causa modulação Amplitude a 90Hz (Efeito Vocoder).
                double subpixel_remainder = dy - pixels_movidos;
                last_perf_y = curr_perf_y + subpixel_remainder;
            }
        }
        // --- FIM DO AUDIO LINE-SCANNER ---

        bool furo_detectado_agora = false;
        long cx_a = -1, cy_a = -1;
        bool capturar = false;
        
        if(!furos_validos.empty() && furos_validos[0].acionou) {
            furo_detectado_agora = true;
            if(!perfuracao_na_linha) {
                contador_perfs_ciclo++;
                perfuracao_na_linha = true;
                
                if(contador_perfs_ciclo >= 4) { // Assumindo ciclo de 4 furos para frame cheio no 35mm
                    int qtd = std::min(4, (int)furos_validos.size());
                    
                    long sum_cx = 0;
                    for(int i=0; i<qtd; i++) sum_cx += furos_validos[i].cx_g;
                    cx_a = sum_cx / qtd;
                    
                    if(qtd > 1) {
                        double soma_pitch = 0;
                        for(int i=1; i<qtd; i++) soma_pitch += (furos_validos[i].cy_g - furos_validos[i-1].cy_g);
                        double pitch_instantaneo = soma_pitch / (qtd - 1);
                        
                        if(pitch_instantaneo > 0) {
                            buffer_pitches.push_back(pitch_instantaneo);
                            if(buffer_pitches.size() >= 10) {
                                double p_medio = 0;
                                for(auto p : buffer_pitches) p_medio += p;
                                ultimo_pitch_medio = p_medio / buffer_pitches.size();
                                
                                double calc_pct = (1.0 - (ultimo_pitch_medio / pitch_padrao)) * 100.0;
                                encolhimento_atual_pct = std::max(-5.0, std::min(10.0, calc_pct));
                                buffer_pitches.clear();
                            }
                        }
                        
                        double soma_centros_y = 0;
                        for(int i=0; i<qtd; i++) {
                            double multiplicador = 1.5 - (double)i;
                            soma_centros_y += ((double)furos_validos[i].cy_g + (multiplicador * pitch_instantaneo));
                        }
                        cy_a = std::round(soma_centros_y / qtd);
                    } else {
                        cy_a = furos_validos[0].cy_g + 150;
                    }
                    capturar = true;
                    contador_perfs_ciclo = 0;
                }
            }
        }
        
        if(!furo_detectado_agora) {
            perfuracao_na_linha = false;
        }
        
        py::array_t<uint8_t> result_array({binary_small.rows, binary_small.cols});
        py::buffer_info buf_res = result_array.request();
        std::memcpy(buf_res.ptr, binary_small.data, binary_small.total() * binary_small.elemSize());
        
        // Converte o std::vector de audio para numpy array
        py::array_t<float> audio_numpy(audio_samples.size(), audio_samples.data());
        
        py::dict result;
        result["capturar"] = capturar;
        result["cx_a"] = cx_a;
        result["cy_a"] = cy_a;
        result["debug_visual"] = debug_visual;
        result["binary_small"] = result_array;
        result["perfuracao_na_linha"] = perfuracao_na_linha;
        result["contador_perfs_ciclo"] = contador_perfs_ciclo;
        result["encolhimento_atual_pct"] = encolhimento_atual_pct;
        result["ultimo_pitch_medio"] = ultimo_pitch_medio;
        result["achou_furo"] = furo_detectado_agora;
        result["audio_chunk"] = audio_numpy; 
        
        return result;
    }

    void reset_ciclo() {
        contador_perfs_ciclo = 0;
        last_perf_y = -1.0;
    }
};

PYBIND11_MODULE(miniola_cv, m) {
    m.doc() = "Miniola CV Extension using OpenCV and Pybind11";
    py::class_<ScannerVision>(m, "ScannerVision")
        .def(py::init<>())
        .def("process_frame", &ScannerVision::process_frame,
             py::arg("input_array"),
             py::arg("roi_x"), py::arg("roi_y"), py::arg("roi_w"), py::arg("roi_h"),
             py::arg("thresh_val"), py::arg("linha_gatilho_y"), py::arg("margem_gatilho"),
             py::arg("pitch_padrao"),
             py::arg("audio_enabled") = false, py::arg("audio_x") = 0, py::arg("audio_w") = 0,
             py::arg("audio_slit_y") = 0, py::arg("audio_thresh_val") = 100)
        .def("reset_ciclo", &ScannerVision::reset_ciclo);
}
