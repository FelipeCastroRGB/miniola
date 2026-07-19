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
    double ultimo_pitch_medio = -1.0;
    double ultimo_pitch_instantaneo = -1.0;
    double encolhimento_atual_pct = 0.0;
    
    // Tracking para o Virtual Rotary Encoder (Audio)
    double last_perf_y = -1.0;
    
    // Gatilho de Fase Circular (PLL - Phase-Locked Loop)
    bool gatilho_fase_armado = false;
    double last_erro_fase = 0.0;
    
    // Tracking de Autocorrelação (Auto-Stitching)
    std::vector<float> audio_tail;

public:
    ScannerVision() {}

    py::dict process_frame(py::array_t<uint8_t> input_array,
                           int roi_x, int roi_y, int roi_w, int roi_h,
                           int thresh_val, int linha_gatilho_y, int margem_gatilho,
                           double pitch_padrao,
                           bool audio_enabled, int audio_x, int audio_w, int audio_slit_y) {
        
        py::buffer_info buf = input_array.request();
        int rows = buf.shape[0];
        int cols = buf.shape[1];
        
        cv::Mat frame;
        if (buf.ndim == 2) {
            frame = cv::Mat(rows, cols, CV_8UC1, buf.ptr);
        } else if (buf.ndim == 3) {
            frame = cv::Mat(rows, cols, CV_8UC3, buf.ptr);
        } else {
            py::dict err; err["capturar"] = false; return err;
        }
        
        cv::Rect roi_rect(
            std::max(0, roi_x),
            std::max(0, roi_y),
            std::max(0, std::min(roi_w, cols - std::max(0, roi_x))),
            std::max(0, std::min(roi_h, rows - std::max(0, roi_y)))
        );
                          
        if (roi_rect.width <= 0 || roi_rect.height <= 0) {
            py::dict err; err["capturar"] = false; return err;
        }

        cv::Mat roi_color = frame(roi_rect);
        cv::Mat roi_gray, binary_small;
        
        // Se a imagem já for monocromática (RAW8), pulamos a conversão para cinza (economia gigante de CPU!)
        if (frame.channels() == 3) {
            cv::cvtColor(roi_color, roi_gray, cv::COLOR_RGB2GRAY);
        } else {
            roi_gray = roi_color;
        }
        
        // --- INÍCIO DA PROJEÇÃO 1D HÍBRIDA ---
        
        // Mantemos a imagem pequena apenas para preview no painel UI (reduz uso de banda de vídeo)
        cv::Mat roi_small;
        cv::resize(roi_gray, roi_small, cv::Size(), 0.5, 0.5, cv::INTER_NEAREST);
        cv::threshold(roi_small, binary_small, thresh_val, 255, cv::THRESH_BINARY);
        
        int limite_superior = linha_gatilho_y - margem_gatilho;
        int limite_inferior = linha_gatilho_y + margem_gatilho;
        
        struct Furo {
            double cy_roi;
            double cx_g;
            double cy_g;
            bool acionou;
            cv::Rect rect;
        };
        
        std::vector<Furo> furos_validos;
        py::list debug_visual;
        
        // 1. Binarização na resolução total para máxima precisão na projeção
        cv::Mat binary_full;
        cv::threshold(roi_gray, binary_full, thresh_val, 255, cv::THRESH_BINARY);
        
        // 2. Scanline Profiling: Conta pixels brancos por linha
        int min_width_px = std::max(2, binary_full.cols / 10); // Tolerância a sujeira (ex: mín 8px para ROI de 80px)
        bool in_hole = false;
        int start_y = 0;
        
        for (int y = 0; y < binary_full.rows; ++y) {
            int count_white = 0;
            const uint8_t* ptr = binary_full.ptr<uint8_t>(y);
            for (int x = 0; x < binary_full.cols; ++x) {
                if (ptr[x] > 0) count_white++;
            }
            
            bool is_bright = count_white >= min_width_px;
            
            // Força fechamento se o furo encostar no rodapé do ROI
            if (y == binary_full.rows - 1 && in_hole && is_bright) {
                is_bright = false;
            }

            if (is_bright && !in_hole) {
                in_hole = true;
                start_y = y;
            } else if (!is_bright && in_hole) {
                in_hole = false;
                int end_y = y - 1;
                int h = end_y - start_y + 1;
                
                // Filtro 1D super robusto: basta ter entre 10 e 150 pixels de altura
                if (h >= 10 && h <= std::max(150, binary_full.rows / 2)) {
                    
                    cv::Rect rect(0, start_y, binary_full.cols, h);
                    
                    // 3. O TIRO DE PRECISÃO: Momentos 2D sub-pixel apenas dentro da "gaiola" 1D
                    cv::Mat perf_crop = binary_full(rect); // Já temos a binária!
                    cv::Moments M = cv::moments(perf_crop, true);
                    
                    double cx_roi = (M.m00 != 0) ? (M.m10 / M.m00) + rect.x : (rect.x + rect.width / 2.0);
                    double cy_roi = (M.m00 != 0) ? (M.m01 / M.m00) + rect.y : (rect.y + rect.height / 2.0);
                    
                    double cx_global = cx_roi + roi_rect.x;
                    double cy_global = cy_roi + roi_rect.y;
                    
                    bool acionou = (cy_roi >= limite_superior && cy_roi <= limite_inferior);
                    furos_validos.push_back({cy_roi, cx_global, cy_global, acionou, rect});
                    
                    py::dict debug_item;
                    debug_item["rect"] = py::make_tuple(rect.x + roi_rect.x, rect.y + roi_rect.y, rect.width, rect.height);
                    debug_item["color"] = acionou ? py::make_tuple(0, 0, 255) : py::make_tuple(0, 255, 0); 
                    debug_visual.append(debug_item);
                }
            }
        }
        // --- FIM DA PROJEÇÃO 1D HÍBRIDA ---
        
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
                double raw_dy = last_perf_y - curr_perf_y; 
                while (raw_dy < -(real_pitch * 0.5)) { raw_dy += real_pitch; }
                while (raw_dy >  (real_pitch * 0.5)) { raw_dy -= real_pitch; }
                
                // Estimativa grosseira do avanço guiada pelo furo
                int estimated_dy = std::max(0, (int)std::round(std::abs(raw_dy)));
                
                if (estimated_dy > 0) {
                    int safe_x = std::max(0, std::min(audio_x, cols - 1));
                    int safe_w = std::max(1, std::min(audio_w, cols - safe_x));
                    int base_y = std::min(audio_slit_y + 150, rows - 1); 
                    
                    // Parâmetros de Autocorrelação
                    int tail_size = 20;     // O tamanho da impressão digital
                    int search_margin = 15; // Margem para compensar a distorção da lente
                    
                    // Extraímos mais áudio do que o necessário para podermos deslizar o molde
                    int read_h = audio_tail.empty() ? (estimated_dy + tail_size) : (estimated_dy + tail_size + search_margin);
                    int safe_y = std::max(0, base_y - read_h);
                    read_h = base_y - safe_y; 
                    
                    if (read_h >= tail_size && safe_w > 0) {
                        cv::Rect process_rect(safe_x, safe_y, safe_w, read_h);
                        cv::Mat slice_color = frame(process_rect);
                        cv::Mat slice_gray;
                        
                        if (frame.channels() == 3) {
                            cv::cvtColor(slice_color, slice_gray, cv::COLOR_RGB2GRAY);
                        } else {
                            slice_gray = slice_color;
                        }
                        
                        std::vector<float> current_chunk;
                        current_chunk.reserve(read_h);
                        
                        // Varredura da área alargada
                        for (int r = 0; r < read_h; ++r) {
                            double media_luma_linha = 0.0;
                            for (int x = 0; x < safe_w; ++x) {
                                media_luma_linha += slice_gray.at<uint8_t>(r, x);
                            }
                            media_luma_linha /= (double)safe_w;
                            float val = (float)((255.0 - media_luma_linha) / 255.0);
                            current_chunk.push_back((val * 2.0f) - 1.0f);
                        }
                        
                        int start_copy_idx = 0;
                        
                        // O MILAGRE DA AUTOCORRELAÇÃO: SAD (Sum of Absolute Differences)
                        if (!audio_tail.empty() && (int)current_chunk.size() >= (tail_size + search_margin)) {
                            double min_sad = 1e9;
                            int best_offset = 0;
                            
                            // Desliza a cauda velha sobre o topo da onda nova
                            int max_search = std::min(search_margin * 2, (int)current_chunk.size() - tail_size);
                            for (int offset = 0; offset <= max_search; ++offset) {
                                double current_sad = 0.0;
                                for (int i = 0; i < tail_size; ++i) {
                                    current_sad += std::abs(audio_tail[i] - current_chunk[offset + i]);
                                }
                                if (current_sad < min_sad) {
                                    min_sad = current_sad;
                                    best_offset = offset;
                                }
                            }
                            // O áudio novo, que ainda não foi exportado, começa logo após o molde perfeito!
                            start_copy_idx = best_offset + tail_size;
                        } 
                        
                        // Anexa o áudio costurado à prova de lente
                        for (int i = start_copy_idx; i < (int)current_chunk.size(); ++i) {
                            audio_samples.push_back(current_chunk[i]);
                        }
                        
                        // Corta e guarda os últimos 'tail_size' pixéis como molde para o frame seguinte
                        audio_tail.clear();
                        int tail_start = std::max(0, (int)current_chunk.size() - tail_size);
                        for (int i = tail_start; i < (int)current_chunk.size(); ++i) {
                            audio_tail.push_back(current_chunk[i]);
                        }
                    }
                }
                last_perf_y = curr_perf_y;
            }
        }
        // --- FIM DO AUDIO LINE-SCANNER ---

        // --- DETECÇÃO DE BORDA VIA FASE CIRCULAR (PLL) ---
        // Em vez de rastrear furos individuais que podem sumir e confundir o sistema,
        // nós tratamos a película inteira como uma onda contínua.
        // Calculamos a "Fase" média da onda (0 a pitch_padrao) baseada em todos os furos visíveis.
        
        double fase_filme = -1.0;
        bool furo_na_zona_agora = false;
        
        if (!furos_validos.empty()) {
            furo_na_zona_agora = true; // Simplificação para a UI Python
            
            double soma_seno = 0.0;
            double soma_cosseno = 0.0;
            
            for (const auto& f : furos_validos) {
                // Mapeia o cy_global do furo para um ângulo na circunferência do Pitch
                double angulo = (f.cy_g / pitch_padrao) * 2.0 * M_PI;
                soma_seno += std::sin(angulo);
                soma_cosseno += std::cos(angulo);
            }
            
            // Tira a média dos ângulos (isso imuniza o sistema contra furos sumindo e encolhimentos locais!)
            double angulo_medio = std::atan2(soma_seno, soma_cosseno);
            if (angulo_medio < 0) angulo_medio += 2.0 * M_PI;
            
            // Converte de volta para pixels na régua da Fase
            fase_filme = (angulo_medio / (2.0 * M_PI)) * pitch_padrao;
        }
        
        long cx_a = -1, cy_a = -1;
        bool capturar = false;
        
        // A Linha de Gatilho também tem uma Fase fixa
        double fase_gatilho = std::fmod((double)(linha_gatilho_y + roi_rect.y), pitch_padrao);
        
        if (fase_filme >= 0) {
            double erro_fase = fase_gatilho - fase_filme;
            
            // Normaliza o erro para o caminho mais curto [-pitch/2 a +pitch/2]
            while (erro_fase < -(pitch_padrao * 0.5)) erro_fase += pitch_padrao;
            while (erro_fase >=  (pitch_padrao * 0.5)) erro_fase -= pitch_padrao;
            
            // Histerese Matemática: Arma o gatilho quando a fase do filme está "atrás" 
            // da linha de gatilho em pelo menos 15% do pitch (ex: ~40 pixels de distância).
            if (erro_fase > pitch_padrao * 0.15) {
                gatilho_fase_armado = true;
            }
            
            // DISPARO: Ocorre exatamente quando o erro cruza o zero (de positivo para negativo).
            // Isso significa que a fase do filme acabou de ultrapassar a fase do gatilho!
            if (gatilho_fase_armado && last_erro_fase > 0 && erro_fase <= 0) {
                gatilho_fase_armado = false; // Desarma
                perfuracao_na_linha = true;
                contador_perfs_ciclo++;
                
                if(contador_perfs_ciclo >= 4) {
                    int qtd = std::min(4, (int)furos_validos.size());
                    
                    long sum_cx = 0;
                    for(int i=0; i<qtd; i++) sum_cx += furos_validos[i].cx_g;
                    cx_a = sum_cx / std::max(1, qtd);
                    
                    // Cálculo do pitch dinâmico (mantido do legado Python para telemetria)
                    if(qtd > 1) {
                        double soma_pitch = 0;
                        for(int i=1; i<qtd; i++) soma_pitch += (furos_validos[i].cy_g - furos_validos[i-1].cy_g);
                        double pitch_instantaneo = soma_pitch / (qtd - 1);
                        ultimo_pitch_instantaneo = pitch_instantaneo;
                        
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
                    }
                    
                    // ÂNCORA VERTICAL DE ALTA PRECISÃO (Sub-pixel PLL Tracking):
                    // Para absorver os DROPS de frame da USB, o crop não pode ser fixo.
                    // Se o frame atrasou 30ms, o filme andou. O 'erro_fase' contém exatamente 
                    // a quantidade de pixels que o filme "passou" da linha de gatilho nesse atraso!
                    // Subtraindo o erro_fase (que é negativo após cruzar), nós empurramos 
                    // o crop exatamente para onde o buraco foi parar, mantendo o quadro perfeitamente imóvel!
                    cy_a = (long)std::round((linha_gatilho_y + roi_rect.y) - erro_fase);
                    capturar = true;
                    contador_perfs_ciclo = 0;
                }
            } else {
                perfuracao_na_linha = false;
            }
            
            last_erro_fase = erro_fase;
        } else {
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
        result["pitch_instantaneo"] = ultimo_pitch_instantaneo;
        result["achou_furo"] = furo_na_zona_agora;
        result["audio_chunk"] = audio_numpy; 
        
        return result;
    }

    void reset_ciclo() {
        contador_perfs_ciclo = 0;
        last_perf_y = -1.0;
        gatilho_fase_armado = false;
        last_erro_fase = 0.0;
        audio_tail.clear();
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
             py::arg("audio_enabled") = false, py::arg("audio_x") = 0, py::arg("audio_w") = 0, py::arg("audio_slit_y") = 0)
        .def("reset_ciclo", &ScannerVision::reset_ciclo);
}
