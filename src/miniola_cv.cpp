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

public:
    ScannerVision() {}

    py::dict process_frame(py::array_t<uint8_t> input_array,
                           int roi_x, int roi_y, int roi_w, int roi_h,
                           int thresh_val, int linha_gatilho_y, int margem_gatilho,
                           double pitch_padrao) {
        
        py::buffer_info buf = input_array.request();
        int rows = buf.shape[0];
        int cols = buf.shape[1];
        
        // frame_raw em RGB888 -> 3 canais
        cv::Mat frame(rows, cols, CV_8UC3, buf.ptr);
        
        // Proteção contra falhas de limites da ROI
        cv::Rect roi_rect(std::max(0, roi_x), std::max(0, roi_y),
                          std::min(roi_w, cols - roi_x), std::min(roi_h, rows - roi_y));
                          
        if (roi_rect.width <= 0 || roi_rect.height <= 0) {
            py::dict err; err["capturar"] = false; return err;
        }

        cv::Mat roi_color = frame(roi_rect);
        cv::Mat roi_gray, roi_small, binary_small;
        cv::cvtColor(roi_color, roi_gray, cv::COLOR_RGB2GRAY);
        cv::resize(roi_gray, roi_small, cv::Size(), 0.5, 0.5);
        cv::threshold(roi_small, binary_small, thresh_val, 255, cv::THRESH_BINARY);
        
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
            double area_aprox = (w_s * h_s) * 4.0;
            
            if(area_aprox > 200 && area_aprox < 10000 && (w_s/h_s) > 0.2 && (w_s/h_s) < 2.5) {
                int cy_roi = (rect.y * 2) + ((rect.height * 2) / 2);
                int cx_global = (rect.x * 2) + ((rect.width * 2) / 2) + roi_rect.x;
                int cy_global = cy_roi + roi_rect.y;
                
                bool acionou = (cy_roi >= limite_superior && cy_roi <= limite_inferior);
                furos_validos.push_back({cy_roi, cx_global, cy_global, acionou, rect});
                
                py::dict debug_item;
                debug_item["rect"] = py::make_tuple(rect.x*2 + roi_rect.x, rect.y*2 + roi_rect.y, rect.width*2, rect.height*2);
                // Mesmo comportamento do Python: RGB Red=(0,0,255), Green=(0,255,0)
                debug_item["color"] = acionou ? py::make_tuple(0, 0, 255) : py::make_tuple(0, 255, 0); 
                debug_visual.append(debug_item);
            }
        }
        
        std::sort(furos_validos.begin(), furos_validos.end(), [](const Furo& a, const Furo& b) {
            return a.cy_roi < b.cy_roi;
        });
        
        bool furo_detectado_agora = false;
        long cx_a = -1, cy_a = -1;
        bool capturar = false;
        
        if(!furos_validos.empty() && furos_validos[0].acionou) {
            furo_detectado_agora = true;
            if(!perfuracao_na_linha) {
                contador_perfs_ciclo++;
                perfuracao_na_linha = true;
                
                if(contador_perfs_ciclo >= 4) {
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
        
        // Copia standalone do Numpy da Matriz binária para desacoplar da vida da cv::Mat
        py::array_t<uint8_t> result_array({binary_small.rows, binary_small.cols});
        py::buffer_info buf_res = result_array.request();
        std::memcpy(buf_res.ptr, binary_small.data, binary_small.total() * binary_small.elemSize());
        
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
        result["achou_furo"] = furo_detectado_agora; // Ajuda para bypass no frontend logico
        
        return result;
    }
};

PYBIND11_MODULE(miniola_cv, m) {
    m.doc() = "Miniola CV Extension using OpenCV and Pybind11";
    py::class_<ScannerVision>(m, "ScannerVision")
        .def(py::init<>())
        .def("process_frame", &ScannerVision::process_frame);
}
