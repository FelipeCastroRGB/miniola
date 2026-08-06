import serial
import serial.tools.list_ports
import time
import sys

def find_pico_port():
    """Procura automaticamente por uma porta COM que possa ser a SKR Pico"""
    ports = serial.tools.list_ports.comports()
    for port in ports:
        # A SKR Pico (RP2040) geralmente aparece como um dispositivo USB Serial
        if "USB" in port.description or "Serial" in port.description:
            return port.device
    
    # Fallback se não encontrar por nome
    if ports:
        return ports[0].device
    return None

def test_encoder():
    print("=== TESTE DO ENCODER (MINIOLA) ===")
    
    port = find_pico_port()
    if not port:
        print("[ERRO] Nenhuma porta COM encontrada! Verifique o cabo USB.")
        sys.exit(1)
        
    print(f"Tentando conectar na SKR Pico na porta: {port}")
    
    try:
        ser = serial.Serial(port, 115200, timeout=1)
        print(f"[SUCESSO] Conectado na {port}!")
        print("Gire o rolete do encoder... (Pressione Ctrl+C para sair)")
        print("-" * 50)
        
        while True:
            if ser.in_waiting > 0:
                line = ser.readline().decode('utf-8', errors='ignore').strip()
                if line.startswith("E "):
                    pulses = line.split(" ")[1]
                    # O carriage return (\r) faz a linha sobrescrever ela mesma, 
                    # dando um efeito visual de contador limpo.
                    sys.stdout.write(f"\rPulsos lidos: {pulses}         ")
                    sys.stdout.flush()
            time.sleep(0.01)
            
    except serial.SerialException as e:
        print(f"\n[ERRO] Não foi possível abrir a porta {port}: {e}")
    except KeyboardInterrupt:
        print("\n\nTeste finalizado.")
        if 'ser' in locals() and ser.is_open:
            ser.close()

if __name__ == "__main__":
    test_encoder()
