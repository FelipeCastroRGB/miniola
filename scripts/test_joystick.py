import evdev
import sys

def main():
    devices = [evdev.InputDevice(path) for path in evdev.list_devices()]
    gamepad = None
    
    for d in devices:
        name = d.name.lower()
        if "joystick" in name or "gamepad" in name or "usb" in name:
            if "keyboard" not in name and "mouse" not in name:
                gamepad = d
                break
                
    if not gamepad:
        print("Nenhum controle Knup ou USB Genérico encontrado!")
        sys.exit(1)
        
    print("==================================================")
    print(f"CONTROLE DETECTADO: {gamepad.name}")
    print("==================================================")
    print("Aperte os botões ou gatilhos para descobrir seus nomes internos.")
    print("Aperte Ctrl+C para sair.\n")
    
    try:
        for event in gamepad.read_loop():
            if event.type == evdev.ecodes.EV_KEY:
                # Botões
                c = evdev.categorize(event)
                print(f"[BOTÃO DIGITAL] Nome: {c.keycode} | Valor: {c.keystate} | Código Raw: {event.code}")
            elif event.type == evdev.ecodes.EV_ABS:
                # Analógicos
                # Filtra ruído do zero central
                if abs(event.value) > 10 and event.value != 128 and event.value != 127: 
                    # evdev.ecodes.ABS é um dict que mapeia int -> str ou list de str
                    name = evdev.ecodes.ABS.get(event.code, "DESCONHECIDO")
                    print(f"[EIXO ANALÓGICO] Nome: {name} | Valor: {event.value} | Código Raw: {event.code}")
    except KeyboardInterrupt:
        print("\nSaindo...")

if __name__ == "__main__":
    main()
