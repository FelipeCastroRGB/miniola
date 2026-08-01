import sys

class ScannerStateProxy:
    """
    Este proxy permite que as rotas da pasta web/ acessem e modifiquem
    as variáveis globais do miniola.py (que roda como __main__) sem precisar
    refatorar todo o código legado de uma só vez.
    
    No futuro, quando o miniola.py for refatorado, esta classe se tornará
    um repositório de dados real (Single Source of Truth).
    """
    def __getattr__(self, name):
        import __main__
        if hasattr(__main__, name):
            return getattr(__main__, name)
        raise AttributeError(f"ScannerStateProxy: O orquestrador não possui a variável '{name}'")
        
    def __setattr__(self, name, value):
        import __main__
        if hasattr(__main__, name):
            setattr(__main__, name, value)
        else:
            # Se não existir no main, salva localmente por segurança
            super().__setattr__(name, value)

state = ScannerStateProxy()
