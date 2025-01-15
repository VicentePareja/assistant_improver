# main.py

import importlib
import re
import sys
import time

nombres_con_gdocs = [
    #"MyU",
    "Ai Bot You",
    #"Spencer Consulting",
    #"Oncoprecisión",
    #"Laboratorio Biomed",
    #"Trayecto Bookstore",
    #"Ortodoncia de la Fuente",
    #"KLIK Muebles",
    #"Nomad Genetics",
    "House of Spencer"
]

def actualizar_parameters(nombre):
    """
    Actualiza dinámicamente el archivo parameters.py con el nuevo nombre de asistente.
    """
    with open("parameters.py", "r", encoding="utf-8") as f:
        contenido = f.read()

    # Reemplaza la línea ASSISTANT_NAME="..." con la nueva
    nuevo_contenido = re.sub(
        r'(ASSISTANT_NAME\s*=\s*")[^"]+"',
        f'\\1{nombre}"',
        contenido
    )

    with open("parameters.py", "w", encoding="utf-8") as f:
        f.write(nuevo_contenido)

    print(f"parameters.py actualizado para: {nombre}")

def ejecutar_para_todos():
    for nombre in nombres_con_gdocs:
        # 1) Sobrescribir parameters.py
        actualizar_parameters(nombre)

        try:
            # 2) Forzar la recarga de "parameters"
            importlib.invalidate_caches()
            if "parameters" in sys.modules:
                del sys.modules["parameters"]
            parameters = importlib.import_module("parameters")
            importlib.reload(parameters)

            # 3) Forzar la recarga de "assistant_improver"
            if "src.assistant_improver.assistant_improver" in sys.modules:
                del sys.modules["src.assistant_improver.assistant_improver"]
            assistant_improver_module = importlib.import_module("src.assistant_improver.assistant_improver")
            importlib.reload(assistant_improver_module)

            # 4) Ahora, extraer la clase AssistantImprover y usarla
            AssistantImprover = assistant_improver_module.AssistantImprover

            print(f"Corriendo para: {parameters.ASSISTANT_NAME}")

            main_app = AssistantImprover()
            main_app.run()

        except Exception as e:
            print(f"Error al procesar {nombre}: {e}")

if __name__ == "__main__":
    initial_time = time.time()
    ejecutar_para_todos()
    final_time = time.time()
    print(f"Tiempo total de todas las ejecuciones: {final_time - initial_time} segundos.")