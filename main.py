from src.assistant_improver.assistant_improver import AssistantImprover

import importlib
import re


nombres_con_gdocs = [
    "MyU",
    "Ai Bot You",
    "Spencer Consulting",
    "Oncoprecisión",
    "Laboratorio Biomed",
    "Trayecto Bookstore",
    "Ortodoncia de la Fuente",
    "KLIK Muebles",
    "Nomad Genetics",
    "House of Spencer"
]

def actualizar_parameters(nombre):
    """
    Actualiza dinámicamente el archivo parameters.py con el nuevo nombre de asistente.
    """
    with open("parameters.py", "r", encoding="utf-8") as f:
        contenido = f.read()

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
        # Update parameters.py
        actualizar_parameters(nombre)

        # Reload the parameters module dynamically
        try:
            importlib.invalidate_caches()
            parameters = importlib.import_module("parameters")
            importlib.reload(parameters)

            # Create a new Main instance and run it
            main_app = AssistantImprover()
            main_app.run()
        except Exception as e:
            print(f"Error al procesar {nombre}: {e}")

if __name__ == "__main__":

    ejecutar_para_todos()
