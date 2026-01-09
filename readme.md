# ia-langgraph-agent-gemini
este es un chat conversatorio usando un agente langgraph y llm para las respuestas. para la interfaz de usuario se usa gradio.


## Instalación
Crear ambiente virtual:
```
python3 -m venv .venv
source venv/bin/activate
```

instalar dependencias:
```
pip install -r requirements.txt
````

## configuración
crear archivo .env con las variables de entorno
```
GOOGLE_API_KEY=<tu_api_key>
GEMINI_MODEL=<nombre del modelo>
```

## Ejecución
correr el archivo main.py en terminal
```
python main.py
```

## Visualización
abrir navegador en:
```
http://localhost:7860
```


