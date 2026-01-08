import os
from typing import TypedDict, Annotated, List
from dotenv import load_dotenv

import gradio as gr
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, END

# 1. Configuración de entorno
load_dotenv()

# Verificar que la API KEY esté configurada
if not os.getenv("GOOGLE_API_KEY"):
    print("Error: GOOGLE_API_KEY no encontrada en el archivo .env")

# 2. Definición del Estado del Agente
class AgentState(TypedDict):
    # Usamos una lista de mensajes para mantener el historial
    messages: Annotated[List[BaseMessage], "The list of messages in the conversation"]

# 3. Inicialización del Modelo Gemini
# Usamos 'gemini-1.5-flash-8b' que es el modelo más económico y rápido actualmente.
# Si tienes problemas, prueba con 'gemini-pro' o 'gemini-1.5-flash'.
llm = ChatGoogleGenerativeAI(model=os.getenv("GEMINI_MODEL"))

# 4. Definición del Nodo de Procesamiento (IA)
def call_model(state: AgentState):
    """
    Nodo que toma los mensajes actuales, consulta a Gemini y devuelve la respuesta.
    """
    messages = state["messages"]
    response = llm.invoke(messages)
    # Retornamos la actualización del estado (el nuevo mensaje del AI)
    return {"messages": [response]}

# 5. Construcción del Grafo (LangGraph)
workflow = StateGraph(AgentState)

# Añadimos el nodo de procesamiento
workflow.add_node("agent", call_model)

# Definimos el flujo: Entrada -> Nodo Agente -> Fin
workflow.set_entry_point("agent")
workflow.add_edge("agent", END)

# Compilamos el grafo
app = workflow.compile()

# 6. Función para la Interfaz de Gradio
def chat_with_agent(message, history):
    """
    Función que conecta la interfaz de Gradio con el flujo de LangGraph.
    """
    # Convertimos el historial de Gradio y el nuevo mensaje al formato de LangChain
    input_messages = []
    for human, ai in history:
        input_messages.append(HumanMessage(content=human))
        input_messages.append(AIMessage(content=ai))
    
    input_messages.append(HumanMessage(content=message))
    
    # Ejecutamos el grafo
    # El estado inicial solo contiene los mensajes acumulados
    config = {"configurable": {"thread_id": "1"}}
    output = app.invoke({"messages": input_messages}, config)
    
    # Extraemos el último mensaje (la respuesta de la IA)
    return output["messages"][-1].content

# 7. Lanzamiento de la Interfaz Web con Gradio
demo = gr.ChatInterface(
    fn=chat_with_agent,
    title="Chatbot IA con LangGraph y Gemini",
    description="Un agente inteligente que utiliza LangGraph para gestionar el flujo de conversación.",
    examples=["¿Qué es LangGraph?", "¿Cómo funciona el modelo Gemini?", "Cuéntame un chiste corto."]
)

if __name__ == "__main__":
    # Launch local server
    demo.launch()
