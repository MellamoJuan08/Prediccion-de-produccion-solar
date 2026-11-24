from flask import Flask, render_template, request, jsonify, url_for, redirect
import joblib
import pandas as pd
import numpy as np
from datetime import datetime
import os
import json
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import io
import base64
from openai import OpenAI

# ========================
# CONFIGURACIÓN DE FLASK
# ========================
app = Flask(__name__)

# ======== CARGAR MODELOS ========
modelo_horas = joblib.load("C:\\Users\\Juan\\prediccion_produccion_solar\\solar_components\\modelo_solar (1).pkl")
modelo_mensual = joblib.load("C:\\Users\\Juan\\prediccion_produccion_solar\\solar_components\\modelo_entrenado_mensual.pkl")
modelo_anual = joblib.load("C:\\Users\\Juan\\prediccion_produccion_solar\\solar_components\\modelo_entrenado_anual.pkl")

# ======== CLIENTE GPT ========
client = OpenAI(
    api_key="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",
    base_url="https://openrouter.ai/api/v1"
)

# ======== CARPETA PARA GUARDAR JSON ========
CARPETA_JSON = "predicciones_json"
os.makedirs(CARPETA_JSON, exist_ok=True)


# ===========================================================
# FUNCIONES AUXILIARES
# ===========================================================
def determinar_avatar_y_estado(potencias):
    if not potencias or len(potencias) == 0:
        return "neutral", "¡Hola! Estoy listo para analizar tu producción solar."

    max_potencia = np.max(potencias)
    avg_potencia = np.mean(potencias)

    if max_potencia > 800:
        return "feliz", "¡Excelente! Los datos muestran una producción solar óptima."
    elif max_potencia > 500:
        return "contento", "Buena producción solar detectada en el análisis."
    elif max_potencia > 200:
        return "pensativo", "Producción moderada. Podríamos optimizar algunos factores."
    else:
        return "preocupado", "Producción baja detectada. Veamos cómo mejorar."


def generar_grafica_con_efecto(x, y, x_usuario=None, y_usuario=None, titulo_eje_x="Hora del día", tipo="horas"):
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor('#f8f9fa')
    ax.set_facecolor("#ffffff")

    # Suavizar la línea para gráficas mensuales y anuales
    if tipo in ["mensual", "anual"]:
        # Crear puntos interpolados para una línea más suave
        x_smooth = np.linspace(x.min(), x.max(), 300)
        y_smooth = np.interp(x_smooth, x, y)
        ax.plot(x_smooth, y_smooth, color='#FF6B35', linewidth=3.5, 
                label="Potencia Solar", alpha=0.8,
                path_effects=[pe.SimpleLineShadow(offset=(3, -3), alpha=0.3), pe.Normal()])
        # Agregar puntos originales como marcadores
        ax.scatter(x, y, color='#FF6B35', s=50, alpha=0.6, zorder=5)
    else:
        ax.plot(x, y, color='#FF6B35', linewidth=3.5, label="Potencia Solar",
                path_effects=[pe.SimpleLineShadow(offset=(3, -3), alpha=0.3), pe.Normal()])

    ax.fill_between(x, y, alpha=0.2, color='#FF6B35')

    max_idx = np.argmax(y)
    ax.plot(
        x[max_idx], y[max_idx], 'o', markersize=8, color='red',
        label=f'Máximo: {y[max_idx]:.1f}W'
    )

    if x_usuario is not None and y_usuario is not None:
        ax.plot(
            x_usuario, y_usuario, 's', markersize=10,
            color='blue', label=f'Tu dato: {y_usuario:.1f}W',
            markerfacecolor='yellow', markeredgecolor='blue', markeredgewidth=2
        )

    # Personalizar título según el tipo
    if tipo == "horas":
        titulo = "🔆 Predicción de Potencia Solar Horaria"
        etiqueta_x = "Hora del día"
    elif tipo == "mensual":
        titulo = "📅 Predicción de Potencia Solar Mensual"
        etiqueta_x = "Meses del año"
    else:  # anual
        titulo = "📊 Predicción de Potencia Solar Anual"
        etiqueta_x = "Años"

    ax.set_title(titulo, fontsize=16, weight="bold", pad=20)
    ax.set_xlabel(etiqueta_x, fontsize=12, fontweight='bold')
    ax.set_ylabel("Potencia PV (W)", fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='upper right')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight", dpi=120, facecolor='#f8f9fa', edgecolor='none')
    buf.seek(0)
    grafica_url = base64.b64encode(buf.getvalue()).decode()
    buf.close()
    plt.close(fig)
    return grafica_url


# ===========================================================
# RUTAS PRINCIPALES
# ===========================================================
@app.route("/")
def index():
    # En la vista principal aún no hay predicciones
    return render_template(
        "index.html",
        tipo="",
        resultado=None,
        grafica_url=None,
        hora_val="",
        iluminancia_val="",
        temperatura_val="",
        mes_val="",
        anio_val="",
        avatar_actual=url_for('static', filename='avatar_neutral.png'),
        mensaje_avatar="¡Hola! Estoy listo para analizar tu producción solar.",
        predicciones=[],
        avatar_analizando=False,
        prediccion_puntual=None,
        valor_x_usuario=None
    )


# ===========================================================
# RUTAS DE PREDICCIÓN POR TIPO
# ===========================================================
@app.route("/predictor/horas", methods=["GET", "POST"])
def predictor_horas():
    return ejecutar_prediccion(
        modelo=modelo_horas,
        col_x="Hora",
        campo_x="hora",
        campo_ilum="iluminancia",
        campo_temp="temperatura",
        etiqueta_x="Hora del día",
        tipo_prediccion="horas"
    )


@app.route("/predictor/mensual", methods=["GET", "POST"])
def predictor_mensual():
    return ejecutar_prediccion(
        modelo=modelo_mensual,
        col_x="Dia_del_año",
        campo_x="mes",
        campo_ilum="iluminancia", 
        campo_temp="temperatura",
        etiqueta_x="Meses del año",
        tipo_prediccion="mensual"
    )


@app.route("/predictor/anual", methods=["GET", "POST"])
def predictor_anual():
    return ejecutar_prediccion(
        modelo=modelo_anual,
        col_x="Dia_del_año",
        campo_x="anio",
        campo_ilum="iluminancia",
        campo_temp="temperatura",
        etiqueta_x="Años",
        tipo_prediccion="anual"
    )


# ===========================================================
# RUTAS PARA LIMPIAR PREDICCIONES
# ===========================================================
@app.route("/limpiar/horas")
def limpiar_horas():
    return redirect("/predictor/horas")

@app.route("/limpiar/mensual")
def limpiar_mensual():
    return redirect("/predictor/mensual")

@app.route("/limpiar/anual")
def limpiar_anual():
    return redirect("/predictor/anual")


# ===========================================================
# FUNCIÓN CENTRAL DE PREDICCIÓN (MEJORADA CON VARIACIÓN NATURAL)
# ===========================================================
def ejecutar_prediccion(modelo, col_x, campo_x, campo_ilum, campo_temp, etiqueta_x, tipo_prediccion):
    predicciones_list = []
    resultado = None
    grafica_url = None
    valor_x = ""
    iluminancia_user = ""
    temperatura_user = ""
    avatar_estado = "neutral"
    mensaje_avatar = "¡Hola! Estoy listo para analizar tu producción solar."
    avatar_analizando = False
    prediccion_puntual_val = None
    valor_x_usuario = None

    # Verificar si es una solicitud GET (limpiar) o POST (calcular)
    if request.method == "POST":
        try:
            valor_x = request.form.get(campo_x, "").strip()
            iluminancia_user = request.form.get(campo_ilum, "").strip()
            temperatura_user = request.form.get(campo_temp, "").strip()

            if not valor_x or not iluminancia_user or not temperatura_user:
                raise ValueError("Por favor completa todos los campos antes de enviar.")

            iluminancia_base = float(iluminancia_user)
            temperatura_base = float(temperatura_user)
            
            # Convertir valor_x según el tipo
            if tipo_prediccion == "horas":
                # Convertir hora HH:MM a decimal de forma segura
                if ':' in valor_x:
                    try:
                        partes = valor_x.split(':')
                        horas = float(partes[0])
                        minutos = float(partes[1]) if len(partes) > 1 else 0
                        x_valor = horas + minutos / 60
                    except (ValueError, IndexError):
                        raise ValueError(f"Formato de hora inválido: {valor_x}. Use HH:MM")
                else:
                    try:
                        x_valor = float(valor_x)
                    except ValueError:
                        raise ValueError(f"Valor de hora inválido: {valor_x}")
                        
            elif tipo_prediccion == "mensual":
                # Para modelo mensual, usar el mes ingresado como referencia
                try:
                    mes_usuario = int(valor_x)
                    if mes_usuario < 1 or mes_usuario > 12:
                        raise ValueError("El mes debe estar entre 1 y 12")
                    x_valor = mes_usuario  # Guardamos el mes del usuario
                except ValueError:
                    raise ValueError("Mes inválido. Debe ser un número entre 1 y 12")
                    
            else:  # anual
                # Para modelo anual, usar el año ingresado como referencia (LÍMITE 2018-2025)
                try:
                    anio_usuario = int(valor_x)
                    if anio_usuario < 2018 or anio_usuario > 2025:
                        raise ValueError("El año debe estar entre 2018 y 2025")
                    x_valor = anio_usuario  # Guardamos el año del usuario
                except ValueError:
                    raise ValueError("Año inválido. Debe ser un número entre 2018 y 2025")

            valor_x_usuario = x_valor
            avatar_analizando = True

            # PREDICCIÓN PUNTUAL (para el valor específico del usuario)
            if tipo_prediccion == "horas":
                nuevos_datos = pd.DataFrame([{
                    "Hora": x_valor, 
                    "Iluminancia (lux)": iluminancia_base, 
                    "Temperatura (°C)": temperatura_base
                }])
            else:
                # Para predicción puntual mensual/anual
                if tipo_prediccion == "mensual":
                    dia_del_año_puntual = (x_valor - 1) * 30 + 15  # Mitad del mes
                else:  # anual
                    dia_del_año_puntual = (x_valor - 2018) * 365 + 180  # Mitad del año relativo a 2018
                
                nuevos_datos = pd.DataFrame([{
                    "Dia_del_año": dia_del_año_puntual,
                    "Temperatura (°C)": temperatura_base,
                    "Humedad (%)": 65.0,
                    "Iluminancia (lux)": iluminancia_base
                }])
                column_order = ["Dia_del_año", "Temperatura (°C)", "Humedad (%)", "Iluminancia (lux)"]
                nuevos_datos = nuevos_datos[column_order]

            prediccion_puntual = modelo.predict(nuevos_datos)
            prediccion_puntual_val = float(prediccion_puntual[0])
            resultado = f"🔆 Predicción de Potencia PV: {prediccion_puntual_val:.2f} W"

            # GENERAR PREDICCIONES COMPLETAS SEGÚN EL TIPO CON VARIACIÓN NATURAL
            if tipo_prediccion == "horas":
                # Para horas: predecir para todas las horas del día (0-23)
                rango_x = np.linspace(0, 23, 24)
                # Variación natural para horas del día
                iluminancias = []
                temperaturas = []
                for hora in rango_x:
                    # Iluminancia: más alta al mediodía, más baja en la mañana/tarde
                    ilum_factor = np.sin((hora - 6) * np.pi / 12)  # Pico al mediodía
                    ilum_factor = max(0, ilum_factor)  # No negativa
                    iluminancias.append(iluminancia_base * (0.3 + 0.7 * ilum_factor))
                    
                    # Temperatura: más cálida al mediodía
                    temp_variation = 5 * np.sin((hora - 12) * np.pi / 24)  # Variación de ±5°C
                    temperaturas.append(temperatura_base + temp_variation)
                
                datos_grafica = pd.DataFrame({
                    "Hora": rango_x,
                    "Iluminancia (lux)": iluminancias,
                    "Temperatura (°C)": temperaturas
                })
                
            elif tipo_prediccion == "mensual":
                # Para mensual: predecir para todos los meses del año (1-12)
                rango_x = np.arange(1, 13)  # 12 meses
                
                # Convertir meses a días del año (mitad de cada mes)
                rango_x_dias = [(mes - 1) * 30 + 15 for mes in rango_x]
                
                # Variación natural por estaciones
                iluminancias = []
                temperaturas = []
                for mes in rango_x:
                    # Iluminancia: más alta en verano, más baja en invierno
                    estacional_factor = 0.7 + 0.3 * np.sin((mes - 6) * np.pi / 6)  # Pico en junio
                    iluminancias.append(iluminancia_base * estacional_factor)
                    
                    # Temperatura: variación estacional
                    temp_variation = 8 * np.sin((mes - 6) * np.pi / 6)  # Variación de ±8°C
                    temperaturas.append(temperatura_base + temp_variation)
                
                datos_grafica = pd.DataFrame({
                    "Dia_del_año": rango_x_dias,
                    "Temperatura (°C)": temperaturas,
                    "Humedad (%)": [65.0] * 12,  # Humedad constante
                    "Iluminancia (lux)": iluminancias
                })
                datos_grafica = datos_grafica[["Dia_del_año", "Temperatura (°C)", "Humedad (%)", "Iluminancia (lux)"]]
                
            else:  # anual
                # Para anual: predecir para el rango completo de años (2018-2025)
                rango_x = np.arange(2018, 2026)  # Años 2018 a 2025
                
                # Convertir años a días del año (mitad de cada año, relativo a 2018)
                rango_x_dias = [(anio - 2018) * 365 + 180 for anio in rango_x]
                
                # Variación natural por años (tendencia + variación aleatoria)
                np.random.seed(42)  # Para reproducibilidad
                iluminancias = []
                temperaturas = []
                
                for i, anio in enumerate(rango_x):
                    # Tendencia suave de mejora en iluminancia (tecnología mejorando)
                    tendencia_ilum = 1.0 + (i * 0.02)  # 2% de mejora por año
                    
                    # Variación aleatoria suave
                    variacion_ilum = np.random.normal(0, 0.05)  # ±5% de variación
                    iluminancias.append(iluminancia_base * tendencia_ilum * (1 + variacion_ilum))
                    
                    # Temperatura con variación climática leve
                    variacion_temp = np.random.normal(0, 1.5)  # ±1.5°C de variación
                    temperaturas.append(temperatura_base + variacion_temp)
                
                datos_grafica = pd.DataFrame({
                    "Dia_del_año": rango_x_dias,
                    "Temperatura (°C)": temperaturas,
                    "Humedad (%)": [65.0] * len(rango_x),
                    "Iluminancia (lux)": iluminancias
                })
                datos_grafica = datos_grafica[["Dia_del_año", "Temperatura (°C)", "Humedad (%)", "Iluminancia (lux)"]]

            # Realizar predicciones para el rango completo
            predicciones = modelo.predict(datos_grafica)
            predicciones_list = predicciones.tolist()

            # Ajustar el valor del usuario para la gráfica
            if tipo_prediccion == "mensual":
                x_usuario_grafica = x_valor  # El mes ingresado
            elif tipo_prediccion == "anual":
                x_usuario_grafica = x_valor  # El año ingresado
            else:
                x_usuario_grafica = x_valor  # La hora ingresada

            grafica_url = generar_grafica_con_efecto(
                rango_x, predicciones, 
                x_usuario_grafica,  # Posición del dato del usuario en la gráfica
                prediccion_puntual_val, 
                etiqueta_x, 
                tipo_prediccion
            )

            avatar_estado, mensaje_avatar = determinar_avatar_y_estado(predicciones_list)

        except ValueError as e:
            resultado = f"⚠️ {str(e)}"
            avatar_estado = "preocupado"
            mensaje_avatar = "Parece que hay un error en los datos ingresados."
        except Exception as e:
            resultado = f"⚠️ Ocurrió un error: {e}"
            avatar_estado = "preocupado"
            mensaje_avatar = "¡Ups! Algo salió mal en el análisis."

    # Para GET requests (incluyendo limpiar), mostrar formulario vacío
    avatar_actual = (
        url_for('static', filename='avatar_analizando.png')
        if avatar_analizando
        else url_for('static', filename=f'avatar_{avatar_estado}.png')
    )

    return render_template(
        "index.html",
        tipo=tipo_prediccion,
        resultado=resultado,
        grafica_url=grafica_url,
        hora_val=valor_x if tipo_prediccion == "horas" else "",
        mes_val=valor_x if tipo_prediccion == "mensual" else "",
        anio_val=valor_x if tipo_prediccion == "anual" else "",
        iluminancia_val=iluminancia_user,
        temperatura_val=temperatura_user,
        avatar_actual=avatar_actual,
        mensaje_avatar=mensaje_avatar,
        predicciones=predicciones_list,
        avatar_analizando=avatar_analizando,
        prediccion_puntual=prediccion_puntual_val,
        valor_x_usuario=valor_x_usuario
    )


# ===========================================================
# APIs (GPT, chatbot, JSON)
# ===========================================================
@app.route("/api/analisis_gpt", methods=["POST"])
def analizar_con_gpt():
    try:
        data = request.json
        potencias = data.get("potencias", [])
        tipo = data.get("tipo", "horas")
        
        if not potencias:
            return jsonify({"error": "No se enviaron datos de potencia"}), 400

        potencias_array = np.array(potencias)
        promedio = np.mean(potencias_array)
        maximo = np.max(potencias_array)
        minimo = np.min(potencias_array)
        
        if tipo == "horas":
            hora_max = np.argmax(potencias_array)
            energia_total = np.sum(potencias_array) * (1 / 6)
            contexto = f"análisis horario con pico a las {hora_max:.1f} horas"
        elif tipo == "mensual":
            mes_max = np.argmax(potencias_array) + 1
            energia_total = np.sum(potencias_array)
            contexto = f"análisis mensual con mejor mes el {int(mes_max)}"
        else:  # anual
            anio_max = np.argmax(potencias_array)
            anio_real = 2018 + anio_max  # Convertir índice a año real
            energia_total = np.sum(potencias_array)
            contexto = f"análisis anual con mejor año el {int(anio_real)}"

        prompt = f"""
        Eres SolarBot, un asistente especializado en energía solar.
        📊 DATOS DE PRODUCCIÓN SOLAR ({contexto}):
        - Potencia máxima: {maximo:.1f} W
        - Potencia promedio: {promedio:.1f} W
        - Energía total estimada: {energia_total:.1f} Wh
        - Rango de producción: {minimo:.1f}W - {maximo:.1f}W
        
        Proporciona un análisis breve y recomendaciones prácticas.
        """

        respuesta = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=150,
            temperature=0.7
        )

        texto_gpt = respuesta.choices[0].message.content.strip()
        return jsonify({"comentario": texto_gpt})

    except Exception as e:
        return jsonify({"error": f"Error en el análisis: {str(e)}"}), 500


@app.route("/api/chatbot", methods=["POST"])
def chatbot():
    try:
        data = request.json
        mensaje = data.get("mensaje", "")
        if not mensaje:
            return jsonify({"error": "No se envió ningún mensaje"}), 400

        # Lista de palabras clave relacionadas con energía renovable
        palabras_clave_energia = [
            'solar', 'energía', 'energia', 'panel', 'paneles', 'fotovoltaico', 'fotovoltaica',
            'renovable', 'renovables', 'electricidad', 'potencia', 'producción', 'produccion',
            'luz', 'iluminancia', 'temperatura', 'clima', 'radiación', 'radiacion', 'sol',
            'eficiencia', 'ahorro', 'consumo', 'mantenimiento', 'instalación', 'instalacion',
            'batería', 'bateria', 'inversor', 'watt', 'vatio', 'kilowatt', 'kwh',
            'sostenible', 'sostenibilidad', 'medio ambiente', 'ecológico', 'ecologico',
            'emisiones', 'carbono', 'verde', 'limpia', 'autoconsumo', 'red eléctrica',
            'amperio', 'voltio', 'corriente', 'tensión', 'tension', 'residencial',
            'comercial', 'industrial', 'subvención', 'subvencion', 'incentivo', 'impuesto'
        ]

        # Verificar si el mensaje está relacionado con energía renovable
        mensaje_lower = mensaje.lower()
        es_relacionado = any(palabra in mensaje_lower for palabra in palabras_clave_energia)

        if not es_relacionado:
            # Si no está relacionado, redirigir amablemente al tema de energía renovable
            prompt = f"""
            Eres SolarBot, un asistente especializado en energía solar y renovable. 
            El usuario ha preguntado: "{mensaje}"
            
            Esta pregunta parece estar fuera de mi área principal de especialización. 
            Responde de manera amable y cordial, reconociendo su pregunta pero redirigiendo 
            elegantemente hacia temas de energía solar o renovable.
            
            Mantén un tono positivo y ofrece ayuda relacionada con:
            - Energía solar fotovoltaica
            - Paneles solares y su mantenimiento
            - Eficiencia energética
            - Ahorro en facturas eléctricas
            - Sostenibilidad y energías renovables
            
            No menciones directamente que la pregunta está fuera de tema, sino encuentra 
            una conexión natural con la energía renovable.
            
            Responde en español, de manera breve y amigable.
            """

        else:
            # Si está relacionado, responder normalmente
            prompt = f"""
            Eres SolarBot, asistente experto en energía solar.
            El usuario dice: "{mensaje}"
            
            Responde en español, breve y profesional, enfocándote en temas de energía solar, 
            paneles fotovoltaicos, eficiencia energética y sostenibilidad.
            """

        respuesta = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=200,
            temperature=0.7
        )

        texto_respuesta = respuesta.choices[0].message.content.strip()
        return jsonify({"respuesta": texto_respuesta})

    except Exception as e:
        return jsonify({"error": f"Error en el chatbot: {str(e)}"}), 500


@app.route("/prediccion", methods=["GET"])
def obtener_predicciones_json():
    try:
        archivos = [f for f in os.listdir(CARPETA_JSON) if f.endswith(".json")]
        if not archivos:
            return jsonify({"error": "No hay archivos de predicciones aún"}), 404

        archivo_mas_reciente = max(archivos, key=lambda x: os.path.getmtime(os.path.join(CARPETA_JSON, x)))
        ruta_archivo = os.path.join(CARPETA_JSON, archivo_mas_reciente)
        with open(ruta_archivo, "r", encoding="utf-8") as f:
            data = json.load(f)

        return jsonify(data)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ===========================================================
# EJECUCIÓN
# ===========================================================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)