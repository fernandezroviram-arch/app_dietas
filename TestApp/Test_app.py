import streamlit as st
import numpy as np
import pandas as pd
from scipy.optimize import nnls

# ==========================================
# 1. BASE DE DATOS LOCAL (Simulación de API)
# ==========================================
# Valores nutricionales por cada 100 gramos de producto
datos_alimentos = {
    "Pechuga de Pollo": {"Categoria": "Carnes", "Prot": 23.1, "Carb": 0.0, "Grasa": 1.2},
    "Ternera Magra": {"Categoria": "Carnes", "Prot": 21.0, "Carb": 0.0, "Grasa": 5.0},
    "Salmón": {"Categoria": "Pescados", "Prot": 20.0, "Carb": 0.0, "Grasa": 13.0},
    "Arroz Blanco": {"Categoria": "Carbohidratos", "Prot": 2.7, "Carb": 28.0, "Grasa": 0.3},
    "Patata Cocida": {"Categoria": "Carbohidratos", "Prot": 2.0, "Carb": 20.0, "Grasa": 0.1},
    "Brócoli": {"Categoria": "Verduras", "Prot": 2.8, "Carb": 4.0, "Grasa": 0.4},
    "Aceite de Oliva": {"Categoria": "Grasas", "Prot": 0.0, "Carb": 0.0, "Grasa": 100.0},
}

# Convertimos el diccionario a un DataFrame de Pandas para manejarlo fácilmente
df_db = pd.DataFrame(datos_alimentos).T

# ==========================================
# 2. INTERFAZ DE USUARIO (Frontend con Streamlit)
# ==========================================
st.set_page_config(page_title="Optimizador de Macros", layout="wide")
st.title("⚖️ Optimizador de Dietas (Motor Matricial)")

# --- SECCIÓN A: Objetivos ---
st.sidebar.header("🎯 Tus Objetivos Diarios")
obj_prot = st.sidebar.number_input("Proteína (g)", min_value=0, value=150)
obj_carb = st.sidebar.number_input("Carbohidratos (g)", min_value=0, value=200)
obj_grasa = st.sidebar.number_input("Grasas (g)", min_value=0, value=60)

# --- SECCIÓN B: Selección de Alimentos ---
st.header("🛒 Selecciona tus alimentos")
col1, col2, col3 = st.columns(3)

alimentos_seleccionados = []

with col1:
    st.subheader("🥩 Proteínas")
    if st.checkbox("Pechuga de Pollo"): alimentos_seleccionados.append("Pechuga de Pollo")
    if st.checkbox("Ternera Magra"): alimentos_seleccionados.append("Ternera Magra")
    if st.checkbox("Salmón"): alimentos_seleccionados.append("Salmón")

with col2:
    st.subheader("🍚 Carbohidratos")
    if st.checkbox("Arroz Blanco"): alimentos_seleccionados.append("Arroz Blanco")
    if st.checkbox("Patata Cocida"): alimentos_seleccionados.append("Patata Cocida")

with col3:
    st.subheader("🥑 Verduras y Grasas")
    if st.checkbox("Brócoli"): alimentos_seleccionados.append("Brócoli")
    if st.checkbox("Aceite de Oliva"): alimentos_seleccionados.append("Aceite de Oliva")


# ==========================================
# 3. MOTOR MATEMÁTICO (Backend)
# ==========================================
st.divider()

if st.button("Calcular Cantidades Óptimas", type="primary"):
    if len(alimentos_seleccionados) < 3:
        st.warning("Selecciona al menos 3 alimentos distintos (idealmente fuentes de proteína, carbos y grasas) para que el sistema tenga grados de libertad.")
    else:
        # Filtramos la base de datos solo con lo que el usuario ha elegido
        df_select = df_db.loc[alimentos_seleccionados]
        
        # Construimos la Matriz A (Macros aportados por cada 1 gramo de alimento)
        # Dividimos entre 100 porque nuestra BD está en base a 100g
        A = np.array([
            df_select["Prot"].values / 100.0,
            df_select["Carb"].values / 100.0,
            df_select["Grasa"].values / 100.0
        ])
        
        # Construimos el Vector b (Nuestros objetivos)
        b = np.array([obj_prot, obj_carb, obj_grasa])
        
        # Resolvemos el sistema mediante Mínimos Cuadrados No Negativos (NNLS)
        x, error_residual = nnls(A, b)
        
        # ==========================================
        # 4. PRESENTACIÓN DE RESULTADOS
        # ==========================================
        st.subheader("📊 Cantidades Calculadas")
        
        # Mostramos los gramos a comer
        resultados = pd.DataFrame({
            "Alimento": alimentos_seleccionados,
            "Cantidad a comer (gramos)": np.round(x, 1) # Redondeamos a 1 decimal
        })
        # Filtramos para no mostrar alimentos con 0 gramos
        resultados = resultados[resultados["Cantidad a comer (gramos)"] > 0]
        st.dataframe(resultados, hide_index=True, use_container_width=True)
        
        # Comprobación (¿Cuánto hemos conseguido realmente?)
        macros_conseguidos = A.dot(x)
        st.caption(f"**Macros alcanzados matemáticamente:** Proteína: {macros_conseguidos[0]:.1f}g | Carbos: {macros_conseguidos[1]:.1f}g | Grasas: {macros_conseguidos[2]:.1f}g")