import streamlit as st

st.title("Módulo 2: Clasificación de Imágenes (Conducción Distraída)")
st.write("Sube una imagen para predecir el comportamiento del conductor usando la CNN (ResNet18 / MobileNetV2).")

uploaded_file = st.file_uploader("Sube una imagen...", type=["jpg", "png", "jpeg"])
if uploaded_file is not None:
    st.image(uploaded_file, caption="Imagen cargada", use_container_width=True)
    st.write("Realizando inferencia...")
    # Aquí iría el código de inferencia:
    # tensor = transformar_imagen(uploaded_file)
    # pred = modelo(tensor)
    # st.write(f"Predicción: {pred}")
