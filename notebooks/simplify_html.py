import re
import os

HTML_PATH = r"reporte_tecnico_rutaviva.html"
OUTPUT_PATH = r"reporte/reporte_simplificado.html"

print("Simplificando reporte técnico...")

if not os.path.exists(HTML_PATH):
    print(f"Error: {HTML_PATH} no existe.")
    exit(1)

with open(HTML_PATH, 'r', encoding='utf-8', errors='ignore') as f:
    html = f.read()

# Replace base64 image sources
clean_html = re.sub(r'src="data:image/[^"]+"', 'src="[IMAGE_EXPORT_PENDING]"', html)
# Also clean any other potential huge strings
clean_html = re.sub(r'src=\'data:image/[^\']+\'', 'src="[IMAGE_EXPORT_PENDING]"', clean_html)

os.makedirs("reporte", exist_ok=True)
with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
    f.write(clean_html)

print(f"Reporte simplificado guardado en {OUTPUT_PATH}")
print(f"Tamaño original: {os.path.getsize(HTML_PATH):,} bytes")
print(f"Tamaño simplificado: {os.path.getsize(OUTPUT_PATH):,} bytes")
