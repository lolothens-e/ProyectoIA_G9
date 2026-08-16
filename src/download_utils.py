import os
import re
import requests
import urllib3

urllib3.disable_warnings()

def download_file_from_google_drive(file_id: str, destination_path: str, progress_callback=None):
    """
    Descarga un archivo desde Google Drive manejando automáticamente la advertencia
    de análisis de virus para archivos grandes (>100MB).
    """
    session = requests.Session()
    url = "https://drive.google.com/uc?export=download"
    params = {"id": file_id}
    
    # 1. Obtener la página inicial
    res = session.get(url, params=params, stream=True, verify=False)
    
    # Si la respuesta es directamente binaria / no HTML
    content_type = res.headers.get("content-type", "")
    if "text/html" not in content_type:
        download_response = res
    else:
        # 2. Extraer parámetros del formulario de confirmación
        html = res.text
        form_match = re.search(r'<form[^>]+action="([^"]+)"[^>]*>(.*?)</form>', html, re.DOTALL)
        if form_match:
            action_url = form_match.group(1)
            form_content = form_match.group(2)
            inputs = re.findall(r'<input[^>]+name="([^"]+)"[^>]+value="([^"]*)"', form_content)
            data = {k: v for k, v in inputs}
            download_response = session.get(action_url, params=data, stream=True, verify=False)
        else:
            # Intento directo de fallback
            direct_url = f"https://drive.usercontent.google.com/download?id={file_id}&export=download&confirm=t"
            download_response = session.get(direct_url, stream=True, verify=False)

    os.makedirs(os.path.dirname(destination_path), exist_ok=True)
    
    total_size = int(download_response.headers.get("content-length", 0))
    downloaded_size = 0
    chunk_size = 1024 * 1024  # 1 MB
    
    with open(destination_path, "wb") as f:
        for chunk in download_response.iter_content(chunk_size=chunk_size):
            if chunk:
                f.write(chunk)
                downloaded_size += len(chunk)
                if progress_callback and total_size > 0:
                    progress_callback(min(downloaded_size / total_size, 1.0))
                    
    return destination_path

if __name__ == "__main__":
    file_id = "1MpKowch9JG9Qx4AU1JD8D2FUfORM1PXF"
    temp_path = "models/test_baseline_models.pkl"
    print("Iniciando prueba de descarga...")
    download_file_from_google_drive(file_id, temp_path)
    if os.path.exists(temp_path):
        size_mb = os.path.getsize(temp_path) / (1024 * 1024)
        print(f"[OK] Descarga completada con exito. Tamano: {size_mb:.2f} MB")
        os.remove(temp_path)
