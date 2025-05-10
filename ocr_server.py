from flask import Flask, request, send_file
from PIL import Image
import os
import tempfile
import pandas as pd
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import torch

app = Flask(__name__)

processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")

@app.route("/ocr", methods=["POST"])
def ocr():
    files = request.files.getlist("images")
    rows = []

    for file in files:
        image = Image.open(file.stream).convert("RGB")
        pixel_values = processor(images=image, return_tensors="pt").pixel_values
        generated_ids = model.generate(pixel_values)
        text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

        rows.append({
            "Source File": file.filename,
            "Extracted Text": text
        })

    output_path = os.path.join(tempfile.gettempdir(), "Invoices_Summary.xlsx")
    pd.DataFrame(rows).to_excel(output_path, index=False)
    return send_file(output_path, as_attachment=True)

@app.route("/", methods=["GET"])
def home():
    return "✅ TrOCR OCR Server Running"

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
