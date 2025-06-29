# src/ner/geolocation_infer.py

import json
import os
import logging
import re
from typing import List, Optional
from dotenv import load_dotenv
import google.generativeai as genai

logger = logging.getLogger(__name__)

class GeolocationInfer:
    def __init__(self, llm_enabled=True, api_key_env="GEMINI_API_KEY"):
        self.llm_enabled = llm_enabled
        load_dotenv()  # Load .env file
        self.api_key = os.getenv(api_key_env)
        self.model_name = "gemini-2.0-flash"  # Updated to 2.0 flash

        if self.llm_enabled and self.api_key:
            genai.configure(api_key=self.api_key)
            self.model = genai.GenerativeModel(self.model_name)
            logger.info(f"✅ Gemini client initialized with model: {self.model_name}")
        elif self.llm_enabled:
            logger.warning("⚠️ LLM enabled but GEMINI_API_KEY not set — skipping LLM fallback.")

        self.project_lookup = {
            "Minyari Dome Project": [-22.867, 120.712],
            "Lake Hope Project": [-32.45, 120.15],
        }

    def infer_coordinates(self, project_name: str, context: str) -> Optional[List[float]]:
        coords = self.project_lookup.get(project_name)
        if coords:
            logger.info(f"Found coordinates in DB for {project_name}: {coords}")
            return coords

        if self.llm_enabled and self.model:
            prompt = (
                f"You are a geocoding assistant. Given this snippet:\n\n"
                f"\"{context}\"\n\n"
                f"and knowing the project name is \"{project_name}\", "
                f"return ONLY the two numbers of likely geographic coordinates "
                f"(latitude and longitude) in decimal degrees format. "
                f"If you don't know, respond with 'Unknown'."
            )
            try:
                response = self.model.generate_content(prompt)
                text = response.text.strip()
                logger.info(f"LLM raw response: {text}")
                if "unknown" in text.lower():
                    return None
                latlon = self.extract_coordinates_from_text(text)
                if latlon:
                    logger.info(f"LLM returned coordinates for {project_name}: {latlon}")
                    return latlon
            except Exception as e:
                logger.error("LLM call failed", exc_info=True)

        return None

    def extract_coordinates_from_text(self, text: str) -> Optional[List[float]]:
        matches = re.findall(r"[-+]?\d+\.\d+", text)
        if len(matches) >= 2:
            try:
                return [float(matches[0]), float(matches[1])]
            except:
                return None
        return None

def main():
    logging.basicConfig(level=logging.INFO)
    geo = GeolocationInfer(llm_enabled=True)

    input_file = "output/ner_predictions.jsonl"
    output_file = "output/ner_predictions_with_geo.jsonl"

    with open(input_file, "r", encoding="utf-8") as f_in, \
         open(output_file, "w", encoding="utf-8") as f_out:
        for line in f_in:
            record = json.loads(line)
            if record.get("coordinates") is None:
                coords = geo.infer_coordinates(record["project_name"], record["context_sentence"])
                record["coordinates"] = coords
            f_out.write(json.dumps(record) + "\n")

    logger.info(f"Geolocation inference complete. Results saved to {output_file}")

if __name__ == "__main__":
    main()
