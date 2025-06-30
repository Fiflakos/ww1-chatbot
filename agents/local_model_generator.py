from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

MODEL_PATH = "models/hf_generator"

class LocalHistBERTGenerator:
    """
    Loads your fine-tuned Seq2Seq model and generates answers
    given a question + top‐k retrieved passages.
    """

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        self.model     = AutoModelForSeq2SeqLM.from_pretrained(MODEL_PATH).to(self.device)

    def generate(self, question: str, hits: list):
        # build context from top‐k snippets
        context = ""
        for fn, score, snippet in hits:
            context += f"{fn}:\n{snippet}\n\n"

        prompt = (
            "Answer the question based on the following WW1 letters or diaries.\n\n"
            f"Context:\n{context}\n"
            f"Question: {question}"
        )

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            padding="longest",
        ).to(self.device)

        outputs = self.model.generate(
            inputs["input_ids"],
            max_length=256,
            num_beams=4,
            early_stopping=True,
        )

        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
