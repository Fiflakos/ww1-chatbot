class GraphRetriever:
    def __init__(self, texts, file_names):
        self.texts = texts
        self.file_names = file_names

    def retrieve(self, query, top_k=5):
        import numpy as np
        results = [(self.texts[i], self.file_names[i], 1.0 - (i / len(self.texts))) for i in range(min(top_k, len(self.texts)))]
        return results
