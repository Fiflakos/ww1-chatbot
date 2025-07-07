class MemoryRetriever:
    def __init__(self, texts, file_names):
        self.texts = texts
        self.file_names = file_names

    def retrieve(self, query, top_k=5):
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
        import numpy as np

        vectorizer = TfidfVectorizer().fit(self.texts)
        query_vec = vectorizer.transform([query])
        doc_vecs = vectorizer.transform(self.texts)

        scores = cosine_similarity(query_vec, doc_vecs)[0]
        top_indices = np.argsort(scores)[::-1][:top_k]

        return [(self.texts[i], self.file_names[i], float(scores[i])) for i in top_indices]
