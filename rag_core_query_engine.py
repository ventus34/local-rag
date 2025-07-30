import os
import pickle
import faiss
import requests
from sentence_transformers import SentenceTransformer, CrossEncoder
from typing import Callable, List
import json
from langdetect import detect, LangDetectException
import logging

# --- Logging Configuration to suppress warnings ---
logging.basicConfig()
logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)


class HybridRAGQueryEngine:
    def __init__(self, project_name: str, model_config: dict):
        """
        Initializes the hybrid engine by loading resources for a specific project.
        The engine can operate even if only one type of index (code or docs) is available.
        """
        print(f"Initializing Advanced RAG Engine for project: {project_name}...")

        # --- Model Definitions ---
        self.docs_model_name = model_config.get("docs_model_path", 'BAAI/bge-m3')
        self.code_model_name = model_config.get("code_model_path", 'jinaai/jina-embeddings-v2-base-code')
        self.reranker_model_name = model_config.get("reranker_model_path", 'BAAI/bge-reranker-large')

        # --- Resource Loading ---
        self.has_code = self._load_resources('code', project_name)
        self.has_docs = self._load_resources('docs', project_name)

        if not self.has_code and not self.has_docs:
            raise FileNotFoundError(f"No valid index found for project '{project_name}'. Please run the indexer first.")

        print("Loading Reranker model...")
        self.reranker = CrossEncoder(self.reranker_model_name)
        print("✅ All resources loaded.")

    def _load_resources(self, resource_type: str, project_name: str) -> bool:
        """Helper method to load a set of resources (code or docs)."""
        model_name = getattr(self, f"{resource_type}_model_name")
        index_path = f"{project_name}_{resource_type}_index.faiss"
        chunks_path = f"{project_name}_{resource_type}_chunks.pkl"

        if os.path.exists(index_path) and os.path.exists(chunks_path):
            print(f"Loading {resource_type.upper()} resources...")
            setattr(self, f"{resource_type}_model", SentenceTransformer(model_name))
            setattr(self, f"{resource_type}_index", faiss.read_index(index_path))
            with open(chunks_path, "rb") as f:
                setattr(self, f"{resource_type}_chunks", pickle.load(f))
            return True
        print(f"⚠️  {resource_type.upper()} index for '{project_name}' not found. Proceeding without this search capability.")
        return False

    def _call_llm(self, messages: List[dict], llm_url: str, temperature: float, max_tokens: int) -> str:
        """A single, reusable method to call any OpenAI-compatible API."""
        payload = {
            "model": "local-model",
            "messages": messages,
            "stream": False,
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        print("\n" + "=" * 50)
        print(f"--- PROMPT SENT TO: {llm_url} ---")
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        print("=" * 50 + "\n")
        try:
            response = requests.post(
                llm_url,
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=120
            )
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"]
        except requests.exceptions.RequestException as e:
            return f"Error connecting to LLM at {llm_url}: {e}"
        except (KeyError, IndexError) as e:
            return f"Error parsing LLM response: {e}\nResponse: {response.text}"

    def transform_query_hyde(self, query: str, llm_url: str) -> str:
        """Generates a hypothetical document to improve search relevance (HyDE)."""
        print("Transforming query with HyDE...")
        system_prompt = "Please write a short, clear code snippet or technical explanation that is a perfect answer to the following user question."
        messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": query}]
        hypothetical_answer = self._call_llm(messages, llm_url, temperature=0.3, max_tokens=512)
        if hypothetical_answer.startswith("Error"):
            print(f"HyDE generation failed, using original query. Reason: {hypothetical_answer}")
            return query
        print(f"HyDE Response: {hypothetical_answer[:100]}...")
        return hypothetical_answer

    def retrieve(self, query: str, k: int = 20) -> list:
        """Performs the initial retrieval from available indexes."""
        results = []
        if self.has_code:
            code_query_vec = self.code_model.encode([query], show_progress_bar=False).astype('float32')
            _, indices = self.code_index.search(code_query_vec, k)
            if indices.size > 0:
                results.extend([self.code_chunks[i] for i in indices[0]])
        if self.has_docs:
            docs_query_vec = self.docs_model.encode([query], show_progress_bar=False).astype('float32')
            _, indices = self.docs_index.search(docs_query_vec, k)
            if indices.size > 0:
                results.extend([self.docs_chunks[i] for i in indices[0]])
        return results

    def rerank(self, query: str, chunks: list, top_k: int = 5) -> list:
        """Re-ranks retrieved chunks using a CrossEncoder for higher precision."""
        print(f"Reranking {len(chunks)} candidates...")
        if not chunks:
            return []
        pairs = [(query, chunk['text']) for chunk in chunks]
        scores = self.reranker.predict(pairs, show_progress_bar=False)
        scored_chunks = sorted(zip(scores, chunks), key=lambda x: x[0], reverse=True)
        return [chunk for score, chunk in scored_chunks[:top_k]]

    def generate_final_answer(self, context: list, question: str, llm_url: str, lang: str) -> str:
        """Generates the final answer, instructing the LLM to use a specific language."""
        print(f"Generating final answer in language: {lang}")
        system_prompt = f"""You are an expert technical assistant. Your task is to answer the user's question based on the provided context snippets.
1. Analyze the user's question and the provided context.
2. Synthesize a concise and clear answer.
3. For every piece of information you use, you MUST cite the source file, like `(source: path/to/file.py)`.
4. Format your entire response in clear and readable Markdown.
5. You MUST write your entire response in the following language: {lang}"""

        messages = [{"role": "system", "content": system_prompt}]
        for item in context:
            context_message = f"CONTEXT FROM: {item['source']}\n\n---\n\n{item['text']}"
            messages.append({"role": "user", "content": context_message})
        messages.append({"role": "user", "content": f"Based on the context provided, answer this question: {question}"})

        return self._call_llm(messages, llm_url, temperature=0.5, max_tokens=2048)

    def get_response(self, query: str, llm_url: str, log_callback: Callable, use_hyde: bool):
        """Orchestrates the RAG pipeline with language detection."""
        try:
            lang_code = detect(query)
            log_callback(f"Query language detected: {lang_code}")
        except LangDetectException:
            lang_code = "en"
            log_callback("Could not detect language, defaulting to 'en'.")

        transformed_query = query
        if use_hyde:
            log_callback("Step: Transforming query (HyDE)...")
            transformed_query = self.transform_query_hyde(query, llm_url)
        else:
            log_callback("Step: Skipping query transformation (HyDE).")

        log_callback("Step: Retrieving initial candidates...")
        candidate_chunks = self.retrieve(transformed_query, k=20)
        if not candidate_chunks:
            log_callback("No candidate chunks found in the knowledge base.")
            return "No candidate chunks found in the knowledge base."

        log_callback("Step: Re-ranking for precision...")
        reranked_chunks = self.rerank(query, candidate_chunks, top_k=7)
        if not reranked_chunks:
            log_callback("No sufficiently relevant chunks found after re-ranking.")
            return "No sufficiently relevant chunks found after re-ranking."

        log_callback("Step: Generating final answer...")
        final_answer = self.generate_final_answer(reranked_chunks, query, llm_url, lang_code)
        return final_answer