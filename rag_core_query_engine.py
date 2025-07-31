import os
import requests
from sentence_transformers import SentenceTransformer, CrossEncoder
from typing import List, Iterator, Dict
import json
from langdetect import detect, LangDetectException
import logging
import torch
import re
import chromadb
from chromadb.config import Settings

# --- Logging Configuration ---
logging.basicConfig()
logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)
logging.getLogger("chromadb").setLevel(logging.ERROR)


class HybridRAGQueryEngine:
    def __init__(self, repo_path: str, user_config: dict, chroma_client: chromadb.Client):
        """
        Initializes the hybrid engine with on-demand model loading.
        """
        print(f"Initializing RAG Engine for project: {repo_path}...")

        self.repo_path = repo_path
        self.model_config = user_config.get("models", {})
        hw_config = user_config.get("hardware", {})
        self.backend_urls = user_config.get("endpoints", {})
        project_name = os.path.basename(repo_path)

        self.fp16 = hw_config.get("fp16", False)
        user_device_choice = hw_config.get("device", "Auto")

        if user_device_choice == "CPU":
            self.device = "cpu"
        elif user_device_choice == "GPU (CUDA)":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}. FP16: {self.fp16}")

        self.docs_model_name = self.model_config.get("docs_model_path", 'BAAI/bge-m3')
        self.code_model_name = self.model_config.get("code_model_path", 'jinaai/jina-embeddings-v2-base-code')
        self.reranker_model_name = self.model_config.get("reranker_model_path", 'BAAI/bge-reranker-large')

        self.code_model, self.docs_model, self.reranker = None, None, None

        self.chroma_client = chroma_client
        try:
            self.code_collection = self.chroma_client.get_collection(name=f"{project_name}-code")
            self.has_code = True
        except Exception:
            self.code_collection = None;
            self.has_code = False
            print("⚠️ Code collection not found in ChromaDB.")
        try:
            self.docs_collection = self.chroma_client.get_collection(name=f"{project_name}-docs")
            self.has_docs = True
        except Exception:
            self.docs_collection = None;
            self.has_docs = False
            print("⚠️ Docs collection not found in ChromaDB.")

        if not self.has_code and not self.has_docs:
            raise FileNotFoundError(f"No valid collection found for project '{project_name}'. Please re-index it.")

        print("✅ Engine ready for on-demand model loading.")

    def _load_model(self, model_type: str):
        """Loads a specific model into memory and applies FP16 if configured."""
        try:
            model_instance = None
            if model_type == 'code' and not self.code_model:
                model_instance = SentenceTransformer(self.code_model_name, device=self.device)
                self.code_model = model_instance
            elif model_type == 'docs' and not self.docs_model:
                model_instance = SentenceTransformer(self.docs_model_name, device=self.device)
                self.docs_model = model_instance
            elif model_type == 'reranker' and not self.reranker:
                model_instance = CrossEncoder(self.reranker_model_name, device=self.device)
                self.reranker = model_instance

            if model_instance and self.fp16 and self.device == 'cuda':
                print(f"Converting {model_type} model to FP16.")
                model_instance.half()
        except Exception as e:
            print(f"Error loading model {model_type}: {e}")

    def _unload_model(self, model_type: str):
        """Unloads a model to free VRAM."""
        if model_type == 'code':
            self.code_model = None
        elif model_type == 'docs':
            self.docs_model = None
        elif model_type == 'reranker':
            self.reranker = None
        if self.device == 'cuda':
            torch.cuda.empty_cache()

    def load_query_models(self):
        """Loads all models required for a query session."""
        print("Loading models for query...")
        if self.has_code: self._load_model('code')
        if self.has_docs: self._load_model('docs')
        self._load_model('reranker')

    def unload_query_models(self):
        """Unloads all models after a query session."""
        print("Unloading models...")
        if self.has_code: self._unload_model('code')
        if self.has_docs: self._unload_model('docs')
        self._unload_model('reranker')

    def _call_llm_stream(self, messages: List, llm_url: str, model_name: str) -> Iterator[str]:
        """A single, reusable method to call any OpenAI-compatible API with streaming using requests."""
        payload = {"model": model_name, "messages": messages, "stream": True, "temperature": 0.7}
        try:
            with requests.post(llm_url, json=payload, stream=True, timeout=120) as response:
                response.raise_for_status()
                for line in response.iter_lines():
                    if line:
                        line_str = line.decode('utf-8')
                        if line_str.startswith("data: "): line_str = line_str[len("data: "):]
                        if line_str.strip() == "[DONE]": break
                        try:
                            chunk = json.loads(line_str)
                            if "choices" in chunk and chunk["choices"][0]["delta"].get("content"):
                                yield chunk["choices"][0]["delta"]["content"]
                        except json.JSONDecodeError:
                            continue
        except requests.exceptions.RequestException as e:
            yield f"Error connecting to LLM at {llm_url}: {e}"
        except Exception as e:
            yield f"An unexpected error occurred: {e}"

    def transform_query_hyde(self, query: str, llm_url: str, model_name: str) -> str:
        """Generates a hypothetical document to improve search relevance (HyDE)."""
        print("Transforming query with HyDE...")
        messages = [{"role": "system",
                     "content": "Write a short code snippet or technical explanation that is a perfect answer to the user question."},
                    {"role": "user", "content": query}]
        response_generator = self._call_llm_stream(messages, llm_url, model_name)
        hypothetical_answer = "".join([part for part in response_generator])
        if hypothetical_answer.startswith("Error"): print(f"HyDE failed: {hypothetical_answer}"); return query
        print(f"HyDE Response: {hypothetical_answer[:100]}...")
        return hypothetical_answer

    def retrieve(self, query: str, k: int = 20) -> list:
        """Performs retrieval from ChromaDB collections using pre-loaded models."""
        all_results = []
        if self.has_code and self.code_model and self.code_collection:
            query_vec = self.code_model.encode([query]).tolist()
            res = self.code_collection.query(query_embeddings=query_vec, n_results=k)
            for i, doc in enumerate(res['documents'][0]): all_results.append(
                {'text': doc, 'source': res['metadatas'][0][i]['source']})
        if self.has_docs and self.docs_model and self.docs_collection:
            query_vec = self.docs_model.encode([query]).tolist()
            res = self.docs_collection.query(query_embeddings=query_vec, n_results=k)
            for i, doc in enumerate(res['documents'][0]): all_results.append(
                {'text': doc, 'source': res['metadatas'][0][i]['source']})
        return all_results

    def rerank(self, query: str, chunks: list, top_k: int = 7) -> list:
        """Performs reranking using pre-loaded models."""
        if not chunks: return []
        if not self.reranker: print("Warning: Reranker not loaded."); return chunks[:top_k]
        pairs = [(query, chunk['text']) for chunk in chunks]
        scores = self.reranker.predict(pairs, show_progress_bar=False)
        scored_chunks = sorted(zip(scores, chunks), key=lambda x: x[0], reverse=True)
        return [chunk for score, chunk in scored_chunks[:top_k]]

    def generate_final_answer(self, context: list, history: list, llm_url: str, lang: str, model_name: str) -> Iterator[
        str]:
        """Generates a final answer as a stream, without citation."""
        system_prompt = f"You are an expert technical assistant. Answer the user's question based on the provided context snippets and conversation history. Synthesize a concise, clear answer in readable Markdown. DO NOT add sources like (source: ...). You MUST write your entire response in the following language: {lang}"
        messages = [{"role": "system", "content": system_prompt}]
        messages.extend(history[:-1])
        for item in context: messages.append(
            {"role": "user", "content": f"CONTEXT FROM: {item['source']}\n\n---\n\n{item['text']}"})
        messages.append(history[-1])
        yield from self._call_llm_stream(messages, llm_url, model_name)

    def verify_and_cite_answer(self, answer: str, context: List[Dict]) -> str:
        """Verifies and cites a completed answer using pre-loaded models."""
        if not context or not self.docs_model: return answer
        sentences = re.split(r'(?<=[.!?])\s+', answer)
        cited_answer = ""
        source_map = {item['text']: item['source'] for item in context}
        context_texts = list(source_map.keys())
        context_embeddings = self.docs_model.encode(context_texts, convert_to_tensor=True, device=self.device)
        for sentence in sentences:
            if not sentence.strip(): continue
            sentence_embedding = self.docs_model.encode(sentence, convert_to_tensor=True, device=self.device)
            similarities = torch.nn.functional.cosine_similarity(sentence_embedding, context_embeddings)
            best_match_idx = torch.argmax(similarities).item()
            if similarities[best_match_idx] > 0.65:
                source_file = source_map[context_texts[best_match_idx]]
                sentence += f""
            cited_answer += sentence + " "
        return cited_answer.strip()

    def retrieve_and_rerank_context(self, query: str, use_hyde: bool, llm_url: str, model_name: str) -> List[Dict]:
        """A blocking method that performs retrieval and reranking and returns the context."""
        transformed_query = query
        if use_hyde:
            if not llm_url or not model_name: raise ValueError("LLM URL and model name are required for HyDE.")
            transformed_query = self.transform_query_hyde(query, llm_url, model_name)
        candidate_chunks = self.retrieve(transformed_query, k=20)
        if not candidate_chunks: return []
        return self.rerank(query, candidate_chunks, top_k=7)

    def stream_answer_with_context(self, history: list, context: list, llm_url: str, lang_choice: str,
                                   model_name: str) -> Iterator[str]:
        """Streams a final answer given a pre-computed context."""
        query = history[-1]['content']
        lang_map = {"Polish": "pl", "English": "en"}
        if lang_choice in lang_map:
            lang_code = lang_map[lang_choice]
        else:
            try:
                lang_code = detect(query)
            except LangDetectException:
                lang_code = "en"
        if not context:
            yield "No sufficiently relevant chunks found in the knowledge base."
            return
        yield from self.generate_final_answer(context, history, llm_url, lang_code, model_name)