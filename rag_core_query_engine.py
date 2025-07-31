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

        self.docs_model_name = self.model_config.get("docs_model_path", 'intfloat/multilingual-e5-large-instruct')
        self.code_model_name = self.model_config.get("code_model_path", 'jinaai/jina-embeddings-v2-base-code')
        self.reranker_model_name = self.model_config.get("reranker_model_path", 'Qwen/Qwen3-Reranker-0.6B')

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
                print(f"Loading code model: {self.code_model_name}")
                model_instance = SentenceTransformer(self.code_model_name, device=self.device)
                self.code_model = model_instance
            elif model_type == 'docs' and not self.docs_model:
                print(f"Loading docs model: {self.docs_model_name}")
                model_instance = SentenceTransformer(self.docs_model_name, device=self.device)
                self.docs_model = model_instance
            elif model_type == 'reranker' and not self.reranker:
                print(f"Loading reranker model: {self.reranker_model_name}")
                model_instance = CrossEncoder(self.reranker_model_name, device=self.device)
                self.reranker = model_instance

            if model_instance and hasattr(model_instance, 'tokenizer') and model_instance.tokenizer.pad_token is None:
                print(f"Warning: No padding token found for {model_type} model. Setting to EOS token.")
                model_instance.tokenizer.pad_token = model_instance.tokenizer.eos_token

            if model_instance and self.fp16 and self.device == 'cuda':
                print(f"Converting {model_type} model to FP16.")
                model_instance.half()
        except Exception as e:
            print(f"Error loading model {model_type}: {e}")

    def _unload_model(self, model_type: str):
        """Unloads a model to free VRAM."""
        unloaded = False
        if model_type == 'code' and self.code_model:
            self.code_model = None
            unloaded = True
        elif model_type == 'docs' and self.docs_model:
            self.docs_model = None
            unloaded = True
        elif model_type == 'reranker' and self.reranker:
            self.reranker = None
            unloaded = True

        if unloaded:
            print(f"Unloaded {model_type} model.")
            if self.device == 'cuda':
                torch.cuda.empty_cache()

    def unload_query_models(self):
        """Unloads all models as a safeguard after a query session."""
        print("Safeguard: Unloading all models...")
        self._unload_model('code')
        self._unload_model('docs')
        self._unload_model('reranker')

    def _call_llm_stream(self, messages: List, llm_url: str, model_name: str) -> Iterator[str]:
        """A single, reusable method to call any OpenAI-compatible API with streaming using requests."""
        payload = {"model": model_name, "messages": messages, "stream": True, "temperature": 0.1, "top_p": 0.9}
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

        # --- NEW: Improved HyDE Prompt ---
        messages = [
            {"role": "system",
             "content": "You are a technical writing expert. Your task is to generate a detailed, self-contained document that comprehensively answers the following user question. This document will be used for a vector search. Begin directly with the content, without any preamble like 'Here is the answer'."},
            {"role": "user", "content": query}
        ]

        response_generator = self._call_llm_stream(messages, llm_url, model_name)
        hypothetical_answer = "".join([part for part in response_generator])
        if hypothetical_answer.startswith("Error"):
            print(f"HyDE failed: {hypothetical_answer}")
            return query
        print(f"HyDE Response: {hypothetical_answer[:100]}...")
        return hypothetical_answer

    def retrieve(self, query: str, k: int = 20) -> list:
        """Performs retrieval from ChromaDB collections and unloads models immediately."""
        all_results = []
        if self.has_code and self.code_collection:
            self._load_model('code')
            if self.code_model:
                query_prefix = "query: " if "e5-large-instruct" in self.code_model_name else ""
                query_vec = self.code_model.encode([f"{query_prefix}{query}"]).tolist()
                res = self.code_collection.query(query_embeddings=query_vec, n_results=k)
                for i, doc in enumerate(res['documents'][0]): all_results.append(
                    {'text': doc, 'source': res['metadatas'][0][i]['source']})
            self._unload_model('code')

        if self.has_docs and self.docs_collection:
            self._load_model('docs')
            if self.docs_model:
                query_prefix = "query: " if "e5-large-instruct" in self.docs_model_name else ""
                query_vec = self.docs_model.encode([f"{query_prefix}{query}"]).tolist()
                res = self.docs_collection.query(query_embeddings=query_vec, n_results=k)
                for i, doc in enumerate(res['documents'][0]): all_results.append(
                    {'text': doc, 'source': res['metadatas'][0][i]['source']})
            self._unload_model('docs')
        return all_results

    def rerank(self, query: str, chunks: list, top_k: int = 7) -> list:
        """Performs reranking and unloads the model immediately."""
        if not chunks: return []

        self._load_model('reranker')
        if not self.reranker:
            print("Warning: Reranker not loaded. Returning top_k chunks without reranking.")
            return chunks[:top_k]

        pairs = [(query, chunk['text']) for chunk in chunks]
        scores = self.reranker.predict(pairs, show_progress_bar=False)
        self._unload_model('reranker')

        scored_chunks = sorted(zip(scores, chunks), key=lambda x: x[0], reverse=True)
        return [chunk for score, chunk in scored_chunks[:top_k]]

    def generate_final_answer(self, context: list, history: list, llm_url: str, lang: str, model_name: str) -> Iterator[
        str]:
        """Generates a final answer as a stream, using an improved prompt structure."""

        # --- NEW: Improved System Prompt ---
        system_prompt = f"""You are an expert-level technical AI assistant.
Your persona is helpful, precise, and professional.
You will be given a conversation history and a set of context snippets extracted from a knowledge base.
Your task is to synthesize the information from the context to answer the user's latest question.
Follow these rules:
1. Base your answer **only** on the provided context snippets and conversation history. Do not use any outside knowledge.
2. If the context does not contain the answer, state that you cannot find the information in the provided documents.
3. Synthesize a concise, clear, and comprehensive answer in well-formatted Markdown.
4. You MUST write your entire response in the following language: **{lang}**
"""

        messages = [{"role": "system", "content": system_prompt}]

        # Add conversation history (excluding the last user message which is the current question)
        messages.extend(history[:-1])

        # --- NEW: Consolidate context into a single block ---
        context_str_parts = []
        for i, item in enumerate(context):
            context_str_parts.append(f"--- CONTEXT SNIPPET {i + 1} (from: {item['source']}) ---\n{item['text']}")

        consolidated_context = "\n\n".join(context_str_parts)

        final_user_prompt = f"""Here is the context I have retrieved:

<CONTEXT>
{consolidated_context}
</CONTEXT>

Based on the conversation history and the context provided above, please answer this question:

"{history[-1]['content']}"
"""

        messages.append({"role": "user", "content": final_user_prompt})

        yield from self._call_llm_stream(messages, llm_url, model_name)

    def verify_and_cite_answer(self, answer: str, context: List[Dict]) -> str:
        """Verifies and cites a completed answer and unloads the model immediately."""
        if not context: return answer

        self._load_model('docs')
        if not self.docs_model: return answer

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

        self._unload_model('docs')
        return cited_answer.strip()

    def retrieve_and_rerank_context(self, query: str, use_hyde: bool, use_reranker: bool, llm_url: str,
                                    model_name: str) -> List[Dict]:
        """A blocking method that performs retrieval and reranking and returns the context."""
        transformed_query = query
        if use_hyde:
            if not llm_url or not model_name: raise ValueError("LLM URL and model name are required for HyDE.")
            transformed_query = self.transform_query_hyde(query, llm_url, model_name)

        candidate_chunks = self.retrieve(transformed_query, k=20)
        if not candidate_chunks: return []

        if use_reranker:
            print("Reranking context...")
            return self.rerank(query, candidate_chunks, top_k=7)
        else:
            print("Skipping reranker.")
            return candidate_chunks[:7]

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
            yield "No sufficiently relevant chunks found in the knowledge base. Please try rephrasing your question."
            return
        yield from self.generate_final_answer(context, history, llm_url, lang_code, model_name)