import os
import numpy as np
import pypdf
import docx
import openpyxl
from sentence_transformers import SentenceTransformer
from typing import List, Callable
import threading
from tree_sitter import Parser, Language
import logging
from pydocx import PyDocX
import html2text
import torch
from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface import HuggingFaceEmbeddings
import chromadb
from chromadb.config import Settings
import re
import hashlib

# --- Suppress transformers library warnings ---
logging.basicConfig()
logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)
logging.getLogger("chromadb").setLevel(logging.ERROR)

# --- Dynamically load tree-sitter languages ---

# --- Dynamically load tree-sitter languages ---
SUPPORTED_LANGUAGES = {}
try:
    from tree_sitter_python import language as python_language
    lang_obj = python_language() if callable(python_language) else python_language
    # Ensure we have a proper Language object
    if not isinstance(lang_obj, Language):
        if hasattr(lang_obj, '__class__') and 'PyCapsule' in str(type(lang_obj)):
            SUPPORTED_LANGUAGES['python'] = Language(lang_obj)
        else:
            print(f"Warning: Unexpected python language object type: {type(lang_obj)}")
    else:
        SUPPORTED_LANGUAGES['python'] = lang_obj
except (ImportError, Exception) as e:
    print(f"Warning: tree-sitter-python not available. {e}")

try:
    from tree_sitter_javascript import language as js_language
    lang_obj = js_language() if callable(js_language) else js_language
    if not isinstance(lang_obj, Language):
        if hasattr(lang_obj, '__class__') and 'PyCapsule' in str(type(lang_obj)):
            SUPPORTED_LANGUAGES['javascript'] = Language(lang_obj)
        else:
            print(f"Warning: Unexpected javascript language object type: {type(lang_obj)}")
    else:
        SUPPORTED_LANGUAGES['javascript'] = lang_obj
except (ImportError, Exception) as e:
    print(f"Warning: tree-sitter-javascript not available. {e}")

try:
    from tree_sitter_java import language as java_language
    lang_obj = java_language() if callable(java_language) else java_language
    if not isinstance(lang_obj, Language):
        if hasattr(lang_obj, '__class__') and 'PyCapsule' in str(type(lang_obj)):
            SUPPORTED_LANGUAGES['java'] = Language(lang_obj)
        else:
            print(f"Warning: Unexpected java language object type: {type(lang_obj)}")
    else:
        SUPPORTED_LANGUAGES['java'] = lang_obj
except (ImportError, Exception) as e:
    print(f"Warning: tree-sitter-java not available. {e}")

try:
    from tree_sitter_go import language as go_language
    lang_obj = go_language() if callable(go_language) else go_language
    if not isinstance(lang_obj, Language):
        if hasattr(lang_obj, '__class__') and 'PyCapsule' in str(type(lang_obj)):
            SUPPORTED_LANGUAGES['go'] = Language(lang_obj)
        else:
            print(f"Warning: Unexpected go language object type: {type(lang_obj)}")
    else:
        SUPPORTED_LANGUAGES['go'] = lang_obj
except (ImportError, Exception) as e:
    print(f"Warning: tree-sitter-go not available. {e}")

try:
    from tree_sitter_cpp import language as cpp_language
    lang_obj = cpp_language() if callable(cpp_language) else cpp_language
    if not isinstance(lang_obj, Language):
        if hasattr(lang_obj, '__class__') and 'PyCapsule' in str(type(lang_obj)):
            SUPPORTED_LANGUAGES['cpp'] = Language(lang_obj)
        else:
            print(f"Warning: Unexpected cpp language object type: {type(lang_obj)}")
    else:
        SUPPORTED_LANGUAGES['cpp'] = lang_obj
except (ImportError, Exception) as e:
    print(f"Warning: tree-sitter-cpp not available. {e}")

try:
    from tree_sitter_typescript import typescript as ts_language
    lang_obj = ts_language() if callable(ts_language) else ts_language
    if not isinstance(lang_obj, Language):
        if hasattr(lang_obj, '__class__') and 'PyCapsule' in str(type(lang_obj)):
            SUPPORTED_LANGUAGES['typescript'] = Language(lang_obj)
        else:
            print(f"Warning: Unexpected typescript language object type: {type(lang_obj)}")
    else:
        SUPPORTED_LANGUAGES['typescript'] = lang_obj
except (ImportError, Exception) as e:
    print(f"Warning: tree-sitter-typescript not available. {e}")



def recursive_character_text_splitter(text: str, chunk_size: int = 768, chunk_overlap: int = 100) -> List[str]:
    """A fallback text splitter based on character count."""
    if not text: return []
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - chunk_overlap
    return chunks


def chunk_code_with_tree_sitter(file_path: str, content: str, log_callback: Callable, chunk_size: int = 768,
                                chunk_overlap: int = 100) -> List[str]:
    """
    Chunks code using tree-sitter to respect logical boundaries (functions, classes),
    then applies a recursive splitter within those boundaries if they are too large.
    """
    file_extension = os.path.splitext(file_path)[1]
    lang_map = {
        '.py': 'python', '.js': 'javascript', '.go': 'go',
        '.java': 'java', '.cpp': 'cpp', '.c': 'cpp', '.h': 'cpp', '.hpp': 'cpp', '.ts': 'typescript'
    }
    queries = {
        'python': "(function_definition) @item\n(class_definition) @item",
        'javascript': "(function_declaration) @item\n(class_declaration) @item\n(method_definition) @item\n(arrow_function) @item",
        'typescript': "(function_declaration) @item\n(class_declaration) @item\n(method_definition) @item\n(arrow_function) @item",
        'java': "(class_declaration) @item\n(method_declaration) @item\n(constructor_declaration) @item",
        'cpp': "(function_definition) @item\n(class_specifier) @item\n(struct_specifier) @item",
        'go': "(function_declaration) @item\n(method_declaration) @item\n(type_declaration) @item"
    }
    lang_name = lang_map.get(file_extension)

    if lang_name and lang_name in SUPPORTED_LANGUAGES and lang_name in queries:
        try:
            language = SUPPORTED_LANGUAGES[lang_name]

            # Validate that we have a proper Language object
            if not isinstance(language, Language):
                log_callback(f"Invalid language object for {lang_name}, falling back to recursive split.")
                return recursive_character_text_splitter(content, chunk_size, chunk_overlap)

            parser = Parser()

            # Try both set_language and direct assignment for compatibility
            try:
                parser.set_language(language)
            except (AttributeError, TypeError):
                try:
                    parser.language = language
                except Exception:
                    log_callback(f"Could not set language for parser with {file_path}, using recursive split.")
                    return recursive_character_text_splitter(content, chunk_size, chunk_overlap)

            tree = parser.parse(bytes(content, "utf8"))
            query = language.query(queries[lang_name])

            # Handle different tree-sitter API versions
            captures = []
            try:
                # Try the current API - query.captures()
                captures_result = query.captures(tree.root_node)
                # Handle both tuple and object formats
                for item in captures_result:
                    if isinstance(item, tuple):
                        captures.append(item)
                    elif hasattr(item, 'node') and hasattr(item, 'name'):
                        captures.append((item.node, item.name))
                    else:
                        # Try to extract from match objects
                        if hasattr(item, 'captures'):
                            for capture in item.captures:
                                captures.append((capture.node, capture.name))
            except AttributeError:
                try:
                    # Try the matches API
                    matches = query.matches(tree.root_node)
                    for match in matches:
                        if hasattr(match, 'captures'):
                            for capture in match.captures:
                                captures.append((capture.node, capture.name))
                        elif isinstance(match, dict) and 'captures' in match:
                            for capture in match['captures']:
                                captures.append((capture['node'], capture['name']))
                except (AttributeError, TypeError):
                    # If all APIs fail, fall back to recursive splitting
                    log_callback(f"Tree-sitter API incompatible for {file_path}, using recursive split.")
                    return recursive_character_text_splitter(content, chunk_size, chunk_overlap)

            if not captures:
                log_callback(f"Tree-sitter found no major constructs in {file_path}, using recursive split.")
                return recursive_character_text_splitter(content, chunk_size, chunk_overlap)

            chunks = []
            last_end = 0

            if captures:
                first_node_start = captures[0][0].start_byte
                if first_node_start > 0:
                    chunks.extend(
                        recursive_character_text_splitter(content[0:first_node_start], chunk_size, chunk_overlap))

            for node, _ in captures:
                node_content = node.text.decode('utf8')

                if len(node_content) > chunk_size:
                    chunks.extend(recursive_character_text_splitter(node_content, chunk_size, chunk_overlap))
                else:
                    chunks.append(node_content)

                last_end = node.end_byte

            if last_end < len(content):
                chunks.extend(recursive_character_text_splitter(content[last_end:], chunk_size, chunk_overlap))

            return [chunk for chunk in chunks if chunk.strip()]
        except Exception as e:
            log_callback(f"Tree-sitter failed for {file_path}, falling back to recursive split. Error: {e}")
            return recursive_character_text_splitter(content, chunk_size, chunk_overlap)

    return recursive_character_text_splitter(content, chunk_size, chunk_overlap)


def get_files_from_repo(repo_path: str, extensions: List[str], excluded_dirs: set) -> List[str]:
    """
    Recursively finds all files with given extensions in a directory,
    ignoring excluded folders and minified/generated asset files.
    """
    allowed_extensions = {f".{ext.lstrip('.')}" for ext in extensions}
    found_files = []

    ignore_pattern = re.compile(r'(\.min\.(js|css)|-[a-f0-9]{8,}\.(js|css))$')

    for root, dirs, files in os.walk(repo_path):
        dirs[:] = [d for d in dirs if d not in excluded_dirs]
        for file in files:
            if ignore_pattern.search(file):
                continue

            if any(file.endswith(ext) for ext in allowed_extensions):
                found_files.append(os.path.join(root, file))
    return found_files


def extract_text_from_file(file_path: str, log_callback: Callable):
    """Extracts raw text content from various file types."""
    absolute_file_path = os.path.abspath(file_path)
    _, extension = os.path.splitext(absolute_file_path)
    extension = extension.lower()

    plain_text_extensions = [
        '.md', '.txt', '.py', '.js', '.java', '.go', '.cpp', '.c', '.h', '.hpp',
        '.ts', '.sql', '.css', '.html', '.json', '.xml', '.yaml', '.toml', '.php'
    ]

    try:
        if extension == '.pdf':
            with open(absolute_file_path, 'rb') as f:
                reader = pypdf.PdfReader(f)
                return "\n".join(page.extract_text() for page in reader.pages)
        elif extension == '.docx':
            doc = docx.Document(absolute_file_path)
            return "\n".join(para.text for para in doc.paragraphs)
        elif extension == '.xlsx':
            workbook = openpyxl.load_workbook(absolute_file_path)
            return "\n".join(
                str(cell.value) for sheet in workbook.worksheets for row in sheet.iter_rows() for cell in row if
                cell.value is not None)
        elif extension == '.doc':
            try:
                html = PyDocX.to_html(absolute_file_path)
                h = html2text.HTML2Text()
                h.ignore_links = True
                h.ignore_images = True
                return h.handle(html)
            except Exception as e:
                log_callback(f"Skipping .doc file due to pydocx processing error: {file_path}. Error: {e}")
                return ""
        elif extension in plain_text_extensions:
            with open(absolute_file_path, 'r', encoding='utf-8', errors='ignore') as f:
                return f.read()
    except Exception as e:
        log_callback(f"Warning: Could not process file {file_path}. Error: {e}")
        return ""
    return ""


def run_indexing_logic(
        repo_path: str,
        project_name: str,
        index_code: bool,
        index_docs: bool,
        log_callback: Callable,
        stop_event: threading.Event,
        user_config: dict,
        progress_callback: Callable,
        chroma_client: chromadb.Client
):
    log_callback("Starting indexing process...")
    model_config = user_config.get("models", {})
    hw_config = user_config.get("hardware", {})
    user_device_choice = hw_config.get("device", "Auto")

    if user_device_choice == "CPU":
        device = "cpu"
    elif user_device_choice == "GPU (CUDA)":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    log_callback(f"Using device: {device}")

    excluded_dirs = {
        '.git', '.idea', '.vscode', '__pycache__', 'coverage', '.cache',
        'node_modules', 'vendor', 'target', 'build', 'dist',
        'venv', '.venv', 'env', '.env',
        'assets', 'static', 'public', 'resources', 'cdn'
    }

    def process_files_in_batches(files: List[str], resource_type: str, model_name: str, chunking_function: Callable):
        """Helper function to process a list of files in memory-safe batches."""
        file_batch_size = 50
        total_files = len(files)

        collection_name = f"{project_name}-{resource_type}"
        collection = chroma_client.get_or_create_collection(name=collection_name)

        for i in range(0, total_files, file_batch_size):
            if stop_event.is_set(): log_callback("Indexing stopped by user."); return

            file_batch = files[i:i + file_batch_size]
            batch_label = f"files {i + 1}-{min(i + file_batch_size, total_files)} of {total_files}"
            log_callback(f"Processing {resource_type} {batch_label}...")

            batch_chunks_raw = []
            for file_path in file_batch:
                content = extract_text_from_file(file_path, log_callback)
                if content:
                    relative_path = os.path.relpath(file_path, repo_path)
                    chunks = chunking_function(file_path, content, log_callback)
                    for chunk in chunks:
                        batch_chunks_raw.append({"text": chunk, "source": relative_path})

            if not batch_chunks_raw:
                progress_callback(min(i + file_batch_size, total_files) / total_files)
                continue

            # --- Deduplication logic ---
            unique_chunks_map = {}
            for chunk in batch_chunks_raw:
                text_to_hash = chunk['source'] + chunk['text']
                id_hash = hashlib.sha256(text_to_hash.encode()).hexdigest()
                if id_hash not in unique_chunks_map:
                    unique_chunks_map[id_hash] = chunk

            if not unique_chunks_map:
                progress_callback(min(i + file_batch_size, total_files) / total_files)
                continue

            unique_chunks_list = list(unique_chunks_map.values())
            ids = [f"{collection_name}-{id_hash}" for id_hash in unique_chunks_map.keys()]
            documents_raw = [chunk['text'] for chunk in unique_chunks_list]

            # --- Add prefix for "instruct" models ---
            document_prefix = "passage: " if "e5-large-instruct" in model_name else ""
            documents_to_embed = [f"{document_prefix}{doc}" for doc in documents_raw]

            metadatas = [{"source": chunk['source']} for chunk in unique_chunks_list]

            log_callback(f"Encoding {len(documents_to_embed)} unique chunks for {batch_label}...")
            model = SentenceTransformer(model_name, device=device)

            if model.tokenizer.pad_token is None:
                model.tokenizer.pad_token = model.tokenizer.eos_token

            vectors = model.encode(
                documents_to_embed,
                show_progress_bar=True,
                batch_size=8
            )

            collection.add(
                embeddings=vectors.tolist(),
                documents=documents_raw,  # Save original documents without prefix
                metadatas=metadatas,
                ids=ids
            )

            del model, vectors, batch_chunks_raw, unique_chunks_map
            if device == 'cuda': torch.cuda.empty_cache()

            progress_callback(min(i + file_batch_size, total_files) / total_files)

    if index_code:
        log_callback("\n--- Indexing Code (Syntax-Aware) ---")
        model_name = model_config.get("code_model_path")
        extensions = [".go", ".js", ".py", ".ts", ".java", ".c", ".cpp", ".h", ".hpp", ".php", ".sql"]
        files_to_process = get_files_from_repo(repo_path, extensions, excluded_dirs)
        log_callback(f"Found {len(files_to_process)} code files to process.")
        process_files_in_batches(files_to_process, "code", model_name, chunk_code_with_tree_sitter)

    if stop_event.is_set(): return

    if index_docs:
        log_callback("\n--- Indexing Documents (Semantic) ---")
        model_name = model_config.get("docs_model_path")
        extensions = [".md", ".txt", ".pdf", ".docx", ".xlsx", ".doc"]
        files_to_process = get_files_from_repo(repo_path, extensions, excluded_dirs)
        log_callback(f"Found {len(files_to_process)} document files to process.")

        embeddings_model = HuggingFaceEmbeddings(model_name=model_name, model_kwargs={'device': device})
        semantic_splitter = SemanticChunker(embeddings_model)

        def semantic_chunking_wrapper(file_path, content, log_callback):
            return semantic_splitter.split_text(content)

        process_files_in_batches(files_to_process, "docs", model_name, semantic_chunking_wrapper)

        del embeddings_model, semantic_splitter
        if device == 'cuda': torch.cuda.empty_cache()

    if not stop_event.is_set():
        log_callback("\n--- All Done! ---")