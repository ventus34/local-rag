import os
import pickle
import faiss
import numpy as np
import pypdf
import docx
import openpyxl
from sentence_transformers import SentenceTransformer
from typing import List, Callable
import threading
from tree_sitter import Parser, Language
import logging

# --- Suppress transformers library warnings ---
logging.basicConfig()
logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)

# --- Dynamically load tree-sitter languages ---
SUPPORTED_LANGUAGES = {}
try:
    import tree_sitter_python as tspython
    SUPPORTED_LANGUAGES['python'] = Language(tspython.language())
except ImportError:
    print("Warning: tree-sitter-python not installed. Python code parsing will be disabled.")
try:
    import tree_sitter_javascript as tsjavascript
    SUPPORTED_LANGUAGES['javascript'] = Language(tsjavascript.language())
except ImportError:
    print("Warning: tree-sitter-javascript not installed. JS code parsing will be disabled.")
try:
    import tree_sitter_java as tsjava
    SUPPORTED_LANGUAGES['java'] = Language(tsjava.language())
except ImportError:
    print("Warning: tree-sitter-java not installed. Java code parsing will be disabled.")
try:
    import tree_sitter_go as tsgo
    SUPPORTED_LANGUAGES['go'] = Language(tsgo.language())
except ImportError:
    print("Warning: tree-sitter-go not installed. Go code parsing will be disabled.")
try:
    import tree_sitter_cpp as tscpp
    SUPPORTED_LANGUAGES['cpp'] = Language(tscpp.language())
except ImportError:
    print("Warning: tree-sitter-cpp not installed. C++ code parsing will be disabled.")

# --- Windows-specific import for .doc file support ---
try:
    import win32com.client as win32
    win32_imported = True
except ImportError:
    win32_imported = False


def chunk_code_with_tree_sitter(file_path: str, content: str, log_callback: Callable) -> List[str]:
    """
    Chunks code using tree-sitter with language-specific queries to split by functions and classes.
    Falls back to a recursive character splitter if parsing fails.
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
            parser = Parser()
            parser.language = language
            tree = parser.parse(bytes(content, "utf8"))
            query = language.query(queries[lang_name])
            captures = query.captures(tree.root_node)

            if not captures:
                return recursive_character_text_splitter(content)

            chunks = []
            last_end = 0
            # Iterate through captured nodes (functions, classes) and create chunks
            for node, name in captures:
                start_byte, end_byte = node.start_byte, node.end_byte
                # Add content between major constructs as a separate chunk
                if start_byte > last_end and content[last_end:start_byte].strip():
                    chunks.append(content[last_end:start_byte])
                chunks.append(node.text.decode('utf8'))
                last_end = end_byte
            # Add any remaining content at the end of the file
            if last_end < len(content) and content[last_end:].strip():
                chunks.append(content[last_end:])
            return chunks
        except Exception as e:
            log_callback(f"Tree-sitter failed for {file_path}, falling back to recursive split. Error: {e}")
            return recursive_character_text_splitter(content)

    return recursive_character_text_splitter(content)


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


def get_files_from_repo(repo_path: str, extensions: List[str], excluded_dirs: set) -> List[str]:
    """Recursively finds all files with given extensions in a directory, ignoring excluded folders."""
    allowed_extensions = {f".{ext.lstrip('.')}" for ext in extensions}
    found_files = []
    for root, dirs, files in os.walk(repo_path):
        # Modify dirs in-place to prevent os.walk from traversing them
        dirs[:] = [d for d in dirs if d not in excluded_dirs]
        for file in files:
            if any(file.endswith(ext) for ext in allowed_extensions):
                found_files.append(os.path.join(root, file))
    return found_files


def extract_text_from_file(file_path: str, log_callback: Callable):
    """Extracts raw text content from various file types."""
    absolute_file_path = os.path.abspath(file_path)
    _, extension = os.path.splitext(absolute_file_path)
    extension = extension.lower()
    word_app = None # For .doc handling
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
            return "\n".join(str(cell.value) for sheet in workbook.worksheets for row in sheet.iter_rows() for cell in row if cell.value is not None)
        elif extension == '.doc':
            if not win32_imported:
                log_callback(f"Skipping .doc file due to missing 'pywin32' dependency: {file_path}")
                return ""
            # Handle .doc files on Windows by converting to txt via Word COM object
            word_app = win32.gencache.EnsureDispatch('Word.Application')
            doc = word_app.Documents.Open(absolute_file_path)
            temp_txt_path = os.path.join(os.path.dirname(absolute_file_path), "temp_doc_to_txt.txt")
            doc.SaveAs(temp_txt_path, FileFormat=7) # wdFormatText
            doc.Close(False)
            with open(temp_txt_path, 'r', encoding='utf-8', errors='ignore') as f:
                text = f.read()
            os.remove(temp_txt_path)
            word_app.Quit()
            return text
        elif extension in ['.md', '.txt']:
            with open(absolute_file_path, 'r', encoding='utf-8', errors='ignore') as f:
                return f.read()
    except Exception as e:
        if word_app: word_app.Quit()
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
        model_config: dict
):
    log_callback("Starting indexing process...")
    excluded_dirs = {'.git', '.idea', '.vscode', 'node_modules', 'target', 'build', 'dist', '__pycache__'}

    if index_code:
        log_callback("\n--- Indexing Code (Syntax-Aware) ---")
        model_name_or_path = model_config.get("code_model_path", 'jinaai/jina-embeddings-v2-base-code')
        extensions = [".go", ".js", ".py", ".ts", ".java", ".c", ".cpp", ".h", ".hpp", ".php", ".sql"]
        files = get_files_from_repo(repo_path, extensions, excluded_dirs)
        log_callback(f"Found {len(files)} code files.")

        all_chunks = []
        for file_path in files:
            if stop_event.is_set():
                log_callback("Indexing stopped by user.")
                return
            log_callback(f"Processing: {file_path}")
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                if content:
                    relative_path = os.path.relpath(file_path, repo_path)
                    chunks = chunk_code_with_tree_sitter(file_path, content, log_callback)
                    for i, chunk in enumerate(chunks):
                        all_chunks.append({"text": chunk, "source": relative_path, "chunk_id": i})
            except Exception as e:
                log_callback(f"ERROR reading file {file_path}: {e}")

        if all_chunks and not stop_event.is_set():
            log_callback(f"Encoding {len(all_chunks)} code chunks with '{os.path.basename(model_name_or_path)}'...")
            model = SentenceTransformer(model_name_or_path)
            vectors = model.encode([item['text'] for item in all_chunks], show_progress_bar=False)
            index = faiss.IndexFlatL2(vectors.shape[1])
            index.add(np.array(vectors).astype('float32'))

            faiss.write_index(index, f"{project_name}_code_index.faiss")
            with open(f"{project_name}_code_chunks.pkl", "wb") as f:
                pickle.dump(all_chunks, f)
            log_callback("✅ Code indexing complete.")
        elif not stop_event.is_set():
            log_callback("No code to index.")

    if stop_event.is_set(): return

    if index_docs:
        log_callback("\n--- Indexing Documents ---")
        model_name_or_path = model_config.get("docs_model_path", 'BAAI/bge-m3')
        extensions = [".md", ".txt", ".pdf", ".docx", ".xlsx", ".doc"]
        files = get_files_from_repo(repo_path, extensions, excluded_dirs)
        log_callback(f"Found {len(files)} document files.")

        all_chunks = []
        for file_path in files:
            if stop_event.is_set():
                log_callback("Indexing stopped by user.")
                return
            log_callback(f"Processing: {file_path}")
            content = extract_text_from_file(file_path, log_callback)
            if content:
                chunks = recursive_character_text_splitter(content)
                for i, chunk in enumerate(chunks):
                    all_chunks.append({"text": chunk, "source": file_path, "chunk_id": i})

        if all_chunks and not stop_event.is_set():
            log_callback(f"Encoding {len(all_chunks)} document chunks with '{os.path.basename(model_name_or_path)}'...")
            model = SentenceTransformer(model_name_or_path)
            vectors = model.encode([item['text'] for item in all_chunks], show_progress_bar=False)
            index = faiss.IndexFlatL2(vectors.shape[1])
            index.add(np.array(vectors).astype('float32'))

            faiss.write_index(index, f"{project_name}_docs_index.faiss")
            with open(f"{project_name}_docs_chunks.pkl", "wb") as f:
                pickle.dump(all_chunks, f)
            log_callback("✅ Document indexing complete.")
        elif not stop_event.is_set():
            log_callback("No documents to index.")

    if not stop_event.is_set():
        log_callback("\n--- All Done! ---")