import customtkinter as ctk
from tkinter import filedialog
import threading
import os
import glob
import json
from rag_core_query_engine import HybridRAGQueryEngine
from rag_core import run_indexing_logic
import logging

# --- Logging Configuration to suppress warnings ---
logging.basicConfig()
logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)


class RAGApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Local RAG Engine")
        self.geometry("800x650")
        ctk.set_appearance_mode("dark")

        self.engine = None
        self.stop_event = threading.Event()
        self.config_file = "config.json"
        self.model_config = self.load_config()

        self.backend_urls = {
            "LM Studio": "http://localhost:1234/v1/chat/completions",
            "Ollama": "http://localhost:11434/v1/chat/completions"
        }

        self.tab_view = ctk.CTkTabview(self, anchor="w")
        self.tab_view.pack(expand=True, fill="both", padx=10, pady=10)

        self.indexing_tab = self.tab_view.add("Indexing")
        self.querying_tab = self.tab_view.add("Querying")

        # Call methods to create widgets in each tab
        self.create_indexing_widgets()
        self.create_querying_widgets()

        self.settings_button = ctk.CTkButton(self, text="Model Settings", command=self.open_settings_window)
        self.settings_button.pack(side="bottom", pady=10)

        self.refresh_project_list()

    def create_indexing_widgets(self):
        """Creates all widgets for the Indexing tab."""
        # --- Directory Selection Frame ---
        dir_frame = ctk.CTkFrame(self.indexing_tab)
        dir_frame.pack(fill="x", padx=10, pady=10)

        ctk.CTkLabel(dir_frame, text="Project Folder:").pack(side="left", padx=(10, 5))
        self.repo_path_entry = ctk.CTkEntry(dir_frame)
        self.repo_path_entry.pack(side="left", fill="x", expand=True)
        ctk.CTkButton(dir_frame, text="Browse...", command=self.select_directory).pack(side="left", padx=(5, 10))

        # --- Project Name Frame ---
        name_frame = ctk.CTkFrame(self.indexing_tab)
        name_frame.pack(fill="x", padx=10, pady=(0, 10))
        ctk.CTkLabel(name_frame, text="Project Name:").pack(side="left", padx=(10, 5))
        self.project_name_entry = ctk.CTkEntry(name_frame)
        self.project_name_entry.pack(side="left", fill="x", expand=True, padx=(0, 10))

        # --- Options Frame (Checkboxes) ---
        options_frame = ctk.CTkFrame(self.indexing_tab)
        options_frame.pack(fill="x", padx=10, pady=10)
        self.index_code_var = ctk.BooleanVar(value=True)
        self.index_docs_var = ctk.BooleanVar(value=True)
        ctk.CTkCheckBox(options_frame, text="Index Code", variable=self.index_code_var).pack(side="left", padx=10)
        ctk.CTkCheckBox(options_frame, text="Index Documents", variable=self.index_docs_var).pack(side="left", padx=10)

        # --- Control Buttons Frame ---
        button_frame = ctk.CTkFrame(self.indexing_tab)
        button_frame.pack(fill="x", padx=10, pady=10)
        self.start_button = ctk.CTkButton(button_frame, text="Start Indexing", command=self.start_indexing_thread)
        self.start_button.pack(side="left", padx=10)
        self.stop_button = ctk.CTkButton(button_frame, text="Stop", command=self.stop_indexing, state="disabled")
        self.stop_button.pack(side="left", padx=10)

        # --- Log Window ---
        self.log_textbox = ctk.CTkTextbox(self.indexing_tab, state="disabled")
        self.log_textbox.pack(expand=True, fill="both", padx=10, pady=(0, 10))

    def create_querying_widgets(self):
        """Creates all widgets for the Querying tab."""
        # --- Project Selection Frame ---
        project_frame = ctk.CTkFrame(self.querying_tab)
        project_frame.pack(fill="x", padx=10, pady=10)
        ctk.CTkLabel(project_frame, text="Project:").pack(side="left", padx=(10, 5))
        self.project_var = ctk.StringVar(value="Select a project")
        self.project_menu = ctk.CTkOptionMenu(project_frame, variable=self.project_var, command=self.load_engine)
        self.project_menu.pack(side="left", fill="x", expand=True)
        ctk.CTkButton(project_frame, text="Refresh List", command=self.refresh_project_list).pack(side="left",
                                                                                                  padx=(5, 10))

        # --- Query Options Frame ---
        options_frame = ctk.CTkFrame(self.querying_tab)
        options_frame.pack(fill="x", padx=10, pady=(0, 10))
        ctk.CTkLabel(options_frame, text="Backend:").pack(side="left", padx=(10, 5))
        self.backend_var = ctk.StringVar(value="LM Studio")
        ctk.CTkOptionMenu(options_frame, variable=self.backend_var, values=list(self.backend_urls.keys())).pack(
            side="left", padx=(0, 20))
        self.hyde_var = ctk.BooleanVar(value=True)
        ctk.CTkCheckBox(options_frame, text="Use HyDE", variable=self.hyde_var).pack(side="left")

        # --- Question Input Frame ---
        question_frame = ctk.CTkFrame(self.querying_tab)
        question_frame.pack(fill="x", padx=10, pady=10)
        self.question_entry = ctk.CTkEntry(question_frame, placeholder_text="Ask a question about your project...")
        self.question_entry.pack(side="left", fill="x", expand=True)
        self.search_button = ctk.CTkButton(question_frame, text="Search", command=self.start_query_thread)
        self.search_button.pack(side="left", padx=(5, 10))

        # --- Answer Display ---
        self.answer_textbox = ctk.CTkTextbox(self.querying_tab)
        self.answer_textbox.pack(expand=True, fill="both", padx=10, pady=(0, 10))

    def select_directory(self):
        """Opens a dialog to select a directory and auto-fills the path and project name."""
        path = filedialog.askdirectory()
        if path:
            self.repo_path_entry.delete(0, "end")
            self.repo_path_entry.insert(0, path)
            project_name = os.path.basename(path)
            self.project_name_entry.delete(0, "end")
            self.project_name_entry.insert(0, project_name)

    def log(self, message):
        """Appends a message to the log textbox in a thread-safe way."""

        def _append():
            self.log_textbox.configure(state="normal")
            self.log_textbox.insert("end", message + "\n")
            self.log_textbox.configure(state="disabled")
            self.log_textbox.see("end")

        self.after(0, _append)

    def start_indexing_thread(self):
        """Starts the indexing process in a background thread."""
        repo_path = self.repo_path_entry.get()
        project_name = self.project_name_entry.get()
        if not repo_path or not project_name:
            self.log("ERROR: Please select a directory and provide a project name.")
            return

        self.log_textbox.configure(state="normal")
        self.log_textbox.delete("1.0", "end")
        self.log_textbox.configure(state="disabled")
        self.start_button.configure(state="disabled")
        self.stop_button.configure(state="normal")
        self.stop_event.clear()

        thread = threading.Thread(
            target=self.run_indexing,
            args=(repo_path, project_name, self.index_code_var.get(), self.index_docs_var.get(), self.stop_event)
        )
        thread.daemon = True
        thread.start()

    def stop_indexing(self):
        """Signals the indexing thread to stop."""
        self.stop_event.set()
        self.log("Stopping process...")
        self.stop_button.configure(state="disabled")

    def run_indexing(self, repo_path, project_name, index_code, index_docs, stop_event):
        """Runs the indexing logic and handles UI updates."""
        try:
            run_indexing_logic(repo_path, project_name, index_code, index_docs, self.log, stop_event, self.model_config)
        finally:
            self.after(0, lambda: self.start_button.configure(state="normal"))
            self.after(0, lambda: self.stop_button.configure(state="disabled"))
            self.after(100, self.refresh_project_list)

    def refresh_project_list(self):
        """Scans for .faiss files and updates the project dropdown menu."""
        projects = {os.path.basename(f).split('_')[0] for f in glob.glob("*_index.faiss")}
        project_list = sorted(list(projects)) if projects else ["No Indexes Found"]
        self.project_menu.configure(values=project_list)
        if not projects:
            self.project_var.set("No Indexes Found")
        elif self.project_var.get() not in project_list:
            self.project_var.set(project_list[0])
            self.load_engine(project_list[0])

    def load_engine(self, project_name):
        """Loads the query engine for the selected project."""
        if not project_name or project_name == "No Indexes Found":
            return
        self.log_to_answer(f"Loading engine for project: {project_name}...", clear=True)
        thread = threading.Thread(target=self._load_engine_worker, args=(project_name,))
        thread.daemon = True
        thread.start()

    def _load_engine_worker(self, project_name):
        """The actual engine loading, done in a thread."""
        try:
            self.engine = HybridRAGQueryEngine(project_name, self.model_config)
            self.log_to_answer("✅ Engine ready. Ask a question.", clear=True)
        except Exception as e:
            self.log_to_answer(f"Error loading engine: {e}", clear=True)

    def start_query_thread(self):
        """Starts the query process in a background thread."""
        question = self.question_entry.get()
        if not self.engine:
            self.log_to_answer("Please select a project and wait for the engine to load.", clear=True)
            return
        if not question:
            self.log_to_answer("Please enter a question.", clear=True)
            return

        backend_name = self.backend_var.get()
        llm_url = self.backend_urls.get(backend_name)
        use_hyde = self.hyde_var.get()

        self.log_to_answer(f"Thinking...\n", clear=True)
        self.search_button.configure(state="disabled")

        thread = threading.Thread(target=self.run_query, args=(question, llm_url, use_hyde))
        thread.daemon = True
        thread.start()

    def run_query(self, question, llm_url, use_hyde):
        """Runs the query and displays the response."""
        try:
            response = self.engine.get_response(question, llm_url, self.log_to_answer, use_hyde)
            self.log_to_answer(response, clear=True)
        except Exception as e:
            self.log_to_answer(f"An error occurred: {e}", clear=True)
        finally:
            self.after(0, lambda: self.search_button.configure(state="normal"))

    def log_to_answer(self, message, clear=False):
        """Appends a message to the answer textbox in a thread-safe way."""

        def _update():
            self.answer_textbox.configure(state="normal")
            if clear:
                self.answer_textbox.delete("1.0", "end")
            # Append message without extra newline if it already has one
            end_char = "" if message.endswith('\n') else "\n"
            self.answer_textbox.insert("end", message + end_char)
            self.answer_textbox.configure(state="disabled")
            self.answer_textbox.see("end")

        self.after(0, _update)

    def load_config(self):
        """
        Loads configuration with a clear hierarchy (1 -> 2 -> 3):
        1. Start with hardcoded online model names as a fallback.
        2. Overwrite with auto-detected local models in './models/' if they exist.
        3. Overwrite with user-defined paths from config.json, which has the highest priority.
        """
        config = {
            "code_model_path": 'jinaai/jina-embeddings-v2-base-code',
            "docs_model_path": 'BAAI/bge-m3',
            "reranker_model_path": 'BAAI/bge-reranker-large'
        }
        local_model_map = {
            "code_model_path": "./models/jinaai_jina-embeddings-v2-base-code",
            "docs_model_path": "./models/BAAI_bge-m3",
            "reranker_model_path": "./models/BAAI_bge-reranker-large"
        }
        print("Checking for local models in './models/' directory...")
        for key, path in local_model_map.items():
            if os.path.isdir(path):
                print(f"Auto-configuring with local model: {path}")
                config[key] = path
        if os.path.exists(self.config_file):
            print(f"Loading user overrides from {self.config_file}...")
            with open(self.config_file, 'r') as f:
                try:
                    user_config = json.load(f)
                    config.update(user_config)
                except json.JSONDecodeError:
                    print(f"Warning: {self.config_file} is corrupted. Using auto-detected or default values.")
        return config

    def save_config(self):
        with open(self.config_file, 'w') as f:
            json.dump(self.model_config, f, indent=2)
        if self.project_var.get() and self.project_var.get() != "No Indexes Found":
            self.load_engine(self.project_var.get())

    def open_settings_window(self):
        settings_win = ctk.CTkToplevel(self)
        settings_win.title("Model Settings")
        settings_win.geometry("600x250")
        settings_win.transient(self)

        def create_row(key, text):
            frame = ctk.CTkFrame(settings_win)
            frame.pack(fill="x", padx=10, pady=5)
            label = ctk.CTkLabel(frame, text=text, width=150, anchor="w")
            label.pack(side="left")
            entry = ctk.CTkEntry(frame, width=300)
            entry.insert(0, self.model_config.get(key, ""))

            def select_path():
                path = filedialog.askdirectory()
                if path:
                    entry.delete(0, "end")
                    entry.insert(0, path)

            button = ctk.CTkButton(frame, text="Select Folder...", width=100, command=select_path)
            button.pack(side="left", padx=5)
            return entry

        code_entry = create_row("code_model_path", "Code Model:")
        docs_entry = create_row("docs_model_path", "Documents Model:")
        reranker_entry = create_row("reranker_model_path", "Reranker Model:")

        def on_save():
            self.model_config["code_model_path"] = code_entry.get()
            self.model_config["docs_model_path"] = docs_entry.get()
            self.model_config["reranker_model_path"] = reranker_entry.get()
            self.save_config()
            settings_win.destroy()

        save_button = ctk.CTkButton(settings_win, text="Save and Reload", command=on_save)
        save_button.pack(pady=20)


if __name__ == "__main__":
    app = RAGApp()
    app.mainloop()