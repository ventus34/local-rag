import customtkinter as ctk
from tkinter import filedialog
import threading
import os
import json
import chromadb
from chromadb.config import Settings
from rag_core_query_engine import HybridRAGQueryEngine
from rag_core import run_indexing_logic
import logging

# --- Suppress transformers library warnings ---
logging.basicConfig()
logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)


class RAGApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Local RAG Engine")
        self.geometry("1100x700")
        ctk.set_appearance_mode("dark")

        # --- Instance Variables ---
        self.engine = None
        self.stop_event = threading.Event()
        self.config_file = "config.json"
        self.user_config = self.load_config()
        self.model_config = self.user_config.get("models", {})
        self.conversation_history = []
        self.chat_widgets = []

        # --- Predefined Model Choices for Settings UI ---
        self.model_choices = {
            "code": {
                "jinaai/jina-embeddings-v2-base-code": "default, balanced",
                "BAAI/bge-small-en-v1.5": "small, fast (EN)",
                "intfloat/e5-large-v2": "large, accurate (EN)"
            },
            "docs": {
                "intfloat/multilingual-e5-large-instruct": "default, multilingual, instruct",
                "BAAI/bge-m3": "legacy, multilingual, large",
                "sentence-transformers/all-MiniLM-L6-v2": "small, very fast (EN)",
            },
            "reranker": {
                "Qwen/Qwen3-Reranker-0.6B": "default, fast, multilingual (0.6B)",
            }
        }
        self.update_backend_urls_from_config()

        # --- ChromaDB Client ---
        self.chroma_client = chromadb.PersistentClient(
            path="./db",
            settings=Settings(anonymized_telemetry=False)
        )

        # --- UI Setup ---
        self.tab_view = ctk.CTkTabview(self, anchor="w")
        self.tab_view.pack(expand=True, fill="both", padx=10, pady=10)

        self.indexing_tab = self.tab_view.add("Indexing")
        self.querying_tab = self.tab_view.add("Querying")

        self.create_indexing_widgets()
        self.create_querying_widgets()

        self.settings_button = ctk.CTkButton(self, text="Settings", command=self.open_settings_window)
        self.settings_button.pack(side="bottom", pady=10)

        self.refresh_project_list()

    def create_indexing_widgets(self):
        """Creates all widgets for the Indexing tab."""
        dir_frame = ctk.CTkFrame(self.indexing_tab)
        dir_frame.pack(fill="x", padx=10, pady=10)
        ctk.CTkLabel(dir_frame, text="Project Folder:").pack(side="left", padx=(10, 5))
        self.repo_path_entry = ctk.CTkEntry(dir_frame, width=400)
        self.repo_path_entry.pack(side="left", fill="x", expand=True)
        ctk.CTkButton(dir_frame, text="Browse...", command=self.select_directory).pack(side="left", padx=(5, 10))

        name_frame = ctk.CTkFrame(self.indexing_tab)
        name_frame.pack(fill="x", padx=10, pady=(0, 10))
        ctk.CTkLabel(name_frame, text="Project Name:").pack(side="left", padx=(10, 5))
        self.project_name_entry = ctk.CTkEntry(name_frame)
        self.project_name_entry.pack(side="left", fill="x", expand=True, padx=(0, 10))

        options_frame = ctk.CTkFrame(self.indexing_tab)
        options_frame.pack(fill="x", padx=10, pady=10)
        self.index_code_var = ctk.BooleanVar(value=True)
        self.index_docs_var = ctk.BooleanVar(value=True)
        ctk.CTkCheckBox(options_frame, text="Index Code", variable=self.index_code_var).pack(side="left", padx=10)
        ctk.CTkCheckBox(options_frame, text="Index Documents", variable=self.index_docs_var).pack(side="left", padx=10)

        button_frame = ctk.CTkFrame(self.indexing_tab)
        button_frame.pack(fill="x", padx=10, pady=10)
        self.start_button = ctk.CTkButton(button_frame, text="Start Indexing", command=self.start_indexing_thread)
        self.start_button.pack(side="left", padx=10)
        self.stop_button = ctk.CTkButton(button_frame, text="Stop", command=self.stop_indexing, state="disabled")
        self.stop_button.pack(side="left", padx=10)

        self.progress_bar = ctk.CTkProgressBar(self.indexing_tab)
        self.progress_bar.set(0)
        self.progress_bar.pack(fill="x", padx=10, pady=(5, 10))

        self.log_textbox = ctk.CTkTextbox(self.indexing_tab, state="disabled")
        self.log_textbox.pack(expand=True, fill="both", padx=10, pady=(0, 10))

    def create_querying_widgets(self):
        """Creates all widgets for the Querying tab."""
        top_frame = ctk.CTkFrame(self.querying_tab)
        top_frame.pack(fill="x", padx=10, pady=10)

        project_frame = ctk.CTkFrame(top_frame)
        project_frame.pack(side="left", fill="x", expand=True)
        ctk.CTkLabel(project_frame, text="Project:").pack(side="left", padx=(10, 5))
        self.project_var = ctk.StringVar(value="Select a project")
        self.project_menu = ctk.CTkOptionMenu(project_frame, variable=self.project_var, command=self.load_engine)
        self.project_menu.pack(side="left", fill="x", expand=True)
        ctk.CTkButton(project_frame, text="Refresh", command=self.refresh_project_list).pack(side="left", padx=(5, 10))

        options_frame = ctk.CTkFrame(self.querying_tab)
        options_frame.pack(fill="x", padx=10, pady=(0, 10))
        ctk.CTkLabel(options_frame, text="Backend:").pack(side="left", padx=(10, 5))
        self.backend_var = ctk.StringVar(value="LM Studio")
        ctk.CTkOptionMenu(options_frame, variable=self.backend_var, values=list(self.backend_urls.keys())).pack(
            side="left")
        ctk.CTkLabel(options_frame, text="Model:").pack(side="left", padx=(10, 5))
        self.model_name_entry = ctk.CTkEntry(options_frame, placeholder_text="e.g., llama3", width=120)
        self.model_name_entry.insert(0, self.user_config.get("default_model", "local-model"))
        self.model_name_entry.pack(side="left")
        ctk.CTkLabel(options_frame, text="Language:").pack(side="left", padx=(10, 5))
        self.lang_var = ctk.StringVar(value="Autodetect")
        ctk.CTkOptionMenu(options_frame, variable=self.lang_var, values=["Autodetect", "Polish", "English"]).pack(
            side="left")
        self.hyde_var = ctk.BooleanVar(value=True)
        ctk.CTkCheckBox(options_frame, text="Use HyDE", variable=self.hyde_var).pack(side="left", padx=10)
        self.reranker_var = ctk.BooleanVar(value=True)
        ctk.CTkCheckBox(options_frame, text="Use Reranker", variable=self.reranker_var).pack(side="left", padx=10)

        self.chat_frame = ctk.CTkScrollableFrame(self.querying_tab)
        self.chat_frame.pack(expand=True, fill="both", padx=10, pady=10)

        question_frame = ctk.CTkFrame(self.querying_tab)
        question_frame.pack(fill="x", padx=10, pady=(0, 10))
        self.question_entry = ctk.CTkEntry(question_frame, placeholder_text="Ask a question...")
        self.question_entry.pack(side="left", fill="x", expand=True, padx=(10, 5), pady=10)
        self.question_entry.bind("<Return>", self.start_query_thread)
        self.send_button = ctk.CTkButton(question_frame, text="Send", command=self.start_query_thread, width=80)
        self.send_button.pack(side="right", padx=(5, 10), pady=10)

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

    def update_progress_bar(self, value):
        """Thread-safe method to update the progress bar."""
        self.progress_bar.set(value)

    def start_indexing_thread(self):
        """Starts the indexing process in a background thread."""
        repo_path = self.repo_path_entry.get()
        project_name = self.project_name_entry.get()
        if not repo_path or not project_name:
            self.log("ERROR: Please select a directory and provide a project name.")
            return

        self.update_progress_bar(0)
        self.log_textbox.configure(state="normal");
        self.log_textbox.delete("1.0", "end");
        self.log_textbox.configure(state="disabled")
        self.start_button.configure(state="disabled");
        self.stop_button.configure(state="normal")
        self.stop_event.clear()

        thread = threading.Thread(
            target=self.run_indexing,
            args=(repo_path, project_name, self.index_code_var.get(), self.index_docs_var.get(), self.stop_event,
                  self.update_progress_bar, self.chroma_client)
        )
        thread.daemon = True
        thread.start()

    def stop_indexing(self):
        """Signals the indexing thread to stop."""
        self.stop_event.set()
        self.log("Stopping process...")
        self.stop_button.configure(state="disabled")

    def run_indexing(self, repo_path, project_name, index_code, index_docs, stop_event, progress_callback,
                     chroma_client):
        """Runs the indexing logic and handles UI updates, including saving the project path."""
        try:
            run_indexing_logic(repo_path, project_name, index_code, index_docs, self.log, stop_event, self.user_config,
                               progress_callback, chroma_client)
            if not stop_event.is_set():
                self.log(f"Successfully indexed '{project_name}'. Saving path...")
                if "project_paths" not in self.user_config:
                    self.user_config["project_paths"] = {}
                self.user_config["project_paths"][project_name] = os.path.abspath(repo_path)
                self.save_config()
                self.log("Project path saved.")
        finally:
            self.after(0, lambda: self.start_button.configure(state="normal"))
            self.after(0, lambda: self.stop_button.configure(state="disabled"))
            self.after(100, self.refresh_project_list)
            self.after(0, self.update_progress_bar, 1.0)

    def refresh_project_list(self):
        """Scans the ChromaDB for collections and updates the project dropdown menu."""
        try:
            collections = self.chroma_client.list_collections()
            projects = {col.name.rsplit('-', 1)[0] for col in collections}
            project_list = sorted(list(projects)) if projects else ["No Indexes Found"]

            self.project_menu.configure(values=project_list)

            current_selection = self.project_var.get()

            if not projects:
                self.project_var.set("No Indexes Found")
            elif current_selection not in project_list or current_selection == "No Indexes Found":
                self.project_var.set(project_list[0])
                self.load_engine(project_list[0])

        except Exception as e:
            print(f"Could not refresh project list from ChromaDB: {e}")
            self.project_menu.configure(values=["No Indexes Found"])
            self.project_var.set("No Indexes Found")

    def load_engine(self, project_name):
        """Loads the query engine for the selected project and clears the chat."""
        if not project_name or project_name == "No Indexes Found": return

        self.conversation_history.clear()
        for widget_frame in self.chat_widgets: widget_frame.destroy()
        self.chat_widgets.clear()
        self.add_chat_message("assistant", f"Loading engine for project: '{project_name}'...", is_status=True)

        thread = threading.Thread(target=self._load_engine_worker, args=(project_name,))
        thread.daemon = True
        thread.start()

    def _load_engine_worker(self, project_name):
        """The actual engine loading, done in a thread."""
        try:
            repo_path = self.user_config.get("project_paths", {}).get(project_name)
            if not repo_path:
                raise FileNotFoundError(f"Path for project '{project_name}' not found. Please re-index it.")

            self.engine = HybridRAGQueryEngine(repo_path, self.user_config, self.chroma_client)
            self.add_chat_message("assistant", f"✅ Engine for '{project_name}' is ready. Ask a question.",
                                  is_status=True)
        except Exception as e:
            self.add_chat_message("assistant", f"Error loading engine: {e}", is_status=True)

    def start_query_thread(self, event=None):
        """Starts the query process in a background thread."""
        question = self.question_entry.get()
        if not self.engine or not question.strip(): return

        self.conversation_history.append({"role": "user", "content": question})
        self.add_chat_message("user", question)
        self.question_entry.delete(0, "end")
        self.add_chat_message("assistant", "Processing...", is_status=True)

        params = {
            "llm_url": self.backend_urls.get(self.backend_var.get()),
            "use_hyde": self.hyde_var.get(),
            "use_reranker": self.reranker_var.get(),
            "lang_choice": self.lang_var.get(),
            "model_name": self.model_name_entry.get()
        }
        self.send_button.configure(state="disabled")
        thread = threading.Thread(target=self.run_query, kwargs=params)
        thread.daemon = True
        thread.start()

    def run_query(self, **kwargs):
        """
        Manages the entire query lifecycle with JIT model loading.
        """
        full_response = ""
        assistant_message_widget = None
        try:
            question = self.conversation_history[-1]['content']
            context = self.engine.retrieve_and_rerank_context(
                question,
                kwargs['use_hyde'],
                kwargs['use_reranker'],
                kwargs['llm_url'],
                kwargs['model_name']
            )

            response_generator = self.engine.stream_answer_with_context(
                self.conversation_history,
                context,
                kwargs['llm_url'],
                kwargs['lang_choice'],
                kwargs['model_name']
            )
            assistant_message_widget = self.add_chat_message("assistant", "", return_widget=True)

            for chunk in response_generator:
                full_response += chunk
                self.after(5, self.update_chat_message_stream, assistant_message_widget, chunk)

            self.conversation_history.append({"role": "assistant", "content": full_response})

            if context:
                top_sources = list(dict.fromkeys([chunk['source'] for chunk in context[:5]]))
                self.run_citation(full_response, context, top_sources, assistant_message_widget)

        except Exception as e:
            self.after(0, lambda err=e: self.add_chat_message("assistant", f"An error occurred: {err}"))
        finally:
            if self.engine:
                self.engine.unload_query_models()
            self.after(0, lambda: self.send_button.configure(state="normal"))

    def run_citation(self, text, context, top_sources, widget):
        """Runs citation verification and appends sources in a thread, then schedules a UI update."""
        cited_text = self.engine.verify_and_cite_answer(text, context)

        if top_sources:
            sources_text = "\n\n---\n*Sources used:*\n" + "\n".join(f"- `{source}`" for source in top_sources)
            cited_text += sources_text

        self.after(0, self.update_chat_message_content, widget, cited_text)

    def add_chat_message(self, role: str, message: str, is_status: bool = False, return_widget: bool = False):
        """Adds a message or status widget to the chat frame."""
        if self.chat_widgets and self.chat_widgets[-1].winfo_children():
            last_widget_label = self.chat_widgets[-1].winfo_children()[0]
            if isinstance(last_widget_label, (ctk.CTkLabel, ctk.CTkTextbox)) and (
            "Processing..." in last_widget_label.cget("text") if isinstance(last_widget_label,
                                                                            ctk.CTkLabel) else "Processing..." in last_widget_label.get(
                    "1.0", "end")):
                self.chat_widgets.pop().destroy()

        justify, anchor, fg_color = ("left", "w", "#3A3A3A") if role == "assistant" else ("right", "e", "#2C4B8F")

        frame = ctk.CTkFrame(self.chat_frame, fg_color="transparent")

        # Use CTkTextbox instead of CTkLabel for better text handling
        msg_bubble = ctk.CTkTextbox(frame, height=1, wrap="word", fg_color=fg_color,
                                    corner_radius=10, font=("", 12))

        # Configure the textbox
        msg_bubble.insert("1.0", message)
        msg_bubble.configure(state="disabled")  # Make it read-only

        if is_status:
            msg_bubble.configure(fg_color="#4A4A4A", text_color="#B0B0B0")

        msg_bubble.pack(anchor=anchor, pady=2, padx=10, fill="x")
        frame.pack(fill="x", expand=True)
        self.chat_widgets.append(frame)

        # Auto-resize the textbox based on content
        self.after(10, lambda: self._resize_textbox(msg_bubble))
        self.after(100, self.chat_frame._parent_canvas.yview_moveto, 1.0)

        if return_widget:
            return msg_bubble

    def _resize_textbox(self, textbox):
        """Automatically resize textbox based on content."""
        try:
            textbox.update_idletasks()
            # Count lines and adjust height
            content = textbox.get("1.0", "end-1c")
            lines = content.count('\n') + 1
            # Estimate additional lines for word wrapping
            estimated_lines = max(lines, len(content) // 80 + 1)
            new_height = min(max(estimated_lines * 20, 30), 300)  # Min 30px, max 300px
            textbox.configure(height=new_height)
        except:
            pass

    def update_chat_message_stream(self, widget, chunk):
        """Appends a chunk of text to a message widget."""
        widget.configure(state="normal")
        widget.insert("end", chunk)
        widget.configure(state="disabled")
        # Resize after adding content
        self.after(10, lambda: self._resize_textbox(widget))

    def update_chat_message_content(self, widget, new_text):
        """Replaces the entire text of a message widget with proper error handling."""
        try:
            # Check if widget still exists and is valid
            if not widget or not widget.winfo_exists():
                print("Warning: Widget no longer exists, skipping update")
                return

            # Check if the widget is still in the widget hierarchy
            if not hasattr(widget, 'configure'):
                print("Warning: Widget is not configurable, skipping update")
                return

            widget.configure(state="normal")
            widget.delete("1.0", "end")
            widget.insert("1.0", new_text)
            widget.configure(state="disabled")

            # Use a safer approach for delayed resize
            if widget.winfo_exists():
                self.after(10, lambda w=widget: self._safe_resize_textbox(w))

        except Exception as e:
            print(f"Error updating chat message content: {e}")
            # Don't re-raise the exception to prevent crashes

    def update_chat_message_stream(self, widget, chunk):
        """Appends a chunk of text to a message widget with proper error handling."""
        try:
            # Check if widget still exists and is valid
            if not widget or not widget.winfo_exists():
                print("Warning: Widget no longer exists, skipping stream update")
                return

            widget.configure(state="normal")
            widget.insert("end", chunk)
            widget.configure(state="disabled")

            # Use a safer approach for delayed resize
            if widget.winfo_exists():
                self.after(10, lambda w=widget: self._safe_resize_textbox(w))

        except Exception as e:
            print(f"Error updating chat message stream: {e}")
            # Don't re-raise the exception to prevent crashes

    def _safe_resize_textbox(self, widget):
        """Safely resize textbox with existence check."""
        try:
            if widget and widget.winfo_exists():
                self._resize_textbox(widget)
        except Exception as e:
            print(f"Error during safe resize: {e}")

    def load_config(self):
        """Loads configuration from config.json, with defaults."""
        defaults = {
            "models": {"code_model_path": 'jinaai/jina-embeddings-v2-base-code',
                       "docs_model_path": 'intfloat/multilingual-e5-large-instruct',
                       "reranker_model_path": 'Qwen/Qwen3-Reranker-0.6B'},
            "endpoints": {"lm_studio_url": "http://localhost:1234/v1/chat/completions",
                          "ollama_url": "http://localhost:11434/v1/chat/completions"},
            "hardware": {"device": "Auto", "fp16": True},
            "default_model": "local-model", "project_paths": {}
        }
        if os.path.exists(self.config_file):
            with open(self.config_file, 'r') as f:
                try:
                    user_config = json.load(f)
                    for key, value in defaults.items():
                        if key not in user_config:
                            user_config[key] = value
                        elif isinstance(value, dict):
                            user_config[key] = {**value, **user_config[key]}
                    return user_config
                except json.JSONDecodeError:
                    print(f"Warning: {self.config_file} is corrupted.")
        return defaults

    def save_config(self):
        with open(self.config_file, 'w') as f: json.dump(self.user_config, f, indent=4)

    def update_backend_urls_from_config(self):
        endpoints = self.user_config.get("endpoints", {})
        self.backend_urls = {
            "LM Studio": endpoints.get("lm_studio_url", "http://localhost:1234/v1/chat/completions"),
            "Ollama": endpoints.get("ollama_url", "http://localhost:11434/v1/chat/completions")
        }

    def open_settings_window(self):
        settings_win = ctk.CTkToplevel(self);
        settings_win.title("Settings");
        settings_win.geometry("700x450");
        settings_win.transient(self)

        def create_model_row(key, text):
            frame = ctk.CTkFrame(settings_win);
            frame.pack(fill="x", padx=10, pady=5)
            ctk.CTkLabel(frame, text=text, width=150, anchor="w").pack(side="left")
            choices = [f"{name} ({hint})" for name, hint in self.model_choices[key].items()]
            current_model = self.model_config.get(f"{key}_model_path", list(self.model_choices[key].keys())[0])
            current_hint = self.model_choices[key].get(current_model, "")
            var = ctk.StringVar(value=f"{current_model} ({current_hint})")
            menu = ctk.CTkOptionMenu(frame, variable=var, values=choices, width=450);
            menu.pack(side="left", padx=5)
            return var

        models_frame = ctk.CTkFrame(settings_win);
        models_frame.pack(fill="x", padx=10, pady=10)
        ctk.CTkLabel(models_frame, text="Model Selection", font=("", 14, "bold")).pack(anchor="w", pady=5)
        code_model_var = create_model_row("code", "Code Model:")
        docs_model_var = create_model_row("docs", "Documents Model:")
        reranker_model_var = create_model_row("reranker", "Reranker Model:")

        hw_frame = ctk.CTkFrame(settings_win);
        hw_frame.pack(fill="x", padx=10, pady=10)
        ctk.CTkLabel(hw_frame, text="Hardware Settings", font=("", 14, "bold")).pack(anchor="w", pady=5)
        device_frame = ctk.CTkFrame(hw_frame);
        device_frame.pack(fill="x", pady=2)
        ctk.CTkLabel(device_frame, text="Compute Device:", width=150, anchor="w").pack(side="left", padx=5)
        device_var = ctk.StringVar(value=self.user_config.get("hardware", {}).get("device", "Auto"))
        ctk.CTkOptionMenu(device_frame, variable=device_var, values=["Auto", "GPU (CUDA)", "CPU"]).pack(side="left")
        fp16_var = ctk.BooleanVar(value=self.user_config.get("hardware", {}).get("fp16", False))
        ctk.CTkSwitch(hw_frame, text="Use Half-Precision (FP16)", variable=fp16_var).pack(anchor="w", pady=5, padx=5)

        def on_save():
            def get_model_name(s): return s.split(" (")[0]

            self.user_config["models"]["code_model_path"] = get_model_name(code_model_var.get())
            self.user_config["models"]["docs_model_path"] = get_model_name(docs_model_var.get())
            self.user_config["models"]["reranker_model_path"] = get_model_name(reranker_model_var.get())
            self.user_config["hardware"] = {"device": device_var.get(), "fp16": fp16_var.get()}
            self.save_config()
            self.model_config = self.user_config.get("models", {})
            self.update_backend_urls_from_config()
            settings_win.destroy()
            if self.project_var.get() != "No Indexes Found":
                self.load_engine(self.project_var.get())

        save_button = ctk.CTkButton(settings_win, text="Save and Reload Engine", command=on_save)
        save_button.pack(pady=20)


if __name__ == "__main__":
    app = RAGApp()
    app.mainloop()