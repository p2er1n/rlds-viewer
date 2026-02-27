import os
import io
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from collections import Counter
from contextlib import redirect_stderr, redirect_stdout

import numpy as np
from PIL import Image, ImageTk

try:
    import tensorflow_datasets as tfds
except Exception as exc:
    tfds = None
    _TFDS_IMPORT_ERROR = exc

try:
    from tkinterdnd2 import DND_FILES, TkinterDnD
except Exception:
    DND_FILES = None
    TkinterDnD = None


def _split_drop_paths(data):
    if not data:
        return []
    paths = []
    buf = ""
    in_brace = False
    for ch in data:
        if ch == "{":
            in_brace = True
            buf = ""
        elif ch == "}":
            in_brace = False
            if buf:
                paths.append(buf)
                buf = ""
        elif ch == " " and not in_brace:
            if buf:
                paths.append(buf)
                buf = ""
        else:
            buf += ch
    if buf:
        paths.append(buf)
    return paths


def _find_dataset_dir(path):
    if os.path.isfile(path) and os.path.basename(path) == "dataset_info.json":
        return os.path.dirname(path)
    if os.path.isdir(path) and os.path.isfile(os.path.join(path, "dataset_info.json")):
        return path
    if not os.path.isdir(path):
        return None
    max_depth = 3
    root_depth = path.rstrip(os.sep).count(os.sep)
    for root, dirs, files in os.walk(path):
        depth = root.count(os.sep) - root_depth
        if depth > max_depth:
            dirs[:] = []
            continue
        if "dataset_info.json" in files:
            return root
    return None


class RldsViewerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("RLDS Dataset Viewer")
        self.root.geometry("1200x800")

        self.dataset_dir = None
        self.episodes = []
        self.current_episode = None
        self.current_step = 0
        self.playing = False
        self.play_job = None
        self.max_episodes = 200
        self.exec_globals = {}

        self._build_ui()
        self._init_exec_env()

    def _build_ui(self):
        top = ttk.Frame(self.root, padding=8)
        top.pack(side=tk.TOP, fill=tk.X)

        self.drop_label = ttk.Label(
            top,
            text="Drop RLDS dataset folder here or click Open...",
            anchor="center",
        )
        self.drop_label.pack(side=tk.LEFT, fill=tk.X, expand=True)

        open_btn = ttk.Button(top, text="Open...", command=self._open_dialog)
        open_btn.pack(side=tk.RIGHT)

        status = ttk.Frame(self.root, padding=(8, 0, 8, 8))
        status.pack(side=tk.TOP, fill=tk.X)
        self.path_var = tk.StringVar(value="No dataset loaded")
        ttk.Label(status, textvariable=self.path_var, foreground="#555").pack(side=tk.LEFT)

        main = ttk.Frame(self.root, padding=8)
        main.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        left = ttk.Frame(main)
        left.pack(side=tk.LEFT, fill=tk.Y)

        ttk.Label(left, text="Episodes").pack(anchor="w")
        self.episode_list = tk.Listbox(left, width=24, height=30)
        self.episode_list.pack(side=tk.LEFT, fill=tk.Y, expand=True)
        self.episode_list.bind("<<ListboxSelect>>", self._on_episode_select)

        right = ttk.Frame(main)
        right.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        self.image_label = ttk.Label(right)
        self.image_label.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        controls = ttk.Frame(right, padding=(0, 8, 0, 0))
        controls.pack(side=tk.TOP, fill=tk.X)

        self.play_btn = ttk.Button(controls, text="Play", command=self._toggle_play)
        self.play_btn.pack(side=tk.LEFT)

        self.structure_btn = ttk.Button(
            controls,
            text="Trajectory Structure",
            command=self._show_trajectory_structure,
        )
        self.structure_btn.pack(side=tk.LEFT, padx=(8, 0))

        self.step_var = tk.IntVar(value=0)
        self.step_scale = ttk.Scale(
            controls,
            from_=0,
            to=0,
            orient=tk.HORIZONTAL,
            variable=self.step_var,
            command=self._on_step_change,
        )
        self.step_scale.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=8)

        self.step_label_var = tk.StringVar(value="Step 0/0")
        ttk.Label(controls, textvariable=self.step_label_var, width=14).pack(side=tk.LEFT)

        info = ttk.Frame(right, padding=(0, 8, 0, 0))
        info.pack(side=tk.TOP, fill=tk.X)
        self.info_text = tk.Text(info, height=10, wrap=tk.WORD)
        self.info_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.info_text.configure(state=tk.DISABLED)

        runner = ttk.LabelFrame(right, text="Python Runner", padding=8)
        runner.pack(side=tk.TOP, fill=tk.BOTH, expand=True, pady=(8, 0))

        runner_controls = ttk.Frame(runner)
        runner_controls.pack(side=tk.TOP, fill=tk.X)
        ttk.Button(runner_controls, text="Run", command=self._run_python_code).pack(side=tk.LEFT)
        ttk.Button(
            runner_controls,
            text="Template",
            command=self._insert_python_template,
        ).pack(side=tk.LEFT, padx=(8, 0))

        self.code_text = tk.Text(runner, height=8, wrap=tk.NONE)
        self.code_text.pack(side=tk.TOP, fill=tk.BOTH, expand=True, pady=(8, 0))
        self.code_text.insert(
            "1.0",
            "# Auto-imported: np, tfds, os, Image\n"
            "# Context: dataset_dir, episodes, current_episode, current_step, step, app\n"
            "print('episodes =', len(episodes))\n",
        )

        ttk.Label(runner, text="Output").pack(side=tk.TOP, anchor="w", pady=(8, 0))
        self.code_output_text = tk.Text(runner, height=8, wrap=tk.WORD)
        self.code_output_text.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        self.code_output_text.configure(state=tk.DISABLED)

        if TkinterDnD:
            self.drop_label.drop_target_register(DND_FILES)
            self.drop_label.dnd_bind("<<Drop>>", self._on_drop)

    def _set_info(self, text):
        self.info_text.configure(state=tk.NORMAL)
        self.info_text.delete("1.0", tk.END)
        self.info_text.insert(tk.END, text)
        self.info_text.configure(state=tk.DISABLED)

    def _set_code_output(self, text):
        self.code_output_text.configure(state=tk.NORMAL)
        self.code_output_text.delete("1.0", tk.END)
        self.code_output_text.insert(tk.END, text)
        self.code_output_text.configure(state=tk.DISABLED)

    def _init_exec_env(self):
        self.exec_globals = {
            "__builtins__": __builtins__,
            "np": np,
            "tfds": tfds,
            "os": os,
            "Image": Image,
        }
        self._refresh_exec_context()

    def _current_step_data(self):
        if self.current_episode is None:
            return None
        steps = self.current_episode.get("steps", [])
        if not steps:
            return None
        idx = min(max(self.current_step, 0), len(steps) - 1)
        return steps[idx]

    def _refresh_exec_context(self):
        self.exec_globals.update(
            {
                "app": self,
                "dataset_dir": self.dataset_dir,
                "episodes": self.episodes,
                "current_episode": self.current_episode,
                "current_step": self.current_step,
                "step": self._current_step_data(),
            }
        )

    def _insert_python_template(self):
        template = (
            "# Auto-imported: np, tfds, os, Image\n"
            "# Context: dataset_dir, episodes, current_episode, current_step, step, app\n"
            "print('dataset_dir =', dataset_dir)\n"
            "print('num_episodes =', len(episodes))\n"
            "if current_episode is not None:\n"
            "    print('num_steps =', len(current_episode['steps']))\n"
            "    print('step keys =', list(step.keys()))\n"
        )
        self.code_text.delete("1.0", tk.END)
        self.code_text.insert("1.0", template)

    def _run_python_code(self):
        code = self.code_text.get("1.0", tk.END).strip()
        if not code:
            self._set_code_output("No code to run.")
            return

        self._refresh_exec_context()
        out_buf = io.StringIO()
        result = None
        error = None

        try:
            with redirect_stdout(out_buf), redirect_stderr(out_buf):
                try:
                    compiled = compile(code, "<rlds_viewer>", "eval")
                    result = eval(compiled, self.exec_globals, self.exec_globals)
                except SyntaxError:
                    compiled = compile(code, "<rlds_viewer>", "exec")
                    exec(compiled, self.exec_globals, self.exec_globals)
        except Exception as exc:
            error = exc

        output = out_buf.getvalue()
        if result is not None:
            output += repr(result) + "\n"
        if error is not None:
            output += f"{type(error).__name__}: {error}\n"
        if not output:
            output = "Code executed successfully (no output)."
        self._set_code_output(output)

    def _open_dialog(self):
        path = filedialog.askdirectory()
        if path:
            self._load_dataset(path)

    def _on_drop(self, event):
        paths = _split_drop_paths(event.data)
        if paths:
            self._load_dataset(paths[0])

    def _load_dataset(self, path):
        if tfds is None:
            messagebox.showerror(
                "Missing dependency",
                f"tensorflow_datasets import failed: {_TFDS_IMPORT_ERROR}",
            )
            return

        dataset_dir = _find_dataset_dir(path)
        if not dataset_dir:
            messagebox.showerror(
                "Invalid dataset",
                "Could not find dataset_info.json under the selected path.",
            )
            return

        self.dataset_dir = dataset_dir
        self.path_var.set(f"Loading: {dataset_dir}")
        self.episode_list.delete(0, tk.END)
        self.episodes = []
        self.current_episode = None
        self.current_step = 0
        self._set_info("")
        self._refresh_exec_context()

        thread = threading.Thread(target=self._load_worker, daemon=True)
        thread.start()

    def _load_worker(self):
        try:
            builder = tfds.builder_from_directory(self.dataset_dir)
            ds = builder.as_dataset(split="train")
            episodes = []
            for idx, episode in enumerate(tfds.as_numpy(ds)):
                steps = list(episode["steps"])
                episode = dict(episode)
                episode["steps"] = steps
                episodes.append(episode)
                if idx + 1 >= self.max_episodes:
                    break
            self.episodes = episodes
            self.root.after(0, self._on_dataset_loaded, builder.name)
        except Exception as exc:
            self.root.after(0, self._on_dataset_error, exc)

    def _on_dataset_loaded(self, name):
        self.path_var.set(f"Loaded {name} from {self.dataset_dir} ({len(self.episodes)} episodes)")
        for idx in range(len(self.episodes)):
            self.episode_list.insert(tk.END, f"Episode {idx:04d}")
        self._refresh_exec_context()
        if self.episodes:
            self.episode_list.selection_set(0)
            self._load_episode(0)

    def _on_dataset_error(self, exc):
        messagebox.showerror("Load failed", str(exc))
        self.path_var.set("Load failed")

    def _on_episode_select(self, _event):
        sel = self.episode_list.curselection()
        if not sel:
            return
        self._load_episode(sel[0])

    def _load_episode(self, index):
        if index < 0 or index >= len(self.episodes):
            return
        self.current_episode = self.episodes[index]
        steps = self.current_episode["steps"]
        self.current_step = 0
        self.step_scale.configure(to=max(len(steps) - 1, 0))
        self.step_var.set(0)
        self._render_step()
        self._refresh_exec_context()

    def _on_step_change(self, _value):
        if self.current_episode is None:
            return
        self.current_step = int(float(self.step_var.get()))
        self._render_step()
        self._refresh_exec_context()

    def _render_step(self):
        if self.current_episode is None:
            return
        steps = self.current_episode["steps"]
        if not steps:
            return
        step = steps[self.current_step]
        image = step["observation"]["image"]
        if image is not None:
            pil_img = Image.fromarray(image)
            max_w, max_h = 900, 500
            scale = min(max_w / pil_img.width, max_h / pil_img.height, 1.0)
            if scale < 1.0:
                new_size = (int(pil_img.width * scale), int(pil_img.height * scale))
                pil_img = pil_img.resize(new_size, Image.BILINEAR)
            tk_img = ImageTk.PhotoImage(pil_img)
            self.image_label.configure(image=tk_img)
            self.image_label.image = tk_img

        action = step.get("action")
        state = step.get("observation", {}).get("state")
        instruction = step.get("language_instruction", "")
        action_str = np.array2string(action, precision=4) if action is not None else "None"
        state_str = np.array2string(state, precision=4) if state is not None else "None"
        info = (
            f"Step: {self.current_step}\n"
            f"Instruction: {instruction}\n"
            f"Action: {action_str}\n"
            f"State: {state_str}\n"
        )
        self._set_info(info)
        self.step_label_var.set(f"Step {self.current_step + 1}/{len(steps)}")

    def _describe_value(self, value):
        if isinstance(value, np.ndarray):
            return f"ndarray(shape={value.shape}, dtype={value.dtype})"
        if isinstance(value, np.generic):
            return f"scalar(dtype={value.dtype}, value={value.item()})"
        if isinstance(value, dict):
            return "dict"
        if isinstance(value, (list, tuple)):
            return f"{type(value).__name__}(len={len(value)})"
        return type(value).__name__

    def _collect_paths(self, value, prefix="", out=None):
        if out is None:
            out = {}
        if isinstance(value, dict):
            for key, item in value.items():
                next_prefix = f"{prefix}.{key}" if prefix else str(key)
                self._collect_paths(item, next_prefix, out)
            return out
        if isinstance(value, (list, tuple)):
            path = prefix if prefix else "<root>"
            out.setdefault(path, []).append(self._describe_value(value))
            if value:
                self._collect_paths(value[0], f"{path}[0]", out)
            return out
        path = prefix if prefix else "<root>"
        out.setdefault(path, []).append(self._describe_value(value))
        return out

    def _format_trajectory_structure(self, episode):
        steps = episode.get("steps", [])
        lines = [
            "Trajectory Structure",
            f"Num steps: {len(steps)}",
            "",
        ]

        episode_meta = {k: v for k, v in episode.items() if k != "steps"}
        if episode_meta:
            lines.append("Episode-level fields:")
            for key, value in episode_meta.items():
                lines.append(f"- {key}: {self._describe_value(value)}")
            lines.append("")

        if not steps:
            lines.append("No steps found.")
            return "\n".join(lines)

        lines.append("Per-step fields (summarized across steps):")
        path_desc = {}
        for step in steps:
            self._collect_paths(step, out=path_desc)

        for path in sorted(path_desc):
            counter = Counter(path_desc[path])
            desc = ", ".join(f"{d} x{counter[d]}" for d in sorted(counter))
            lines.append(f"- {path}: {desc}")

        return "\n".join(lines)

    def _show_trajectory_structure(self):
        if self.current_episode is None:
            messagebox.showinfo("No episode", "Please load and select an episode first.")
            return
        self._set_info(self._format_trajectory_structure(self.current_episode))

    def _toggle_play(self):
        if self.current_episode is None:
            return
        self.playing = not self.playing
        self.play_btn.configure(text="Pause" if self.playing else "Play")
        if self.playing:
            self._play_tick()
        else:
            if self.play_job is not None:
                self.root.after_cancel(self.play_job)
                self.play_job = None

    def _play_tick(self):
        if not self.playing or self.current_episode is None:
            return
        steps = self.current_episode["steps"]
        if self.current_step + 1 < len(steps):
            self.current_step += 1
            self.step_var.set(self.current_step)
            self._render_step()
            self.play_job = self.root.after(200, self._play_tick)
        else:
            self.playing = False
            self.play_btn.configure(text="Play")


def main():
    if TkinterDnD:
        root = TkinterDnD.Tk()
    else:
        root = tk.Tk()
    app = RldsViewerApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
