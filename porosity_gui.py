"""
Porosity-Detect GUI

A graphical interface for automated void/porosity detection
in optical micrographs of aerospace materials.

No coding knowledge required — just click buttons.

Requirements:
    pip install numpy scipy matplotlib Pillow

Usage:
    python porosity_gui.py
"""

import tkinter as tk
from tkinter import filedialog, ttk, messagebox
from PIL import Image, ImageTk
import os
import sys
import threading
import json

# Add parent directory to path so we can import porosity_detect
sys.path.insert(0, os.path.dirname(__file__))


class PorosityDetectGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Porosity-Detect — Aerospace Void Analysis")
        self.root.geometry("1100x750")
        self.root.minsize(900, 650)
        self.root.configure(bg="#1a1a2e")

        # State
        self.image_path = None
        self.mask_path = None
        self.output_dir = None
        self.result = None
        self.overlay_image = None

        # Style
        self.style = ttk.Style()
        self.style.theme_use("clam")
        self.colors = {
            "bg": "#1a1a2e",
            "panel": "#16213e",
            "accent": "#0f3460",
            "highlight": "#e94560",
            "text": "#eaeaea",
            "text_dim": "#8892a0",
            "success": "#4ecca3",
            "warning": "#f0a500",
            "input_bg": "#0f3460",
        }

        self._configure_styles()
        self._build_ui()

    def _configure_styles(self):
        c = self.colors
        self.style.configure("Main.TFrame", background=c["bg"])
        self.style.configure("Panel.TFrame", background=c["panel"])
        self.style.configure("Title.TLabel", background=c["bg"], foreground=c["text"],
                             font=("Segoe UI", 18, "bold"))
        self.style.configure("Subtitle.TLabel", background=c["bg"], foreground=c["text_dim"],
                             font=("Segoe UI", 10))
        self.style.configure("Header.TLabel", background=c["panel"], foreground=c["text"],
                             font=("Segoe UI", 11, "bold"))
        self.style.configure("Info.TLabel", background=c["panel"], foreground=c["text_dim"],
                             font=("Segoe UI", 9))
        self.style.configure("Value.TLabel", background=c["panel"], foreground=c["success"],
                             font=("Consolas", 12, "bold"))
        self.style.configure("Result.TLabel", background=c["panel"], foreground=c["highlight"],
                             font=("Segoe UI", 28, "bold"))
        self.style.configure("Path.TLabel", background=c["panel"], foreground=c["text_dim"],
                             font=("Consolas", 8))

        # Buttons
        self.style.configure("Browse.TButton", font=("Segoe UI", 10), padding=(15, 8))
        self.style.configure("Run.TButton", font=("Segoe UI", 12, "bold"), padding=(30, 12))

        # Combobox
        self.style.configure("Preset.TCombobox", font=("Segoe UI", 10))

    def _build_ui(self):
        c = self.colors

        # Main container
        main = ttk.Frame(self.root, style="Main.TFrame")
        main.pack(fill=tk.BOTH, expand=True, padx=15, pady=15)

        # ── Title bar ──
        title_frame = ttk.Frame(main, style="Main.TFrame")
        title_frame.pack(fill=tk.X, pady=(0, 10))
        ttk.Label(title_frame, text="POROSITY-DETECT", style="Title.TLabel").pack(side=tk.LEFT)
        ttk.Label(title_frame, text="Aerospace Void Analysis Tool",
                  style="Subtitle.TLabel").pack(side=tk.LEFT, padx=(15, 0), pady=(5, 0))

        # ── Content area (left panel + right panel) ──
        content = ttk.Frame(main, style="Main.TFrame")
        content.pack(fill=tk.BOTH, expand=True)

        # Left panel — inputs
        left = tk.Frame(content, bg=c["panel"], relief=tk.FLAT, bd=0)
        left.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        left.configure(width=340)
        left.pack_propagate(False)

        left_inner = tk.Frame(left, bg=c["panel"])
        left_inner.pack(fill=tk.BOTH, expand=True, padx=15, pady=15)

        # Section: Image
        tk.Label(left_inner, text="1. SELECT IMAGE", font=("Segoe UI", 10, "bold"),
                 bg=c["panel"], fg=c["text"], anchor="w").pack(fill=tk.X, pady=(0, 5))

        self.image_btn = tk.Button(left_inner, text="Browse Image...",
                                   command=self._browse_image,
                                   bg=c["accent"], fg=c["text"], font=("Segoe UI", 10),
                                   relief=tk.FLAT, cursor="hand2", padx=15, pady=8)
        self.image_btn.pack(fill=tk.X, pady=(0, 3))
        self.image_label = tk.Label(left_inner, text="No image selected", font=("Consolas", 8),
                                    bg=c["panel"], fg=c["text_dim"], anchor="w", wraplength=300)
        self.image_label.pack(fill=tk.X, pady=(0, 15))

        # Section: Mask
        tk.Label(left_inner, text="2. ROI SELECTION", font=("Segoe UI", 10, "bold"),
                 bg=c["panel"], fg=c["text"], anchor="w").pack(fill=tk.X, pady=(0, 5))

        self.use_full_image_var = tk.BooleanVar(value=False)
        self.full_image_check = tk.Checkbutton(
            left_inner, text="Use entire image as ROI (no mask needed)",
            variable=self.use_full_image_var, command=self._toggle_mask,
            bg=c["panel"], fg=c["text"], selectcolor=c["accent"],
            activebackground=c["panel"], activeforeground=c["text"],
            font=("Segoe UI", 9))
        self.full_image_check.pack(fill=tk.X, pady=(0, 5))

        self.mask_btn = tk.Button(left_inner, text="Browse Mask...",
                                  command=self._browse_mask,
                                  bg=c["accent"], fg=c["text"], font=("Segoe UI", 10),
                                  relief=tk.FLAT, cursor="hand2", padx=15, pady=8)
        self.mask_btn.pack(fill=tk.X, pady=(0, 3))
        self.mask_label = tk.Label(left_inner, text="No mask selected", font=("Consolas", 8),
                                   bg=c["panel"], fg=c["text_dim"], anchor="w", wraplength=300)
        self.mask_label.pack(fill=tk.X, pady=(0, 15))

        # Section: Preset
        tk.Label(left_inner, text="3. DETECTION PRESET", font=("Segoe UI", 10, "bold"),
                 bg=c["panel"], fg=c["text"], anchor="w").pack(fill=tk.X, pady=(0, 5))

        self.preset_var = tk.StringVar(value="fabric_cross_section")
        presets = ["composite_high_mag", "composite_low_mag", "fabric_cross_section",
                   "am_metal", "sensitive", "conservative"]
        preset_menu = ttk.Combobox(left_inner, textvariable=self.preset_var,
                                    values=presets, state="readonly", font=("Segoe UI", 10))
        preset_menu.pack(fill=tk.X, pady=(0, 3))

        self.preset_desc = tk.Label(left_inner, text="Tuned for woven fabric full cross-sections",
                                     font=("Segoe UI", 8), bg=c["panel"], fg=c["text_dim"],
                                     anchor="w", wraplength=300)
        self.preset_desc.pack(fill=tk.X, pady=(0, 15))
        preset_menu.bind("<<ComboboxSelected>>", self._update_preset_desc)

        # Section: Advanced (collapsible)
        adv_frame = tk.LabelFrame(left_inner, text=" Advanced Options ",
                                   font=("Segoe UI", 9), bg=c["panel"], fg=c["text_dim"],
                                   relief=tk.GROOVE, bd=1)
        adv_frame.pack(fill=tk.X, pady=(0, 15))

        adv_inner = tk.Frame(adv_frame, bg=c["panel"])
        adv_inner.pack(fill=tk.X, padx=10, pady=8)

        # Strict threshold
        tk.Label(adv_inner, text="Strict threshold:", font=("Segoe UI", 9),
                 bg=c["panel"], fg=c["text_dim"]).grid(row=0, column=0, sticky="w", pady=2)
        self.strict_var = tk.StringVar(value="")
        tk.Entry(adv_inner, textvariable=self.strict_var, font=("Consolas", 9),
                 bg=c["input_bg"], fg=c["text"], insertbackground=c["text"],
                 relief=tk.FLAT, width=8).grid(row=0, column=1, padx=(5, 0), pady=2)

        # Moderate threshold
        tk.Label(adv_inner, text="Moderate threshold:", font=("Segoe UI", 9),
                 bg=c["panel"], fg=c["text_dim"]).grid(row=1, column=0, sticky="w", pady=2)
        self.moderate_var = tk.StringVar(value="")
        tk.Entry(adv_inner, textvariable=self.moderate_var, font=("Consolas", 9),
                 bg=c["input_bg"], fg=c["text"], insertbackground=c["text"],
                 relief=tk.FLAT, width=8).grid(row=1, column=1, padx=(5, 0), pady=2)

        # Min contrast
        tk.Label(adv_inner, text="Min contrast:", font=("Segoe UI", 9),
                 bg=c["panel"], fg=c["text_dim"]).grid(row=2, column=0, sticky="w", pady=2)
        self.min_contrast_var = tk.StringVar(value="")
        tk.Entry(adv_inner, textvariable=self.min_contrast_var, font=("Consolas", 9),
                 bg=c["input_bg"], fg=c["text"], insertbackground=c["text"],
                 relief=tk.FLAT, width=8).grid(row=2, column=1, padx=(5, 0), pady=2)

        # Pixel size
        tk.Label(adv_inner, text="Pixel size (µm):", font=("Segoe UI", 9),
                 bg=c["panel"], fg=c["text_dim"]).grid(row=3, column=0, sticky="w", pady=2)
        self.pixel_size_var = tk.StringVar(value="")
        tk.Entry(adv_inner, textvariable=self.pixel_size_var, font=("Consolas", 9),
                 bg=c["input_bg"], fg=c["text"], insertbackground=c["text"],
                 relief=tk.FLAT, width=8).grid(row=3, column=1, padx=(5, 0), pady=2)

        tk.Label(adv_inner, text="Leave blank to use preset defaults",
                 font=("Segoe UI", 7), bg=c["panel"], fg=c["text_dim"]
                 ).grid(row=4, column=0, columnspan=2, sticky="w", pady=(5, 0))

        # RUN button
        self.run_btn = tk.Button(left_inner, text="▶  ANALYZE",
                                  command=self._run_analysis,
                                  bg=c["highlight"], fg="white",
                                  font=("Segoe UI", 14, "bold"),
                                  relief=tk.FLAT, cursor="hand2",
                                  padx=20, pady=12, activebackground="#ff6b81")
        self.run_btn.pack(fill=tk.X, pady=(5, 10))

        # Status
        self.status_label = tk.Label(left_inner, text="Ready", font=("Segoe UI", 9),
                                      bg=c["panel"], fg=c["text_dim"], anchor="w")
        self.status_label.pack(fill=tk.X)

        # ── Right panel — results ──
        right = tk.Frame(content, bg=c["panel"], relief=tk.FLAT, bd=0)
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        right_inner = tk.Frame(right, bg=c["panel"])
        right_inner.pack(fill=tk.BOTH, expand=True, padx=15, pady=15)

        # Results header
        results_header = tk.Frame(right_inner, bg=c["panel"])
        results_header.pack(fill=tk.X, pady=(0, 10))

        tk.Label(results_header, text="RESULTS", font=("Segoe UI", 12, "bold"),
                 bg=c["panel"], fg=c["text"]).pack(side=tk.LEFT)

        self.save_btn = tk.Button(results_header, text="Save Results...",
                                   command=self._save_results, state=tk.DISABLED,
                                   bg=c["accent"], fg=c["text"], font=("Segoe UI", 9),
                                   relief=tk.FLAT, cursor="hand2", padx=10, pady=4)
        self.save_btn.pack(side=tk.RIGHT)

        # Big porosity number
        porosity_frame = tk.Frame(right_inner, bg=c["accent"], relief=tk.FLAT)
        porosity_frame.pack(fill=tk.X, pady=(0, 10), ipady=8)

        self.porosity_label = tk.Label(porosity_frame, text="—", font=("Segoe UI", 36, "bold"),
                                        bg=c["accent"], fg=c["text"])
        self.porosity_label.pack()
        tk.Label(porosity_frame, text="POROSITY", font=("Segoe UI", 10),
                 bg=c["accent"], fg=c["text_dim"]).pack()

        # Stats grid
        stats_frame = tk.Frame(right_inner, bg=c["panel"])
        stats_frame.pack(fill=tk.X, pady=(0, 10))

        stat_labels = [
            ("Void Count", "void_count"),
            ("ROI Area (px²)", "roi_area"),
            ("Total Void Area (px²)", "void_area"),
            ("Mean Void Size (px²)", "mean_area"),
            ("Max Void Size (px²)", "max_area"),
            ("Mean Circularity", "circularity"),
        ]

        self.stat_values = {}
        for i, (label_text, key) in enumerate(stat_labels):
            row = i // 3
            col = i % 3
            cell = tk.Frame(stats_frame, bg=c["panel"])
            cell.grid(row=row, column=col, padx=5, pady=5, sticky="ew")
            stats_frame.columnconfigure(col, weight=1)

            tk.Label(cell, text=label_text, font=("Segoe UI", 8),
                     bg=c["panel"], fg=c["text_dim"]).pack(anchor="w")
            val = tk.Label(cell, text="—", font=("Consolas", 11, "bold"),
                           bg=c["panel"], fg=c["success"])
            val.pack(anchor="w")
            self.stat_values[key] = val

        # Image preview
        preview_frame = tk.LabelFrame(right_inner, text=" Overlay Preview ",
                                       font=("Segoe UI", 9), bg=c["panel"], fg=c["text_dim"],
                                       relief=tk.GROOVE, bd=1)
        preview_frame.pack(fill=tk.BOTH, expand=True, pady=(5, 0))

        self.preview_canvas = tk.Canvas(preview_frame, bg="#0a0a1a", highlightthickness=0)
        self.preview_canvas.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.preview_canvas.bind("<Configure>", self._on_canvas_resize)

    def _update_preset_desc(self, event=None):
        descs = {
            "composite_high_mag": "High-magnification with visible individual fibers",
            "composite_low_mag": "Lower-magnification panel overview",
            "fabric_cross_section": "Tuned for woven fabric full cross-sections",
            "am_metal": "Additively manufactured Ti-6Al-4V, IN718",
            "sensitive": "Maximum detection — may include borderline regions",
            "conservative": "Minimum false positives — may miss small voids",
        }
        self.preset_desc.config(text=descs.get(self.preset_var.get(), ""))

    def _browse_image(self):
        path = filedialog.askopenfilename(
            title="Select Micrograph Image",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.tif *.tiff *.bmp"),
                ("All files", "*.*"),
            ]
        )
        if path:
            self.image_path = path
            self.image_label.config(text=os.path.basename(path))
            self.image_btn.config(bg=self.colors["success"])

    def _browse_mask(self):
        path = filedialog.askopenfilename(
            title="Select ROI Mask",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.tif *.tiff *.bmp"),
                ("All files", "*.*"),
            ]
        )
        if path:
            self.mask_path = path
            self.mask_label.config(text=os.path.basename(path))
            self.mask_btn.config(bg=self.colors["success"])

    def _toggle_mask(self):
        if self.use_full_image_var.get():
            self.mask_btn.config(state=tk.DISABLED, bg=self.colors["panel"])
            self.mask_label.config(text="Using entire image")
        else:
            self.mask_btn.config(state=tk.NORMAL, bg=self.colors["accent"])
            if self.mask_path:
                self.mask_label.config(text=os.path.basename(self.mask_path))
            else:
                self.mask_label.config(text="No mask selected")

    def _run_analysis(self):
        if not self.image_path:
            messagebox.showwarning("Missing Image", "Please select a micrograph image first.")
            return
        if not self.use_full_image_var.get() and not self.mask_path:
            messagebox.showwarning("Missing Mask", "Please select an ROI mask, or check 'Use entire image as ROI'.")
            return

        self.run_btn.config(state=tk.DISABLED, text="Analyzing...", bg=self.colors["warning"])
        self.status_label.config(text="Processing...", fg=self.colors["warning"])
        self.root.update()

        # Run in a thread so UI doesn't freeze
        thread = threading.Thread(target=self._do_analysis, daemon=True)
        thread.start()

    def _do_analysis(self):
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            import numpy as np
            from porosity_detect.two_pass import TwoPassDetector, TwoPassParams
            from scipy.ndimage import binary_dilation

            # Load image
            img = plt.imread(self.image_path)
            if img.ndim == 2:
                gray = img.copy()
            else:
                gray = np.mean(img[:, :, :3], axis=2)
            if gray.max() > 1.0:
                gray = gray / 255.0
            h, w = gray.shape

            # Load mask or use full image
            if self.use_full_image_var.get():
                roi = np.ones((h, w), dtype=bool)
            else:
                mask_img = plt.imread(self.mask_path)
                if mask_img.ndim == 3:
                    mask_gray = np.mean(mask_img[:, :, :3], axis=2)
                else:
                    mask_gray = mask_img.copy()
                if mask_gray.max() > 1.0:
                    mask_gray = mask_gray / 255.0
                roi = mask_gray > 0.5
                if roi.sum() < roi.size * 0.05:
                    roi = mask_gray < 0.5

            # Create detector
            preset = self.preset_var.get()
            detector = TwoPassDetector(preset=preset)

            # Apply overrides
            strict_val = self.strict_var.get().strip()
            if strict_val:
                detector.params.strict_threshold = float(strict_val)
            moderate_val = self.moderate_var.get().strip()
            if moderate_val:
                detector.params.moderate_threshold = float(moderate_val)
            contrast_val = self.min_contrast_var.get().strip()
            if contrast_val:
                detector.params.min_contrast = float(contrast_val)

            # Run detection
            result = detector.detect(gray, roi_mask=roi)
            self.result = result

            # Build overlay image
            overlay = np.stack([gray, gray, gray], axis=-1).copy()
            overlay[result["labels"] > 0] = [0.9, 0.15, 0.15]
            roi_outline = binary_dilation(roi, iterations=max(1, h // 200)) & ~roi
            overlay[roi_outline] = [0, 0.9, 0]
            overlay[~roi] *= 0.4

            # Store for saving later
            self.overlay_array = overlay
            self.gray = gray
            self.roi = roi

            # Convert to PIL for preview
            overlay_uint8 = (np.clip(overlay, 0, 1) * 255).astype(np.uint8)
            pil_img = Image.fromarray(overlay_uint8)
            self.overlay_pil = pil_img

            # Pixel size
            px_str = self.pixel_size_var.get().strip()
            self.pixel_size = float(px_str) if px_str else 1.0

            # Update UI on main thread
            self.root.after(0, self._update_results)

        except Exception as e:
            self.root.after(0, lambda: self._show_error(str(e)))

    def _update_results(self):
        import numpy as np
        r = self.result
        c = self.colors

        # Porosity
        pct = r["porosity_pct"]
        self.porosity_label.config(text=f"{pct:.3f}%")

        # Color code
        if pct < 2.0:
            color = self.colors["success"]
        elif pct < 4.0:
            color = self.colors["warning"]
        else:
            color = self.colors["highlight"]
        self.porosity_label.config(fg=color)

        # Stats
        self.stat_values["void_count"].config(text=str(r["void_count"]))
        self.stat_values["roi_area"].config(text=f"{r['roi_area_px']:,}")
        self.stat_values["void_area"].config(text=f"{r['total_void_area_px']:,}")

        if r["voids"]:
            areas = [v["area_px"] for v in r["voids"]]
            circs = [v.get("circularity", 0) for v in r["voids"]]
            self.stat_values["mean_area"].config(text=f"{np.mean(areas):.1f}")
            self.stat_values["max_area"].config(text=f"{max(areas):,}")
            self.stat_values["circularity"].config(text=f"{np.mean(circs):.3f}")
        else:
            self.stat_values["mean_area"].config(text="—")
            self.stat_values["max_area"].config(text="—")
            self.stat_values["circularity"].config(text="—")

        # Preview
        self._update_preview()

        # Re-enable
        self.run_btn.config(state=tk.NORMAL, text="▶  ANALYZE", bg=c["highlight"])
        self.save_btn.config(state=tk.NORMAL)
        self.status_label.config(text=f"Complete — {r['void_count']} voids detected", fg=c["success"])

    def _update_preview(self):
        if not hasattr(self, "overlay_pil") or self.overlay_pil is None:
            return

        canvas_w = self.preview_canvas.winfo_width()
        canvas_h = self.preview_canvas.winfo_height()
        if canvas_w < 10 or canvas_h < 10:
            return

        img = self.overlay_pil.copy()
        img_w, img_h = img.size

        # Fit to canvas
        scale = min(canvas_w / img_w, canvas_h / img_h)
        new_w = max(1, int(img_w * scale))
        new_h = max(1, int(img_h * scale))
        img = img.resize((new_w, new_h), Image.LANCZOS)

        self._preview_photo = ImageTk.PhotoImage(img)
        self.preview_canvas.delete("all")
        self.preview_canvas.create_image(canvas_w // 2, canvas_h // 2,
                                          image=self._preview_photo, anchor=tk.CENTER)

    def _on_canvas_resize(self, event):
        self._update_preview()

    def _save_results(self):
        if not self.result:
            return

        output_dir = filedialog.askdirectory(title="Select Output Folder")
        if not output_dir:
            return

        import numpy as np
        from PIL import Image as PILImage

        basename = os.path.splitext(os.path.basename(self.image_path))[0]
        r = self.result

        # 1. Full-resolution overlay PNG
        overlay_uint8 = (np.clip(self.overlay_array, 0, 1) * 255).astype(np.uint8)
        overlay_path = os.path.join(output_dir, f"{basename}_void_overlay.png")
        PILImage.fromarray(overlay_uint8).save(overlay_path)

        # 2. Text report
        report_lines = []
        report_lines.append("=" * 60)
        report_lines.append("POROSITY-DETECT ANALYSIS REPORT")
        report_lines.append("=" * 60)
        report_lines.append("")
        report_lines.append(f"Image:          {self.image_path}")
        report_lines.append(f"ROI Mask:       {self.mask_path if self.mask_path and not self.use_full_image_var.get() else 'Entire image (no mask)'}")
        report_lines.append(f"Image size:     {self.gray.shape[1]} x {self.gray.shape[0]} pixels")
        report_lines.append(f"Preset:         {self.preset_var.get()}")
        report_lines.append(f"Method:         two_pass_reconstruction")
        report_lines.append("")
        report_lines.append("-" * 60)
        report_lines.append("RESULTS")
        report_lines.append("-" * 60)
        report_lines.append(f"  Porosity:          {r['porosity_pct']:.4f}%")
        report_lines.append(f"  Void count:        {r['void_count']}")
        report_lines.append(f"  ROI area:          {r['roi_area_px']} px\u00B2")
        report_lines.append(f"  Total void area:   {r['total_void_area_px']} px\u00B2")

        if r["voids"]:
            areas = [v["area_px"] for v in r["voids"]]
            report_lines.append("")
            report_lines.append("-" * 60)
            report_lines.append("VOID STATISTICS")
            report_lines.append("-" * 60)
            report_lines.append(f"  Mean void area:    {np.mean(areas):.1f} px\u00B2")
            report_lines.append(f"  Max void area:     {max(areas)} px\u00B2")
            report_lines.append(f"  Min void area:     {min(areas)} px\u00B2")
            report_lines.append(f"  Mean circularity:  {np.mean([v.get('circularity', 0) for v in r['voids']]):.4f}")

            report_lines.append("")
            report_lines.append("-" * 60)
            report_lines.append("INDIVIDUAL VOIDS")
            report_lines.append("-" * 60)
            report_lines.append(f"  {'ID':>4s}  {'Area(px\u00B2)':>10s}  {'Circ':>6s}  {'CentX':>7s}  {'CentY':>7s}")
            for v in sorted(r["voids"], key=lambda x: x["area_px"], reverse=True):
                report_lines.append(
                    f"  {v['id']:>4d}  {v['area_px']:>10d}  {v.get('circularity', 0):>6.3f}  "
                    f"{v['centroid_x']:>7.1f}  {v['centroid_y']:>7.1f}"
                )

        report_lines.append("")
        report_lines.append("=" * 60)

        report_path = os.path.join(output_dir, f"{basename}_report.txt")
        with open(report_path, "w") as f:
            f.write("\n".join(report_lines))

        # 3. JSON
        json_data = {
            "image": self.image_path,
            "mask": self.mask_path if self.mask_path and not self.use_full_image_var.get() else "entire_image",
            "preset": self.preset_var.get(),
            "porosity_pct": r["porosity_pct"],
            "void_count": r["void_count"],
            "roi_area_px": r["roi_area_px"],
            "total_void_area_px": r["total_void_area_px"],
            "voids": r["voids"],
        }
        json_path = os.path.join(output_dir, f"{basename}_results.json")
        with open(json_path, "w") as f:
            json.dump(json_data, f, indent=2)

        messagebox.showinfo("Saved", f"Results saved to:\n\n{overlay_path}\n{report_path}\n{json_path}")
        self.status_label.config(text=f"Saved to {output_dir}", fg=self.colors["success"])

    def _show_error(self, msg):
        self.run_btn.config(state=tk.NORMAL, text="▶  ANALYZE", bg=self.colors["highlight"])
        self.status_label.config(text=f"Error: {msg}", fg=self.colors["highlight"])
        messagebox.showerror("Error", msg)


if __name__ == "__main__":
    import numpy as np  # needed for results display
    root = tk.Tk()
    app = PorosityDetectGUI(root)
    root.mainloop()
