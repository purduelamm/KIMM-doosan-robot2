import tkinter as tk

import numpy as np
from PIL import Image as PILImage
from PIL import ImageTk

def show_image_window(title: str, image: PILImage.Image) -> None:
    root = tk.Tk()
    root.title(title)
    photo = ImageTk.PhotoImage(image)
    label = tk.Label(root, image=photo)
    label.image = photo
    label.pack()
    tk.Button(root, text="Close", command=root.destroy).pack(fill=tk.X)
    root.bind("<Return>", lambda _event: root.destroy())
    root.mainloop()


def pick_xy_from_camera(snapshot: np.ndarray) -> tuple[float, float]:
    h, w = snapshot.shape[:2]
    root = tk.Tk()
    root.title("Camera view - click to select, then press Enter")
    image = PILImage.fromarray(snapshot)
    photo = ImageTk.PhotoImage(image)
    canvas = tk.Canvas(root, width=w, height=h)
    canvas.create_image(0, 0, anchor=tk.NW, image=photo)
    canvas.image = photo
    canvas.pack()

    state = {"x": None, "y": None}
    markers = {"items": []}

    def _on_click(event):
        for item in markers["items"]:
            canvas.delete(item)
        markers["items"].clear()

        px = float(np.clip(event.x, 0, w - 1))
        py = float(np.clip(event.y, 0, h - 1))
        state["x"], state["y"] = px, py
        markers["items"].extend(
            [
                canvas.create_line(0, py, w, py, fill="red", dash=(4, 2)),
                canvas.create_line(px, 0, px, h, fill="red", dash=(4, 2)),
                canvas.create_line(px - 8, py, px + 8, py, fill="red", width=2),
                canvas.create_line(px, py - 8, px, py + 8, fill="red", width=2),
                canvas.create_text(
                    px + 6,
                    py - 10,
                    text=f"({px:.1f}, {py:.1f})",
                    fill="red",
                    anchor=tk.W,
                ),
            ]
        )

    def _on_key(event):
        if state["x"] is not None:
            root.destroy()

    canvas.bind("<Button-1>", _on_click)
    root.bind("<Return>", _on_key)
    tk.Button(
        root,
        text="Confirm (or press Enter)",
        command=lambda: root.destroy() if state["x"] is not None else None,
    ).pack(fill=tk.X)

    root.mainloop()

    if state["x"] is None:
        raise RuntimeError("No point selected.")

    print(f"[picker] Selected pixel: x={state['x']:.1f},  y={state['y']:.1f}")
    return state["x"], state["y"]


# ── robot motion ──────────────────────────────────────────────────────────────
