#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
    Created on Wed Jan 23 21:40:07 2025
    @author: STRH/99411425
    RL in Control Comprehensive Project
    Q1: App
'''

import tkinter as tk
from tkinter import filedialog, messagebox
import os
import subprocess

def run_script(args):
    """Run the command-line script with the provided arguments."""
    command = ["python", "main.py"] + args
    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as e:
        messagebox.showerror("Error", f"An error occurred while running the script: {e}")

def select_qtable():
    """Open file dialog to select Q-table file."""
    filepath = filedialog.askopenfilename(filetypes=[("Pickle files", "*.pkl"), ("All files", "*.*")])
    qtable_path.set(filepath)

def start_process():
    """Start the training or testing process based on user input."""
    method = method_var.get()
    mode = mode_var.get()
    episodes = episodes_var.get()

    if not method:
        messagebox.showerror("Error", "Please select a method (Q-learning, SARSA, or Monte Carlo).")
        return

    if mode == "train":
        learning_rate = learning_rate_var.get()
        discount_factor = discount_factor_var.get()
        epsilon = epsilon_var.get()
        epsilon_decay = epsilon_decay_var.get()

        args = ["--method", method, "--train", "train", "--render", "no", "--episodes", str(episodes),
                "--learning_rate", str(learning_rate), "--discount_factor", str(discount_factor),
                "--epsilon", str(epsilon), "--epsilon_decay", str(epsilon_decay)]
    else:
        if not qtable_path.get():
            messagebox.showerror("Error", "Please select a Q-table file for testing.")
            return

        args = ["--method", method, "--train", "test", "--render", "yes", "--episodes", str(episodes),
                "--qtable", qtable_path.get()]

    run_script(args)

# Initialize the GUI
root = tk.Tk()
root.title("CartPole RL Training and Testing")
root.geometry("400x800")
root.configure(bg="black")

# Variables
mode_var = tk.StringVar(value="train")
method_var = tk.StringVar()
episodes_var = tk.IntVar(value=15000)
learning_rate_var = tk.DoubleVar(value=0.01)
discount_factor_var = tk.DoubleVar(value=0.8)
epsilon_var = tk.DoubleVar(value=0.7)
epsilon_decay_var = tk.DoubleVar(value=0.00001)
qtable_path = tk.StringVar()

# Styling
label_font = ("courier 10 pitch", 14, "bold")
entry_font = ("courier 10 pitch", 12, "bold")
button_font = ("courier 10 pitch", 14, "bold")
radio_font = ("courier 10 pitch", 12, "bold")

# Mode Selection
tk.Label(root, text="Select Mode:", font=label_font, fg="white", bg="black").pack(pady=10)
train_radio = tk.Radiobutton(root, text="Train", variable=mode_var, value="train", font=radio_font, fg="black", bg="green", command=lambda: mode_changed("train"))
test_radio = tk.Radiobutton(root, text="Test", variable=mode_var, value="test", font=radio_font, fg="black", bg="green", command=lambda: mode_changed("test"))
train_radio.pack()
test_radio.pack()

# Method Selection
tk.Label(root, text="Select Method:", font=label_font, fg="white", bg="black").pack(pady=10)
methods = ["qlearning", "sarsa", "montecarlo"]
for method in methods:
    tk.Radiobutton(root, text=method.capitalize(), variable=method_var, value=method, font=radio_font, fg="black", bg="green").pack(anchor="w", padx=20)

# Training Parameters
params_frame = tk.Frame(root, bg="black")
params_frame.pack(pady=10, fill="x", padx=20)

params_title = tk.Label(params_frame, text="Training Parameters", font=label_font, fg="white", bg="black")
params_title.pack()

train_widgets = []

# Episodes
tk.Label(params_frame, text="Episodes:", font=entry_font, fg="white", bg="black").pack(anchor="w")
episodes_entry = tk.Entry(params_frame, textvariable=episodes_var, font=entry_font)
episodes_entry.pack(anchor="w", fill="x")

# Learning Rate
tk.Label(params_frame, text="Learning Rate:", font=entry_font, fg="white", bg="black").pack(anchor="w")
learning_rate_entry = tk.Entry(params_frame, textvariable=learning_rate_var, font=entry_font)
learning_rate_entry.pack(anchor="w", fill="x")
train_widgets.append(learning_rate_entry)

# Discount Factor
tk.Label(params_frame, text="Discount Factor:", font=entry_font, fg="white", bg="black").pack(anchor="w")
discount_factor_entry = tk.Entry(params_frame, textvariable=discount_factor_var, font=entry_font)
discount_factor_entry.pack(anchor="w", fill="x")
train_widgets.append(discount_factor_entry)

# Epsilon
tk.Label(params_frame, text="Epsilon:", font=entry_font, fg="white", bg="black").pack(anchor="w")
epsilon_entry = tk.Entry(params_frame, textvariable=epsilon_var, font=entry_font)
epsilon_entry.pack(anchor="w", fill="x")
train_widgets.append(epsilon_entry)

# Epsilon Decay
tk.Label(params_frame, text="Epsilon Decay:", font=entry_font, fg="white", bg="black").pack(anchor="w")
epsilon_decay_entry = tk.Entry(params_frame, textvariable=epsilon_decay_var, font=entry_font)
epsilon_decay_entry.pack(anchor="w", fill="x")
train_widgets.append(epsilon_decay_entry)

# Q-table Selection (Test Mode Only)
tk.Label(root, text="Q-Table (for Test Mode):", font=label_font, fg="white", bg="black").pack(pady=10)
qtable_button = tk.Button(root, text="Select Q-Table", command=select_qtable, font=button_font, bg="gray", fg="white")
qtable_button.pack()
qtable_label = tk.Label(root, textvariable=qtable_path, font=entry_font, fg="white", bg="black", wraplength=300)
qtable_label.pack()

# Start Button
tk.Button(root, text="Start", command=start_process, font=button_font, bg="green", fg="white").pack(pady=20)

# Function to toggle widgets based on mode
def mode_changed(mode):
    if mode == "train":
        for widget in train_widgets:
            widget.configure(state="normal")
        qtable_button.configure(state="disabled")
        qtable_label.configure(text="")
    else:
        for widget in train_widgets:
            widget.configure(state="disabled")
        qtable_button.configure(state="normal")

# Initialize with train mode
mode_changed("train")

root.mainloop()
