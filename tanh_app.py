# Streamlit app: Visualise the Hyperbolic Tangent (Tanh) activation function

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons, make_circles
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from matplotlib.colors import ListedColormap

st.set_page_config(page_title="Tanh Activation Function", layout="wide")

st.title("Hyperbolic Tangent (Tanh)")
st.write("Tanh transforms any input into a value between -1 and 1, and is centred at zero.")
st.latex(r"f(x) = \tanh(x) = \frac{e^{x} - e^{-x}}{e^{x} + e^{-x}}")
st.latex(r"f'(x) = 1 - \tanh^2(x)")

# --- Sidebar controls ---
with st.sidebar:
    st.header("Curve")
    x_input = st.slider("Input value x", -6.0, 6.0, 1.0, 0.1)

    st.header("Neural Network Demo")
    ds = st.selectbox("Dataset", ["Moons", "Circles"])
    n_samples = st.slider("Samples", 100, 1000, 300, 50)
    noise = st.slider("Noise", 0.00, 0.40, 0.20, 0.01)
    with st.expander("Advanced settings"):
        hidden = st.text_input("Hidden layer sizes (e.g., 16,16)", value="16,16")
        max_iter = st.slider("Max Iterations", 100, 3000, 800, 100)

random_state = 42  # fixed seed so results are reproducible

# --- Activation function ---
def tanh(x):
    return np.tanh(x)

def tanh_deriv(x):
    return 1 - np.tanh(x) ** 2

# --- 1. Curve ---
st.subheader("1. Tanh Curve")
xgrid = np.linspace(-6, 6, 400)

col1, col2 = st.columns(2)
box_height = 400  

with col1.container(border=True, height=box_height):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(xgrid, tanh(xgrid), label="Tanh f(x)", color="purple")
    ax.plot(xgrid, tanh_deriv(xgrid), label="Derivative f'(x)", color="orange", linestyle="--")
    ax.scatter([x_input], [tanh(x_input)], color="red", zorder=5, label="Your input")
    ax.axhline(0, color="black", linewidth=0.5)
    ax.axvline(0, color="black", linewidth=0.5)
    ax.set_xlabel("x"); ax.set_ylabel("output")
    ax.set_title("Tanh Activation")
    ax.legend(); ax.grid(True)
    st.pyplot(fig)

with col2.container(border=True, height=box_height):
    st.write("**Output for your input**")
    m1, m2, m3 = st.columns(3)
    m1.metric("x", f"{x_input:.2f}")
    m2.metric("Tanh(x)", f"{tanh(x_input):.4f}")
    m3.metric("f'(x)", f"{tanh_deriv(x_input):.4f}")
    st.write("**Properties**")
    st.write("- Range: (-1, 1)")
    st.write("- Zero-centred output, which helps training compared to sigmoid")
    st.write("- Maximum gradient is 1 (at x = 0)")
    st.write("- Still saturates for large |x| (vanishing gradient)")

# --- 2. Neural network using Tanh ---
st.subheader("2. Neural Network using Tanh")

if ds == "Moons":
    X, y = make_moons(n_samples=n_samples, noise=noise, random_state=random_state)
else:
    X, y = make_circles(n_samples=n_samples, noise=noise, factor=0.5, random_state=random_state)

hidden_sizes = tuple(int(x.strip()) for x in hidden.split(",") if x.strip().isdigit())
if not hidden_sizes:
    hidden_sizes = (16, 16)

clf = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", MLPClassifier(hidden_layer_sizes=hidden_sizes,
                          activation="tanh",
                          learning_rate_init=0.01,
                          max_iter=max_iter,
                          random_state=random_state))
])
clf.fit(X, y)
score = clf.score(X, y)

# --- Plot decision boundary ---
xx, yy = np.meshgrid(
    np.linspace(X[:, 0].min() - 1, X[:, 0].max() + 1, 300),
    np.linspace(X[:, 1].min() - 1, X[:, 1].max() + 1, 300)
)
Z = clf.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

cmap_light = ListedColormap(["#FFEEEE", "#EEFFEE"])
cmap_bold = ["#CC0000", "#007700"]

fig2, ax2 = plt.subplots(figsize=(6, 5))
ax2.contourf(xx, yy, Z, cmap=cmap_light, alpha=0.7)
ax2.scatter(X[:, 0], X[:, 1], c=y, s=25, edgecolor="k", cmap=ListedColormap(cmap_bold))
ax2.set_title(f"MLP with Tanh (acc={score:.3f})")
ax2.set_xlabel("x1"); ax2.set_ylabel("x2")

left, mid, right = st.columns([1, 3, 1])
with mid:
    st.pyplot(fig2)