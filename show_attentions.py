import pdb
import json
import sys

from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
import tarfile
import wget

from matplotlib import cm

from PIL import Image
from toolz import partition_all, dissoc

st.set_page_config(layout="wide")

DIR_DATA = Path("data/whatsup_vlms_data")
PATH_ATTENTIONS = Path("output/attentions-Controlled_Images_B.h5")


def download(url, path):
    if path.exists():
        return False
    # st.info(f"Downloading {url} to {path}...")
    path.parent.mkdir(parents=True, exist_ok=True)
    wget.download(url, str(path))
    return True


def extract_archive(archive_path, extract_to):
    # st.info(f"Extracting {archive_path} to {extract_to}...")
    with tarfile.open(archive_path, "r:gz") as tar:
        tar.extractall(path=extract_to)


@st.cache_data
def download_data():
    URL_DATA_FILELIST = (
        "https://sharing.speed.pub.ro/owncloud/index.php/s/o5rXQGOtHheejET/download"
    )
    path = DIR_DATA / "controlled_clevr_dataset.json"
    download(URL_DATA_FILELIST, path)

    URL_DATA_IMAGES = (
        "https://sharing.speed.pub.ro/owncloud/index.php/s/NCW0DGQqC6uPNvf/download"
    )
    path = DIR_DATA / "whatsup_vlms_data.tar.gz"
    downloaded = download(URL_DATA_IMAGES, path)
    if downloaded:
        extract_archive(path, DIR_DATA)

    URL_ATTENTIONS = (
        "https://sharing.speed.pub.ro/owncloud/index.php/s/bFmatxDAqzy6Bcw/download"
    )
    download(URL_ATTENTIONS, PATH_ATTENTIONS)


@st.cache_resource
def load_h5py_file(path):
    return h5py.File(path, "r")


@st.cache_data
def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def show_attention_to_image(attentions, caption, image, tokens):
    W, H = image.size
    PATCH_SIZE = 32
    W = W // PATCH_SIZE
    H = H // PATCH_SIZE
    IMAGE_TOKEN = "<|image_pad|>"
    image_idxs = [i for i, t in enumerate(tokens) if t == IMAGE_TOKEN]
    attentions = attentions[image_idxs]
    attentions = attentions.reshape(H, W)
    max_val = attentions.max()
    attentions = attentions / max_val
    caption = caption + " · max: {:.2f}".format(max_val)
    image_attentions = cm.viridis(attentions)
    image_attentions = np.uint8(image_attentions * 255)
    image_attentions = Image.fromarray(image_attentions)
    st.image(image_attentions, caption=caption, width="stretch")


def show_attention_to_other(attentions, head_idx, tokens, to_drop_start_tokens=False):
    IMAGE_TOKEN = "<|image_pad|>"
    other_idxs = [i for i, t in enumerate(tokens) if t != IMAGE_TOKEN]
    if to_drop_start_tokens:
        other_idxs = other_idxs[5:]
    other_tokens = [repr(tokens[i]) for i in other_idxs]
    attentions = attentions[:, head_idx]
    attentions = attentions[:, other_idxs]
    attentions = attentions.T  # (num_other_tokens, num_layers)
    _, num_layers = attentions.shape
    layers = ["{}".format(i) for i in range(num_layers)]
    df = pd.DataFrame(
        attentions,
        index=other_tokens,
        columns=layers,
    )
    S = 0.3
    ncols = attentions.shape[0]
    nrows = attentions.shape[1]
    plt.close("all")
    fig, ax = plt.subplots(figsize=(S * ncols, S * nrows))
    sns.heatmap(df, square=True, cbar=False, ax=ax)
    ax.set_title("Head: {} · max: {:.2f}".format(head_idx, attentions.max()))
    ax.set_xlabel("Layer")
    fig.tight_layout()
    st.pyplot(fig)


def get_image_path(data, idx):
    datum = data[idx]
    image_filename = Path(datum["image_path"]).name
    image_path = DIR_DATA / "controlled_clevr" / image_filename
    return image_path


def compute_attention_identity(num_layers, seq_len):
    attn = np.ones((seq_len, seq_len))
    attn = np.tril(attn)
    return attn


def compute_attention_receptive_field(num_layers, seq_len):
    attn = np.zeros((seq_len, seq_len))
    for k in range(seq_len):
        attn += np.diag(np.full(seq_len - k, k + 1), -k)
    attn = attn / attn.sum(axis=-1, keepdims=True)
    return attn


def compute_attention_num_paths(num_layers, seq_len):
    step = np.ones((seq_len, seq_len))
    step = np.tril(step)
    attn = np.eye(seq_len)
    for i in range(num_layers):
        attn = attn @ step
        attn = attn / attn.sum(axis=-1, keepdims=True)
    return attn


ROLLOUT_NORM_FUNCS = {
    "no normalization": compute_attention_identity,
    "receptive field": compute_attention_receptive_field,
    "number of paths": compute_attention_num_paths,
}


def main():
    download_data()

    h5py_file = load_h5py_file(PATH_ATTENTIONS)
    data = load_json(DIR_DATA / "controlled_clevr_dataset.json")

    with st.sidebar:
        num_samples = len(h5py_file.keys())
        idx = st.number_input(
            "Sample index",
            min_value=0,
            max_value=num_samples - 1,
            value=0,
        )

        def get_key_str(result, key):
            return result[key][()].decode("utf-8")

        results = h5py_file[str(idx)]
        question = get_key_str(results, "question")
        answer_pred = get_key_str(results, "generated-text")
        answer_true = get_key_str(results, "preposition")

        # cols = st.columns([1, 2])
        path_image = get_image_path(data, idx)
        image = Image.open(path_image)

        st.image(path_image)
        st.markdown(
            """
        - Prompt: {}
        - Answer (pred): {}
        - Answer (true): {}
                    """.format(
                question,
                answer_pred,
                answer_true,
            )
        )

        st.markdown("---")
        st.markdown("### Attention rollout from a selected token")

        tokens = [t.decode("utf-8") for t in results["input-tokens"][()]]
        tokens_non_image_idxs = [
            i for i, t in enumerate(tokens) if t != "<|image_pad|>"
        ]
        col0, col1 = st.columns(2)
        query_idx = col0.selectbox(
            "Query token",
            tokens_non_image_idxs,
            format_func=lambda i: repr(tokens[i]) + " ({})".format(i),
            index=len(tokens_non_image_idxs) - 1,
        )
        norm_rollout = col1.selectbox(
            "Rollout normalization",
            ROLLOUT_NORM_FUNCS,
            help="How to scale the attention rollout. This normalizations accounts for the fact that earlier tokens are attended to more strongly due to the causal structure of the decoder.",
        )

        attentions = results["attentions"][()]
        num_layers, num_heads, seq_len = attentions.shape

        attentions_rollout = results["attention-rollout"][()]
        attentions_norm = ROLLOUT_NORM_FUNCS[norm_rollout](num_layers, seq_len)
        attentions_norm[attentions_norm == 0] = 1e-9
        attentions_rollout = attentions_rollout / attentions_norm
        attentions_rollout = attentions_rollout / attentions_rollout.sum(
            axis=-1,
            keepdims=True,
        )
        attentions_rollout_q = attentions_rollout[query_idx]
        show_attention_to_image(
            attentions_rollout_q,
            "Attention to image patches",
            image,
            tokens,
        )

        to_drop_start_tokens_0 = st.checkbox(
            "Ignore tokens before the prompt (attention to non-image tokens)",
            value=False,
        )

        if to_drop_start_tokens_0:
            tokens_non_image_idxs = tokens_non_image_idxs[5:]

        tokens_non_image_idxs = tokens_non_image_idxs
        attentions_other = attentions_rollout_q[tokens_non_image_idxs]
        fig, ax = plt.subplots(figsize=(3, 6))
        tokens_labels = [repr(tokens[i]) for i in tokens_non_image_idxs]
        ys = np.arange(len(attentions_other))
        ys = ys[::-1]
        ax.barh(ys, attentions_other)
        ax.set_yticks(ys)
        ax.set_yticklabels(tokens_labels)
        ax.set_title("Attention to non-image tokens")
        fig.set_tight_layout(True)
        st.pyplot(fig)

    st.markdown("### Attention of last token to image patches")
    cols1, cols2 = st.columns(2)
    with cols1:
        col0, col1, col2 = st.columns(3)
        layers_start = col0.number_input(
            "First layer",
            min_value=0,
            max_value=num_layers - 1,
            value=0,
        )
        layers_step = col1.number_input(
            "Step layer",
            min_value=1,
            max_value=num_layers,
            value=5,
        )
        layers_end = col2.number_input(
            "Last layer",
            min_value=layers_start,
            max_value=num_layers,
            value=num_layers,
        )

    with cols2:
        col0, col1, col2 = st.columns(3)
        heads_start = col0.number_input(
            "First head",
            min_value=0,
            max_value=num_heads - 1,
            value=0,
        )
        heads_step = col1.number_input(
            "Step head",
            min_value=1,
            max_value=num_heads,
            value=1,
        )
        heads_end = col2.number_input(
            "Last head",
            min_value=heads_start,
            max_value=num_heads,
            value=16,
            help="Maximum value is {}".format(num_heads),
        )

    if layers_step > (layers_end - layers_start):
        st.error("Step is too large")
        st.stop()

    if layers_end <= layers_start:
        st.error("Last layer must be greater than first layer")
        st.stop()

    layer_idxs = list(range(layers_start, layers_end, layers_step))
    head_idxs = list(range(heads_start, heads_end, heads_step))

    for head_idx in head_idxs:
        cols = st.columns(len(layer_idxs))
        for col, layer_idx in zip(cols, layer_idxs):
            with col:
                caption = "L: {} · H: {}".format(layer_idx, head_idx)
                show_attention_to_image(
                    attentions[layer_idx, head_idx], caption, image, tokens
                )

    st.markdown("### Attention of last token to non-image tokens")
    to_drop_start_tokens = st.checkbox("Ignore tokens before the prompt", value=False)

    ncols = 2
    for group in partition_all(ncols, head_idxs):
        cols = st.columns(ncols)
        for col, head_idx in zip(cols, group):
            with col:
                show_attention_to_other(
                    attentions, head_idx, tokens, to_drop_start_tokens
                )


if __name__ == "__main__":
    main()
