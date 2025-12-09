import streamlit as st
import pandas as pd
import time  
from PIL import Image
import base64

from shift_utils import (
    # Statistical tests
    run_ks_test,
    run_mannwhitney_test,
    run_cramervonmises_test,
    run_chi2_test,
    # Distance-based
    run_mmd_test,
    run_mmd_multivar_test,
    run_mmd_rff_multivar,
    run_wasserstein_test,
    run_sliced_wasserstein_multivar,
    run_mahalanobis_test,
    # Classifier-based & autoencoder
    run_domain_classifier,
    run_c2st_logistic_classifier,
    run_c2st_forest_classifier,
    run_autoencoder_test,
    run_kl_test,
    run_js_test,
    # Visualization
    plot_histograms,
    plot_umap_2d,
    plot_pca_2d,
)

# -------------------------------------------------------------------
# Page Name and Logo
# -------------------------------------------------------------------
st.set_page_config(
    page_title="DomainSAT",   # what shows in the browser tab
    page_icon="Logo.png",
    layout="wide",            # optional, but nice for dashboards
)

# -------------------------------------------------------------------
# Title
# -------------------------------------------------------------------

# Convert local PNG to base64 string
def get_base64_image(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()

logo_base64 = get_base64_image("logo.png")

# Display logo + title horizontally
st.markdown(
    f"""
    <div style="display:flex; align-items:center; gap:12px; margin-bottom:5px;">
        <img src="data:image/png;base64,{logo_base64}" width="40">
        <h1 style="font-size:28px; margin:0;">DomainSAT: Domain Shift Analysis Toolbox</h1>
    </div>
    """,
    unsafe_allow_html=True
)


# -------------------------------------------------------------------
# Sidebar: Upload files
# -------------------------------------------------------------------
#st.sidebar.header("Upload Data Files (.csv)")
# Sidebar big section title: Upload
st.sidebar.markdown("""
### 📁 <span style="font-size:18px;">Upload CSV Files</span>
""", unsafe_allow_html=True)

src_file = st.sidebar.file_uploader("📤 Upload Source Dataset", type="csv")
tgt_file = st.sidebar.file_uploader("📤 Upload Target Dataset", type="csv")

# Note: if either uploader is empty, clear cached data/results
if not src_file or not tgt_file:
    for key in ("src_df", "tgt_df", "results_df", "umap_fig", "pca_fig"):
        st.session_state.pop(key, None)



# Separation line
st.sidebar.markdown("<hr style='margin:10px 0;'>", unsafe_allow_html=True)

# -------------------------------------------------------------------
# Sidebar: Method selection
# -------------------------------------------------------------------
# Sidebar big section title: Method
st.sidebar.markdown("""
### <span style="font-size:18px;">Shift Detection Method</span>
""", unsafe_allow_html=True)



method = st.sidebar.selectbox(
    "⚙️ Select a method",
    [
        # Univariate / per-feature tests
        "KS Test",
        "Mann-Whitney U",
        "Cramer-von Mises",
        "Chi-square Test",
        "MMD (Per-Feature)",
        "Wasserstein (Per-Feature)",
        "KL Divergence",
        "JS Divergence",

        # Multivariate / global distance-based
        "MMD (Multivariate)",
        "Mahalanobis Distance",
        "Wasserstein (Multivariate)",

        # Classifier-based
        "Domain Classifier",
        "C2ST (Logistic Regression)",
        "C2ST (Random Forest)",

        # Representation-based
        "Autoencoder",
    ],
)


# Note: clear results whenever the user changes method
if "last_method" not in st.session_state:
    st.session_state["last_method"] = method
elif method != st.session_state["last_method"]:
    st.session_state["last_method"] = method
    st.session_state["results_df"] = None

# -------------------------------------------------------------------
# Sidebar: Thresholds
# -------------------------------------------------------------------
p_thresh = st.sidebar.slider("P-value threshold", 0.001, 0.1, 0.05)
dist_thresh = st.sidebar.slider(
    "Distance / divergence / reconstruction threshold",
    0.001,
    1.0,
    0.05,
)
auc_thresh = st.sidebar.slider(
    "AUC threshold (for classifier-based methods)",
    0.50,
    1.0,
    0.60,
)

# Reset button
if st.sidebar.button("🔄 Reset App"):
    st.session_state.clear()
    st.rerun()

# -------------------------------------------------------------------
# Initialize session state
# -------------------------------------------------------------------
if "results_df" not in st.session_state:
    st.session_state["results_df"] = None
if "src_df" not in st.session_state:
    st.session_state["src_df"] = None
if "tgt_df" not in st.session_state:
    st.session_state["tgt_df"] = None
    
# Note: cache for UMAP figure
if "umap_fig" not in st.session_state:
    st.session_state["umap_fig"] = None

# Note: cache for PCA figure
if "pca_fig" not in st.session_state:
    st.session_state["pca_fig"] = None
    
    
# -------------------------------------------------------------------
# Load CSVs
# -------------------------------------------------------------------

if src_file and tgt_file:
    st.session_state["src_df"] = pd.read_csv(src_file)
    st.session_state["tgt_df"] = pd.read_csv(tgt_file)
    
    # Note: new data → invalidate UMAP and PCA cache
    st.session_state["umap_fig"] = None
    st.session_state["pca_fig"] = None
    
    st.success("✅ Datasets uploaded successfully!")
    
    

    if st.button("Run Shift Detection"):
        # Copy original DataFrame to avoid modifying session state directly
        src_df = st.session_state["src_df"].copy()
        tgt_df = st.session_state["tgt_df"].copy()

        # ------------------------------
        # Route to the appropriate method
        # ------------------------------
        if method == "KS Test":
            st.session_state["results_df"] = run_ks_test(src_df, tgt_df, p_thresh)

        elif method == "Mann-Whitney U":
            st.session_state["results_df"] = run_mannwhitney_test(src_df, tgt_df, p_thresh)

        elif method == "Chi-square Test":
            st.session_state["results_df"] = run_chi2_test(src_df, tgt_df, p_thresh)

        elif method == "Cramer-von Mises":
            st.session_state["results_df"] = run_cramervonmises_test(src_df, tgt_df, p_thresh)

        elif method == "MMD (Per-Feature)":
            st.session_state["results_df"] = run_mmd_test(src_df, tgt_df, mmd_thresh=dist_thresh)

        elif method == "MMD (Multivariate)":
            st.session_state["results_df"] = run_mmd_rff_multivar(
                src_df,
                tgt_df,
                dist_thresh=dist_thresh,
            )

        elif method == "Wasserstein (Per-Feature)":
            st.session_state["results_df"] = run_wasserstein_test(
                src_df,
                tgt_df,
                dist_thresh=dist_thresh,
            )
            
        elif method == "Wasserstein (Multivariate)":
            st.session_state["results_df"] = run_sliced_wasserstein_multivar(
                src_df,
                tgt_df,
                dist_thresh=dist_thresh,
            )

        elif method == "KL Divergence":
            st.session_state["results_df"] = run_kl_test(
                src_df,
                tgt_df,
                kl_thresh=dist_thresh,
            )

        elif method == "JS Divergence":
            st.session_state["results_df"] = run_js_test(
                src_df,
                tgt_df,
                js_thresh=dist_thresh,
            )

        elif method == "Mahalanobis Distance":
            st.session_state["results_df"] = run_mahalanobis_test(
                src_df,
                tgt_df,
                dist_thresh=dist_thresh,
            )

        elif method == "Domain Classifier":
            st.session_state["results_df"] = run_domain_classifier(
                src_df,
                tgt_df,
                auc_thresh=auc_thresh,
            )

        elif method == "C2ST (Logistic Regression)":
            st.session_state["results_df"] = run_c2st_logistic_classifier(
                src_df,
                tgt_df,
                auc_thresh=auc_thresh,
            )

        elif method == "C2ST (Random Forest)":
            st.session_state["results_df"] = run_c2st_forest_classifier(
                src_df,
                tgt_df,
                auc_thresh=auc_thresh,
            )


        elif method == "Autoencoder":
            st.session_state["results_df"] = run_autoencoder_test(
                src_df,
                tgt_df,
                recon_thresh=dist_thresh,
            )

# -------------------------------------------------------------------
# Display results + visualizations
# -------------------------------------------------------------------
if st.session_state["results_df"] is not None:
    st.subheader(f"🔎 Shift Analysis Results ({method})")
    # Some global methods have Feature="All" only; sort_values will still work
    st.dataframe(
        st.session_state["results_df"]
        .sort_values("Feature")
        .reset_index(drop=True)
    )

    st.download_button(
        "📥 Download Results as CSV",
        st.session_state["results_df"].to_csv(index=False),
        file_name = f"shift_results_{method}.csv",
    )

    # -------------------------------------------------------------------
    # Per-feature histogram + KDE
    # -------------------------------------------------------------------
    features = st.session_state["results_df"]["Feature"].unique().tolist()
    if features and not (len(features) == 1 and features[0] == "All"):
        selected_feature = st.selectbox("📊 Visualize Feature Histogram", features)
        fig_hist = plot_histograms(
            st.session_state["src_df"][selected_feature],
            st.session_state["tgt_df"][selected_feature],
            selected_feature,
        )
        st.pyplot(fig_hist)

    # -------------------------------------------------------------------
    # 2D embeddings (global view): UMAP + PCA
    # -------------------------------------------------------------------
    st.markdown("---")

    src_df = st.session_state.get("src_df")
    tgt_df = st.session_state.get("tgt_df")

    if src_df is not None and tgt_df is not None:
        col1, col2 = st.columns(2)

        # ---- UMAP ----
        with col1:
            if st.checkbox("📈 Show 2D UMAP embedding", value=False):
                # Only compute UMAP once per data change
                if st.session_state.get("umap_fig") is None:
                    with st.spinner("Running… (maybe a bit slow)"):
                        st.session_state["umap_fig"] = plot_umap_2d(src_df, tgt_df)

                st.pyplot(st.session_state["umap_fig"])

        # ---- PCA ----
        with col2:
            if st.checkbox("📈 Show 2D PCA embedding", value=False):
                # Only compute PCA once per data change
                if st.session_state.get("pca_fig") is None:
                    with st.spinner("Running…"):
                        st.session_state["pca_fig"] = plot_pca_2d(src_df, tgt_df)

                st.pyplot(st.session_state["pca_fig"])


