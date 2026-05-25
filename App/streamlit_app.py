"""
main.py
-------
Entry point cá»§a Streamlit app.
Cháº¡y báº±ng:  streamlit run App/streamlit_app.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Äáº£m báº£o nháº­n diá»‡n Ä‘Ãºng cÃ¡c module trong thÆ° má»¥c app
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from datetime import datetime
import streamlit as st

from Pipeline_mamba import train_pipeline

from Ui_components import (
    render_data_preview,
    render_forecast_download,
    render_location_selector,
    render_mamba_results,
    render_sidebar,
    render_train_config,
)


def main() -> None:
    st.set_page_config(page_title="AQI Mamba", layout="wide")
    st.title("ðŸš€ AQI Forecasting: Mamba")
    st.caption("Huáº¥n luyá»‡n vÃ  dá»± bÃ¡o cháº¥t lÆ°á»£ng khÃ´ng khÃ­ vá»›i Mamba.")

    # --- 1. Sidebar & Load Dataset ---
    render_sidebar()

    if "df" not in st.session_state:
        st.info("ðŸ‘ˆ Vui lÃ²ng load dataset tá»« sidebar Ä‘á»ƒ báº¯t Ä‘áº§u.")
        return

    df = st.session_state["df"]

    # --- 2. Cáº¥u hÃ¬nh chung vÃ  chá»n Ä‘á»‹a Ä‘iá»ƒm ---
    selected_locations = render_location_selector(df)
    
    train_cfg = render_train_config()

    # Preview dá»¯ liá»‡u (Ä‘Ã£ bá»c try-except bÃªn trong component)
    render_data_preview(df)

    # --- 3. Khá»Ÿi táº¡o cÃ¡c biáº¿n chá»©a káº¿t quáº£ (TrÃ¡nh lá»—i NameError) ---
    summary, hist_df, future_df = None, None, None

    # Táº¡o Ä‘Æ°á»ng dáº«n lÆ°u káº¿t quáº£ dá»±a trÃªn thá»i gian
    project_root = Path(__file__).resolve().parent.parent
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_run_dir = str(project_root / "runs" / timestamp)

    # --- 4. VÃ²ng láº·p huáº¥n luyá»‡n khi nháº¥n nÃºt ---
    if st.button("ðŸ”¥ Run Training", use_container_width=True):
        if not selected_locations:
            st.warning("âš ï¸ Vui lÃ²ng chá»n Ã­t nháº¥t má»™t Ä‘á»‹a Ä‘iá»ƒm.")
            st.stop()

        feature_cols = list(train_cfg["feature_cols"])
        if train_cfg["target_col"] not in feature_cols:
            feature_cols.append(train_cfg["target_col"])

        if not feature_cols:
            st.warning("âš ï¸ Vui lÃ²ng chá»n Ã­t nháº¥t má»™t cá»™t feature.")
            st.stop()

        status_placeholder = st.empty()

        try:
            status_placeholder.info("â³ Äang huáº¥n luyá»‡n Mamba...")
            mamba_run_dir = os.path.join(base_run_dir, "mamba")
            summary, hist_df, future_df = train_pipeline(
                df=df,
                forecast_base_df=None,
                selected_locations=selected_locations,
                target_col=train_cfg["target_col"],
                feature_cols=feature_cols,
                window_size=train_cfg["window_size"],
                horizon=train_cfg["horizon"],
                sample_stride=train_cfg["sample_stride"],
                epochs=train_cfg["epochs"],
                batch_size=train_cfg["batch_size"],
                lr=train_cfg["lr"],
                weight_decay=train_cfg["weight_decay"],
                d_model=train_cfg["d_model"],
                n_layers=train_cfg["n_layers"],
                loss_name=train_cfg["loss_name"],
                seed=train_cfg["seed"],
                num_workers=4,
                use_gpu=train_cfg["use_gpu"],
                log_interval=50,
                grad_accum_steps=train_cfg["grad_accum_steps"],
                max_grad_norm=train_cfg["max_grad_norm"],
                early_stop_patience=train_cfg["early_stop_patience"],
                early_stop_min_delta=0.0,
                run_dir=mamba_run_dir,
            )

            status_placeholder.success("âœ… HoÃ n thÃ nh huáº¥n luyá»‡n Mamba!")

        except Exception as e:
            st.error(f"âŒ QuÃ¡ trÃ¬nh huáº¥n luyá»‡n gáº·p lá»—i: {e}")
            import traceback
            traceback.print_exc()
            return

    # --- 5. Hiá»ƒn thá»‹ káº¿t quáº£ (Render Results) ---
    # Chá»‰ hiá»ƒn thá»‹ náº¿u biáº¿n káº¿t quáº£ khÃ´ng pháº£i None (nghÄ©a lÃ  Ä‘Ã£ Ä‘Æ°á»£c cháº¡y thÃ nh cÃ´ng)
    
    # Káº¿t quáº£ Mamba
    if summary is not None:
        render_mamba_results(summary, hist_df, future_df, df, selected_locations)
        # NÃºt táº£i dá»± bÃ¡o cho Mamba (náº¿u cÃ³)
        if future_df is not None:
            render_forecast_download(future_df, summary)

    # --- 6. Káº¿t thÃºc render ---


if __name__ == "__main__":
    main()

