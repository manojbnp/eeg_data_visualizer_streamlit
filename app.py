import streamlit as st
import os
import mne
import pandas as pd
import matplotlib.pyplot as plt
import tempfile
import io

# Configure Streamlit page
st.set_page_config(
    page_title="SP EEG Data Cleaner",
    layout="centered",
    initial_sidebar_state="auto"
)

st.title("Hallo SP Project Members 👋")
st.markdown("Upload one or more **.edf** files and get the cleaned, combined **.csv** and **.xlsx** files, along with visualizations.")

# --- Session State Initialization ---
if 'summary_df' not in st.session_state:
    st.session_state.summary_df = None
if 'edf_data_storage' not in st.session_state:
    st.session_state.edf_data_storage = {}
if 'event_data_storage' not in st.session_state:
    st.session_state.event_data_storage = {}
if 'channel_list' not in st.session_state:
    st.session_state.channel_list = []
if 'edf_files_uploaded' not in st.session_state:
    st.session_state.edf_files_uploaded = []
if 'cleaned_csv_buffer' not in st.session_state:
    st.session_state.cleaned_csv_buffer = None
if 'cleaned_excel_buffer' not in st.session_state:
    st.session_state.cleaned_excel_buffer = None

# --- Functions for plotting ---
def create_event_plot(edf_df, event_df, channel, window=2.0):
    fig, ax = plt.subplots(figsize=(15, 6))
    for _, row in event_df.iterrows():
        onset = row['onset']
        description = row['description']
        start_time = onset - window / 2
        end_time = onset + window / 2
        if 'time' in edf_df.columns:
            segment = edf_df[(edf_df['time'] >= start_time) & (edf_df['time'] <= end_time)]
            if not segment.empty and channel in segment.columns:
                ax.plot(segment['time'], segment[channel], label=f'{description} at {onset:.2f}s')
        else:
            st.warning(f"No 'time' column found in EDF data for {edf_df.get('file', 'unknown file')}. Cannot plot.")
            return None

    ax.set_title(f"EEG Channel '{channel}' Around Events")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("EEG Amplitude (µV)")
    ax.legend()
    ax.grid(True)
    plt.tight_layout(pad=3.0)
    return fig

def create_combined_plot(edf_df, event_df, channel):
    if 'time' not in edf_df.columns:
        st.warning(f"No 'time' column found in EDF data for {edf_df.get('file', 'unknown file')}. Cannot create combined plot.")
        return None

    t0 = event_df[event_df['description'] == 'T0']
    t1 = event_df[event_df['description'] == 'T1']
    t2 = event_df[event_df['description'] == 'T2']

    max_rows = max(len(t0), len(t1), len(t2)) + 1
    fig, axes = plt.subplots(max_rows, 3, figsize=(15, max_rows * 3), sharey=True)

    labels = ['T0', 'T1', 'T2']
    colors = ['blue', 'green', 'red']
    events = [t0, t1, t2]

    if max_rows == 1:
        axes = [axes] # Ensure axes is iterable even for 1 row

    for col, (label, ev, color) in enumerate(zip(labels, events, colors)):
        ax = axes[0, col] if max_rows > 1 else axes[col]
        for _, row in ev.iterrows():
            segment = edf_df[(edf_df['time'] >= row['onset']) & (edf_df['time'] <= row['onset'] + row['duration'])]
            if not segment.empty and channel in segment.columns:
                ax.plot(segment['time'], segment[channel], alpha=0.5, color=color)
        ax.set_title(f"All {label} Events - {channel}")
        ax.grid(True)

    if max_rows > 1:
        for i in range(1, max_rows):
            for col, ev in enumerate(events):
                if i - 1 < len(ev):
                    row = ev.iloc[i - 1]
                    segment = edf_df[(edf_df['time'] >= row['onset']) & (edf_df['time'] <= row['onset'] + row['duration'])]
                    ax_individual = axes[i, col]
                    if not segment.empty and channel in segment.columns:
                        ax_individual.plot(segment['time'], segment[channel], color=colors[col])
                    start = row['onset']
                    end = row['onset'] + row['duration']
                    ax_individual.set_title(f"{labels[col]} Event {i} | {start:.1f}s–{end:.1f}s", fontsize=9)
                    ax_individual.grid(True)
                else:
                    axes[i, col].axis("off")

    plt.tight_layout(pad=2.5)
    plt.subplots_adjust(hspace=0.4)
    return fig

# --- Main Application Logic (Input Form) ---
with st.form("upload_form", clear_on_submit=True):
    subject_name = st.text_input("Enter Subject Name:", "subject")
    uploaded_files = st.file_uploader("Select EDF files:", type=["edf"], accept_multiple_files=True)
    process_button = st.form_submit_button("Process Files")

if process_button:
    if not subject_name.strip():
        st.error("Please enter a subject/folder name.")
    elif not uploaded_files:
        st.error("Please upload at least one .edf file.")
    else:
        st.info("⏳ Please wait... Processing the EDF files. This may take a few moments.")
        
        # Clear previous session data for new processing run
        st.session_state.summary_df = None # Clear this early for new processing
        st.session_state.edf_data_storage = {}
        st.session_state.event_data_storage = {}
        st.session_state.channel_list = []
        st.session_state.edf_files_uploaded = []
        st.session_state.cleaned_csv_buffer = None
        st.session_state.cleaned_excel_buffer = None

        matched_data_list = []
        summary_data = []
        non_edf_uploaded = False

        progress_text = "Operation in progress. Please wait."
        my_bar = st.progress(0, text=progress_text)
        
        try:
            for i, uploaded_file in enumerate(uploaded_files):
                filename = uploaded_file.name
                
                if not filename.endswith(".edf"):
                    non_edf_uploaded = True
                    my_bar.progress((i + 1) / len(uploaded_files), text=f"Skipping non-EDF file: {filename}")
                    continue

                file_content = uploaded_file.read()
                
                my_bar.progress((i + 0.1) / len(uploaded_files), text=f"Processing {filename}...")

                try:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".edf") as tmp:
                        tmp.write(file_content)
                        edf_path = tmp.name

                    raw = mne.io.read_raw_edf(edf_path, preload=True, stim_channel=None, verbose=False)
                    edf_df = raw.to_data_frame()
                    edf_df["file"] = filename

                    annotations = raw.annotations
                    event_df = pd.DataFrame({
                        "onset": annotations.onset,
                        "duration": annotations.duration,
                        "description": annotations.description
                    })
                    event_df["file"] = filename

                    st.session_state.edf_data_storage[filename] = edf_df
                    st.session_state.event_data_storage[filename + ".event"] = event_df
                    st.session_state.edf_files_uploaded.append(filename)

                except Exception as e:
                    st.error(f"Unable to process EDF file '{filename}': {e}")
                    if 'edf_path' in locals() and os.path.exists(edf_path):
                        os.unlink(edf_path)
                    continue

                if 'edf_path' in locals() and os.path.exists(edf_path):
                    os.unlink(edf_path)
                
                my_bar.progress((i + 0.5) / len(uploaded_files), text=f"Matching events for {filename}...")

                event_file = filename + ".event"
                if event_file in st.session_state.event_data_storage:
                    edf_df_current = st.session_state.edf_data_storage[filename]
                    event_df_current = st.session_state.event_data_storage[event_file]

                    edf_df_matched = edf_df_current.copy()
                    edf_df_matched["event_description"] = None

                    for _, event in event_df_current.iterrows():
                        onset = event["onset"]
                        duration = event["duration"]
                        description = event["description"]
                        if 'time' in edf_df_matched.columns:
                            in_range = (edf_df_matched["time"] >= onset) & (edf_df_matched["time"] < (onset + duration))
                            edf_df_matched.loc[in_range, "event_description"] = description
                        else:
                            st.warning(f"Missing 'time' column in {filename} for event matching.")

                    matched_data_list.append(edf_df_matched)

                    signal_columns = edf_df_current.columns.difference(['time', 'file', 'event_description'])
                    zero_mask = (edf_df_current[signal_columns] == 0).all(axis=1)
                    zero_count = zero_mask.sum()

                    summary_data.append({
                        "edf_file": filename,
                        "edf_rows": edf_df_current.shape[0],
                        "edf_cols": edf_df_current.shape[1],
                        "edf_zero_rows": zero_count,
                        "event_file": event_file,
                        "event_rows": event_df_current.shape[0],
                        "event_cols": event_df_current.shape[1]
                    })
                
                my_bar.progress((i + 1) / len(uploaded_files), text=f"Finished processing {filename}")

            my_bar.empty()

            if not matched_data_list:
                st.error("No valid .edf files with annotations were processed.")
            else:
                combined_df = pd.concat(matched_data_list, ignore_index=True)
                cleaned_df = combined_df.dropna(subset=["event_description"])

                st.session_state.channel_list = list(combined_df.columns.difference(['time', 'file', 'event_description']))

                csv_buffer = io.StringIO()
                cleaned_df.to_csv(csv_buffer, index=False)
                st.session_state.cleaned_csv_buffer = csv_buffer.getvalue().encode('utf-8')

                excel_buffer = io.BytesIO()
                cleaned_df.to_excel(excel_buffer, index=False)
                st.session_state.cleaned_excel_buffer = excel_buffer.getvalue()

                st.session_state.summary_df = pd.DataFrame(summary_data)
                
                if non_edf_uploaded:
                    st.info("Only .edf files were processed. Others were ignored.")
                st.success("Files processed successfully!")

        except Exception as e:
            st.error(f"An unexpected error occurred during processing: {e}")
            my_bar.empty()

# --- Display Results and Download Links (ALWAYS DISPLAYED IF DATA EXISTS) ---
# THESE BLOCKS ARE DE-INDENTED TO THE MAIN SCRIPT LEVEL
if st.session_state.summary_df is not None:
    st.subheader("Summary of Processed Data")
    st.dataframe(st.session_state.summary_df)

    st.subheader("Download Cleaned Files")
    col1, col2 = st.columns(2)
    with col1:
        if st.session_state.cleaned_csv_buffer:
            st.download_button(
                label="Download CSV",
                data=st.session_state.cleaned_csv_buffer,
                file_name=f"{subject_name}_cleaned_data.csv",
                mime="text/csv",
            )
    with col2:
        if st.session_state.cleaned_excel_buffer:
            st.download_button(
                label="Download Excel",
                data=st.session_state.cleaned_excel_buffer,
                file_name=f"{subject_name}_cleaned_data.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )

    st.subheader("Visualize EEG Data")
    
    if st.session_state.edf_files_uploaded and st.session_state.channel_list:
        selected_file = st.selectbox("Select EDF File for Visualization:", st.session_state.edf_files_uploaded)
        selected_channel = st.selectbox("Select Channel for Visualization:", st.session_state.channel_list)

        if st.button("Show Graphs"):
            with st.spinner("Generating plots..."):
                edf_df_plot = st.session_state.edf_data_storage.get(selected_file)
                event_df_plot = st.session_state.event_data_storage.get(selected_file + ".event")

                if edf_df_plot is not None and event_df_plot is not None:
                    plot1_fig = create_event_plot(edf_df_plot, event_df_plot, selected_channel)
                    if plot1_fig:
                        st.write("### Plot: EEG Events Around Selected Channel")
                        st.pyplot(plot1_fig)
                    plt.close(plot1_fig)

                    plot2_fig = create_combined_plot(edf_df_plot, event_df_plot, selected_channel)
                    if plot2_fig:
                        st.write("### Combined T0, T1, T2 Visualization")
                        st.pyplot(plot2_fig, use_container_width=True)
                    plt.close(plot2_fig)
                else:
                    st.warning("Data not found for selected file/channel. Please re-process or select valid options.")
    else:
        st.info("Upload and process files to enable visualization options.")

st.markdown("---")
st.markdown("If you experience 'out of memory' issues with very large files, consider using smaller files or contact the site owner.")