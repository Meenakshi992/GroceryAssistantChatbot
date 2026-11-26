import pandas as pd
import streamlit as st
from vector_store import index_uploaded_file


def render_sidebar(client, collection, refresh_callback):
    with st.sidebar:
        st.subheader("1. Upload Grocery Files (Min 4)")
        if st.session_state["knowledge_base"]:
            st.success(f"Loaded {len(st.session_state['total_documents'])} saved files from ChromaDB.")

        st.markdown("Need a Surrey flyer? Check **[Flipp.com](https://flipp.com)** and upload the PDF or screenshot below.")

        uploaded_files = st.file_uploader(
            "Select PDF, TXT, MD files",
            type=["pdf", "txt", "md"],
            accept_multiple_files=True,
        )

        st.warning(
            "Note: Please upload text-based PDF, TXT, MD files. "
            "Scanned or image-only files without selectable text cannot be read and will not be accepted."
        )

        with st.expander("Text Cleaning Options", expanded=True):
            cleaning_options = {
                "fix_hyphens": st.checkbox("Fix hyphenated line breaks", value=True),
                "remove_page_nums": st.checkbox("Remove page numbers", value=True),
                "fix_lines": st.checkbox("Join broken lines", value=True),
                "clean_spaces": st.checkbox("Remove extra spaces", value=True),
            }

        with st.expander("Chunking & Search Settings", expanded=False):
            chunk_size = st.slider("Chunk Size", 300, 1200, 700, step=100)
            overlap = st.slider("Chunk Overlap", 0, 300, 120, step=30)
            top_k = st.slider("Top-K Chunks", 1, 5, 4)
            threshold = st.slider("Similarity Cutoff", 0.10, 0.80, 0.35, step=0.05)

        # getting total sum of files - saved in Chroma DB and newly uploaded ones
        saved_names = set()
        for row in st.session_state["total_documents"]:
            saved_names.add(row["File Name"])

        new_names = set()
        if uploaded_files:
            for f in uploaded_files:
                new_names.add(f.name)

        total_docs = len(saved_names.union(new_names))

        if uploaded_files:
            if total_docs < 4:
                st.warning(f"Please upload at least 4 documents ({total_docs}/4 selected).")
            elif st.button("Index Uploaded Documents", type="primary", use_container_width=True):
                total_added = 0
                for file_obj in uploaded_files:
                    try:
                        with st.spinner(f"Processing {file_obj.name}..."):
                            res = index_uploaded_file(client, collection, file_obj, chunk_size, overlap, cleaning_options)
                        if res["chunks_added"] == 0:
                            st.error(f"Skipped '{file_obj.name}': No readable item titles or prices found.")
                        else:
                            total_added += res["chunks_added"]
                            txt_name = file_obj.name.rsplit(".", 1).pop(0) + "_extracted.txt"
                            st.session_state["extracted_files"][txt_name] = res["extracted_text"]
                    except Exception as err:
                        st.error(f"Error reading {file_obj.name}: {err}")

                refresh_callback()
                if total_added > 0:
                    st.success(f"Saved {total_added} chunks!")
                    st.rerun()

        if len(st.session_state["extracted_files"]) > 0:
            with st.expander("Download Extracted Flyer Text (.txt)", expanded=True):
                for txt_name, txt_body in st.session_state["extracted_files"].items():
                    st.download_button(
                        label=f"Download {txt_name}",
                        data=txt_body,
                        file_name=txt_name,
                        mime="text/plain",
                        use_container_width=True,
                    )

        st.divider()
        st.subheader("2. Nearby Surrey Stores")
        pincode = st.text_input("Surrey Postal Code", value="V3T 2W1")
        st.warning("e.g., V3R 0A4 vs. V3T 2W1 vs. V3W 2W8")
        max_radius = st.slider("Radius (km)", 1.0, 20.0, 6.0, step=0.5)
        st.info("Active Flyer Week: Nov 26 – Dec 3, 2026 (Verify printed dates on each flyer).")
        st.warning("Default Stores listed under 6 KMs, you can change to 8.0 or upto 20.0 km-")

        st.divider()
        st.subheader("3. Knowledge Base Status")
        c1, c2, c3 = st.columns(3)
        c1.metric("Files", len(st.session_state["total_documents"]))
        c2.metric("Chunks", st.session_state["chunk_count"])
        c3.metric("Cached", len(st.session_state["embedding_cache"]))

        if len(st.session_state["total_documents"]) > 0:
            st.dataframe(pd.DataFrame(st.session_state["total_documents"]), use_container_width=True, hide_index=True)
            if st.button("Reset Database", use_container_width=True):
                all_ids = collection.get().get("ids") or []
                if len(all_ids) > 0:
                    collection.delete(ids=all_ids)
                st.session_state["messages"] = []
                st.session_state["embedding_cache"] = {}
                st.session_state["extracted_files"] = {}
                refresh_callback()
                st.rerun()

    return {"top_k": top_k, "threshold": threshold, "pincode": pincode, "max_radius": max_radius}
