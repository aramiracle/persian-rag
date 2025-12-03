import os
import time
import requests
import pandas as pd
import gradio as gr
from loguru import logger

# --- CONFIGURATION ---
API_BASE_URL = os.getenv("BACKEND_URL", "http://backend:8000/api/v1")
logger.info(f"🚀 Gradio connecting to Backend at: {API_BASE_URL}")

# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================

def get_system_metrics():
    try:
        # 1. Fetch Data
        try:
            h_resp = requests.get(f"{API_BASE_URL}/system/health", timeout=3).json()
        except: h_resp = {}
        
        try:
            s_resp = requests.get(f"{API_BASE_URL}/system/stats", timeout=3).json()
        except: s_resp = {}

        # 2. Parse Components
        components = h_resp.get("components", {})
        
        # Milvus
        m_info = components.get("milvus", {})
        m_status = m_info.get("status", "unknown").lower()
        
        # Elasticsearch (Replaces BM25)
        es_info = components.get("elasticsearch", {})
        es_count = es_info.get("total_documents", 0)
        es_status = es_info.get("status", "unknown").lower() # green/yellow/red

        # Stats (With Fallbacks)
        total_docs = s_resp.get("total_documents", es_count) 
        
        # 3. Icons & Status Logic
        # API
        api_status = h_resp.get("status", "unknown").upper()
        # FIX: Ensure we match Uppercase API status correctly
        icon_api = "✅" if api_status in ["HEALTHY", "UP", "DEGRADED"] else "⚠️"
        
        # Vector DB
        icon_db = "🟢" if m_status in ["healthy", "up", "connected"] else "🔴"
        
        # Search Engine (ES) - Yellow is OK for single node
        icon_es = "🟢" if es_status in ["green", "yellow"] else "🔴"

        # 4. Render
        return (
            f"### 🖥️ System Dashboard\n"
            f"| Metric | Status | Details |\n"
            f"| :--- | :---: | :--- |\n"
            f"| **API Health** | {icon_api} | `{api_status}` |\n"
            f"| **Milvus (Vec)** | {icon_db} | `{m_status.upper()}` |\n"
            f"| **Elastic (Text)** | {icon_es} | `{es_status.upper()}` |\n"
            f"| **Indexed Docs** | 📚 | **{total_docs:,}** |\n"
        )
    except Exception as e:
        return f"❌ Error: {e}"

def ask_question(question, top_k, min_score=0.0):
    """
    Used ONLY for the Chat/Q&A Tab.
    Returns: (Answer_String, Sources_String, Raw_JSON_Dict)
    """
    if not question or not question.strip():
        return "⚠️ Please enter a question.", "", {}
        
    try:
        payload = {
            "question": question, 
            "top_k": top_k, 
            "min_score": min_score
        }
        
        resp = requests.post(f"{API_BASE_URL}/rag/ask", json=payload, timeout=120)
        resp.raise_for_status()
        data = resp.json()
        
        answer = data.get("answer", "No answer generated.")
        stats = (
            f"**⏱️ Performance:**\n"
            f"- Retrieval: {data.get('retrieval_time_ms', 0):.0f} ms\n"
            f"- Generation: {data.get('generation_time_ms', 0):.0f} ms\n"
            f"- Total: {data.get('total_time_ms', 0):.0f} ms\n"
        )
        
        sources_list = []
        if data.get("sources"):
            for src in data["sources"]:
                doc = src.get("document", {}) or {}
                sources_list.append(
                    f"1. **[{src.get('score', 0):.4f}]** [{doc.get('title', 'No Title')}]({doc.get('url', '#')}) "
                    f"- *{doc.get('news_agency_name')}*"
                )
            sources_text = "\n".join(sources_list)
        else:
            sources_text = "No context used."
            
        return f"{answer}\n\n---\n{stats}", sources_text, data
        
    except Exception as e:
        return f"❌ **Error:** {str(e)}", "", {}

def search_wrapper(query, top_k, min_score):
    """
    Strict Search Function (No LLM).
    Hits /rag/search and formats the list of documents into HTML.
    """
    if not query or not query.strip():
        return "⚠️ Please enter a search term.", {}

    try:
        payload = {
            "query": query,
            "top_k": int(top_k),
            "min_score": float(min_score)
        }

        # Call the Search Endpoint
        resp = requests.post(f"{API_BASE_URL}/rag/search", json=payload, timeout=30)
        resp.raise_for_status()
        data = resp.json()

        # Parse Response
        results = data.get("results", [])
        total_found = data.get("total_found", 0)
        time_ms = data.get("retrieval_time_ms", 0)

        # Build HTML
        html_parts = [f"<div dir='rtl' style='font-family: sans-serif; color: #333;'>"]
        
        # Header
        html_parts.append(
            f"<div style='margin-bottom: 15px; padding-bottom: 10px; border-bottom: 1px solid #eee;'>"
            f"✅ <b>{total_found}</b> results found in <b>{time_ms:.0f} ms</b>"
            f"</div>"
        )

        if not results:
            html_parts.append("<p>No documents found matching your query.</p>")
        else:
            for item in results:
                # Extract Document Details
                doc = item.get("document", {})
                score = item.get("score", 0)
                rank = item.get("rank", 0)
                
                title = doc.get("title", "No Title")
                url = doc.get("url", "#")
                agency = doc.get("news_agency_name", "Unknown Agency")
                raw_date = doc.get("published_at", "")
                date_str = str(raw_date).replace("T", " ") if raw_date else "Unknown Date"
                
                # Snippet logic
                content = doc.get("content", "")
                snippet = content[:400] + "..." if len(content) > 400 else content

                # Result Card HTML
                card_html = (
                    f"<div style='margin-bottom: 25px; padding: 10px;'>"
                    f"   <div style='font-size: 18px; margin-bottom: 4px;'>"
                    f"       <a href='{url}' target='_blank' style='text-decoration: none; color: #1a0dab; font-weight: bold;'>"
                    f"       {rank}. {title}"
                    f"       </a>"
                    f"   </div>"
                    f"   <div style='font-size: 12px; color: #006621; margin-bottom: 6px;'>"
                    f"       {agency} | {date_str} | Score: {score:.4f}"
                    f"   </div>"
                    f"   <div style='font-size: 14px; line-height: 1.6; color: #4d5156;'>"
                    f"       {snippet}"
                    f"   </div>"
                    f"</div>"
                )
                html_parts.append(card_html)

        html_parts.append("</div>")
        
        return "".join(html_parts), data

    except Exception as e:
        return f"<div dir='rtl'>❌ <b>Error executing search:</b><br>{str(e)}</div>", {}

def get_document_preview(limit):
    try:
        resp = requests.get(f"{API_BASE_URL}/documents/preview", params={"limit": limit}, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        if not data:
            return pd.DataFrame(), "⚠️ Database is empty."
        df_rows = []
        for d in data:
            df_rows.append({
                "ID": d.get("id"),
                "Agency": d.get("news_agency_name"),
                "Title": d.get("title"),
                "Topic": d.get("topic"),
                "Date": str(d.get("published_at"))[:10],
            })
        return pd.DataFrame(df_rows), f"✅ Loaded {len(data)} documents."
    except Exception as e:
        return pd.DataFrame(), f"❌ Error: {str(e)}"

# ==============================================================================
# UPLOAD LOGIC
# ==============================================================================

def upload_csv_with_progress(file_path, progress=gr.Progress()):
    if not file_path:
        yield "⚠️ No file selected."
        return

    try:
        yield "### 🚀 Initiating Upload..."
        progress(0, desc="Uploading to server...")
        
        with open(file_path, "rb") as f:
            files = {"file": ("uploaded.csv", f, "text/csv")}
            try:
                response = requests.post(f"{API_BASE_URL}/upload/csv", files=files, timeout=300)
            except requests.exceptions.ConnectionError:
                yield f"❌ **Error:** Could not connect to `{API_BASE_URL}`"
                return

        if response.status_code not in [200, 202]:
            yield f"❌ **Upload Failed:** {response.text}"
            return

        job_data = response.json()
        job_id = job_data.get("job_id")
        
        yield f"### ⏳ Job Started: `{job_id}`"

        while True:
            try:
                status_res = requests.get(f"{API_BASE_URL}/upload/status/{job_id}", timeout=5)
                status_res.raise_for_status()
                status_data = status_res.json()
            except Exception:
                time.sleep(1)
                continue

            state = status_data.get("status", "unknown").lower()
            percent = status_data.get("progress", 0)
            msg = status_data.get("message", "Processing...")

            # Backend sends: "Processing chunk 4/25 12%"
            progress(percent / 100, desc=f"{msg}")

            if state == "completed":
                res = status_data.get("result", {})
                total_rows = res.get('total_processed', 0)
                success_count = res.get('success', 0)
                failed_count = res.get('failed', 0)
                time_ms = res.get('time_ms', 0)
                
                yield (
                    f"## ✅ Processing Complete\n"
                    f"**Job ID:** `{job_id}`\n\n"
                    f"| Metric | Count |\n"
                    f"| :--- | :--- |\n"
                    f"| **Total Rows** | {total_rows:,} |\n"
                    f"| **Successful** | {success_count:,} |\n"
                    f"| **Failed** | {failed_count:,} |\n"
                    f"| **Time Taken** | {time_ms/1000:.2f}s |\n"
                )
                return
            elif state == "failed":
                yield f"## ❌ Processing Failed\n**Reason:** {msg}"
                return
            
            time.sleep(0.5)

    except Exception as e:
        yield f"❌ **Exception:** {str(e)}"

# ==============================================================================
# CSS & LAYOUT
# ==============================================================================

custom_css = """
.output-box { height: auto !important; min-height: 100px !important; }
.rtl-input textarea { direction: rtl; text-align: right; }
.rtl-output { direction: rtl; text-align: right; }
"""

with gr.Blocks(theme=gr.themes.Soft(primary_hue="blue"), css=custom_css, title="Persian News Hybrid RAG") as demo:
    
    gr.Markdown(
        """
        # 🇮🇷 Persian News RAG (Scale Edition)
        ### Hybrid Search (Dense Vectors + Elasticsearch BM25+) with RRF Fusion
        """
    )
    
    with gr.Tabs():
        
        # --- SEARCH TAB ---
        with gr.Tab("🔍 Hybrid Search"):
            search_query = gr.Textbox(
                label="Search Query", placeholder="...جستجو در اخبار", lines=2,
                rtl=True, text_align="right", elem_classes=["rtl-input"]
            )
            with gr.Accordion("Search Settings", open=True):
                search_top_k = gr.Slider(1, 50, 10, step=1, label="Top K")
                search_min_score = gr.Slider(0.0, 1.0, 0.0, step=0.05, label="Min Fusion Score")
            
            search_btn = gr.Button("Search", variant="primary")
            
            # HTML component preserved
            search_output = gr.HTML(label="Results", elem_classes=["rtl-output"])
            search_json = gr.JSON(visible=False, label="Debug JSON")
            
            search_btn.click(
                search_wrapper, 
                inputs=[search_query, search_top_k, search_min_score], 
                outputs=[search_output, search_json]
            )

        # --- Q&A TAB ---
        with gr.Tab("🤖 RAG Chat"):
            question_input = gr.Textbox(
                label="Your Question", placeholder="...سوال خود را بپرسید", lines=3,
                rtl=True, text_align="right", elem_classes=["rtl-input"]
            )
            with gr.Accordion("Chat Settings", open=True):
                qa_top_k = gr.Slider(1, 20, 5, step=1, label="Context Docs (Top K)")
            
            qa_btn = gr.Button("Generate Answer", variant="primary")
            answer_output = gr.Markdown(label="LLM Answer", elem_classes=["rtl-output"])
            sources_output = gr.Markdown(label="Used Sources", elem_classes=["rtl-output"])
            qa_json = gr.JSON(visible=False, label="Debug JSON")
            
            qa_btn.click(
                ask_question, inputs=[question_input, qa_top_k], outputs=[answer_output, sources_output, qa_json]
            )

        # --- UPLOAD TAB ---
        with gr.Tab("📂 Upload Data"):
            gr.Markdown("### 📥 Upload CSV News Data (Stream Processing)")
            file_input = gr.File(label="Select CSV", file_types=[".csv"])
            upload_btn = gr.Button("Start Upload", variant="primary")
            
            with gr.Accordion("Upload Status & Progress", open=True):
                upload_output = gr.Markdown(elem_classes=["output-box"])
            
            upload_btn.click(
                upload_csv_with_progress, inputs=[file_input], outputs=[upload_output]
            )

        # --- DATABASE & STATS TAB ---
        with gr.Tab("📊 System & Data"):
            gr.Markdown("### ⚙️ System Status & Database Preview")
            
            with gr.Row():
                stats_output = gr.Markdown(value="Hit **Refresh** to load stats...")
            
            with gr.Row():
                refresh_btn = gr.Button("🔄 Refresh System Stats & Data", variant="primary")
                preview_limit = gr.Slider(5, 100, 10, step=5, label="Preview Rows")

            preview_status = gr.Markdown("")
            preview_table = gr.Dataframe(
                headers=["ID", "Agency", "Title", "Topic", "Date"], 
                interactive=False,
                label="Document Preview"
            )
            
            refresh_btn.click(
                get_system_metrics, inputs=[], outputs=[stats_output]
            ).then(
                get_document_preview, inputs=[preview_limit], outputs=[preview_table, preview_status]
            )

if __name__ == "__main__":
    demo.queue()
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
