import streamlit as st
import chromadb
from transformers import pipeline
from pathlib import Path
import tempfile
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.backend.docling_parse_v2_backend import DoclingParseV2DocumentBackend
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions, AcceleratorOptions, AcceleratorDevice
from datetime import datetime

def add_custom_css():
    st.markdown("""
    <!-- Load Google Fonts -->
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Bitcount+Grid+Double:wght@355&display=swap" rel="stylesheet">
    <link href="https://fonts.googleapis.com/css2?family=Cal+Sans&display=swap" rel="stylesheet">
    <link href="https://fonts.googleapis.com/css2?family=ZCOOL+QingKe+HuangYou&display=swap" rel="stylesheet">

    <style>
    /* --- Base styling --- */
    html, body, .stApp {
        min-height: 100vh;
        background: linear-gradient(rgba(0,0,0,0.5), rgba(0,0,0,0.5)),
        url("https://cdn.pixabay.com/photo/2021/06/04/06/54/racket-6308994_1280.jpg") no-repeat center center fixed;
        background-size: cover;
        font-family: 'Cal Sans', sans-serif !important;
        font-weight: 400;
        font-style: normal;
        position: relative;
        color: white;
    }

    /* --- Title Styling --- */
    h1.custom-title {
        font-family: "Bitcount Grid Double", system-ui, sans-serif !important;
        font-weight: 355;
        text-align: center;
        font-size: 80px;
        font-variation-settings: "slnt" 0, "CRSV" 0.2, "ELSH" 0, "ELXP" 0;
        margin: 0;
        padding: 0;
    }

    p.custom-subtitle {
        font-family: "Cal Sans", sans-serif;
        font-size: 18px;
        color: white;
        text-align: center;
        font-weight: 400;
        margin: 0;
        padding: 0;
        margin-top: -10px; /* pull closer to title */
        text-transform: uppercase;
        letter-spacing: 1.1px;
    }
               

    /* --- Intro paragraph styling --- */
    p.intro-text {
        font-family: "Cal Sans", sans-serif;
        font-size: 10px;
        color: white;
        text-align: right;
        margin-top: 1rem;
        margin-bottom: 2rem;
    }

    /* --- Button styling --- */
    .stButton > button {
        width: 100%;
        border-radius: 8px;
        height: 3rem;
        font-weight: bold;
        background: linear-gradient(90deg, #C1440E, #1a1a1a );
        color: white;
        border: none;
        font-size: 1.1rem;
        box-shadow: 0 1px 4px rgba(216, 27, 96, 0.08);
        transition: background 0.2s;
        font-family: 'Cal Sans', sans-serif !important;
    }
    .stButton > button:hover {
        background: linear-gradient(90deg, #C1440E, #1a1a1a );
        color: #C1440E;
    }

    /* --- Tabs styling --- */
    .stTabs [data-baseweb="tab-list"] {
        justify-content: center;
        margin-bottom: 1.5rem;
    }
    .stTabs [data-baseweb="tab"] {
        font-size: 1.1rem;
        font-weight: 600;
        color: #FFFFFF;
        background: #C1440E;
        border-radius: 8px 8px 0 0;
        margin-right: 0.5rem;
        padding: 0.7rem 1.5rem;
    }
    .stTabs [aria-selected="true"] {
        background: #555555;
        color: #FFFFFF;
    }

    /* --- Expander header --- */
    .st-expanderHeader {
        font-size: 1.05rem;
        color: #FFFFFF;
    }

    /* --- Alert boxes --- */
    .stAlert {
        border-radius: 8px;
    }

    /* --- Additional utility classes --- */
    .success-box {
        padding: 1rem;
        background-color: #C1440E;
        border: 1px solid #fce4ec;
        border-radius: 8px;
        margin: 1rem 0;
        color: #FFFFFF;
        font-size: 1.1rem;
    }
    .info-box {
        padding: 1rem;
        background-color: #C1440E;
        border: 1px solid #f8bbd0;
        border-radius: 8px;
        margin: 1rem 0;
        color: #FFFFFF;
        font-size: 1.1rem;
    }
    </style>
    """, unsafe_allow_html=True)


# Usage example: 
# In your main app file, call add_custom_css() before rendering title and intro:

add_custom_css()

st.markdown("""
<div style="border-radius: 18px; padding: 1.5rem 0 1.2rem 0; margin-bottom: 0.2rem; margin-top: -2.5rem;">
    <h1 class="custom-title" style="font-family: 'Bitcount Grid Double', system-ui, sans-serif; font-weight: 355; font-size: 110px;">
        <span style="color:#C1440E;">PADEL</span><span style="color:#FFD700;">MATE</span>
    </h1>
    <p class="custom-subtitle" style="font-size: 32px;">YOUR FRIENDLY PADEL CHATBOT</p>
</div>

<div style="border-radius: 14px; padding: 1.2rem 2.5rem 1.2rem 2.5rem; margin-top: 0; margin-bottom: 1.5rem;">
    <p class="intro-text" style="font-size: 22px;">
            Curious about padel? 🤔<br>
            Ask me anything about the sport!<br>
            I'm here to help you learn and discover everything about padel.
    </p>
</div>
""", unsafe_allow_html=True)

def setup_documents():
    """
    This function creates our document database
    NOTE: This runs every time someone uses the app
    In a real app, you'd want to save this data permanently
    """
    client = chromadb.Client()
    try:
        collection = client.get_collection(name="docs")
    except Exception:
        collection = client.create_collection(name="docs")
    
    # STUDENT TASK: Replace these 5 documents with your own!
    # Pick ONE topic: movies, sports, cooking, travel, technology
    # Each document should be 150-200 words
    # IMPORTANT: The quality of your documents affects answer quality!
    
    my_documents = [
        "A Brief History of Padel: Padel originated in Mexico in 1969, when Enrique Corcuera created the first court at his home. The sport quickly spread to Spain and Argentina, where it gained immense popularity. Unlike tennis, padel is played on a smaller court enclosed by walls, which are part of the game. Its combination of squash and tennis elements makes it dynamic and strategic. By the 1990s, padel had become one of Spain’s most popular sports. The World Padel Tour (WPT) was established in 2013, further professionalizing the sport. As of 2025, padel is played in over 90 countries and is among the fastest-growing sports in Europe and the Middle East. Its appeal lies in its accessibility—easy for beginners yet tactically rich for advanced players. Today, efforts are ongoing to make padel an Olympic sport. The game's unique mix of teamwork, reflexes, and wall-play has helped it carve out a distinct identity within the racket sport world.",

        "Rules and How the Game Works: Padel is typically played in doubles, 4 players in total, on a 10x20 meter enclosed court. The scoring system mirrors tennis: games, sets, and matches. Players use solid, stringless rackets and a ball slightly less pressurized than a tennis ball. Serves must be underhand and bounce once before crossing diagonally. After the serve, players can use walls to return shots, making positioning and anticipation crucial. The ball must bounce once on the ground before hitting the walls. Shots that hit the opponent's glass wall before the ground are still valid. Unlike tennis, power alone doesn't win matches—strategy, angles, and teamwork are vital. The net is lower than in tennis (88 cm at the center) and the game is played at a faster pace due to shorter court distances. Padel encourages long rallies, spectacular recoveries, and creative use of the back glass. The sport emphasizes reflexes, placement, and coordination, making it accessible yet complex enough for elite competition.",

        "Padel Equipment: What You Need to Play: Padel equipment is simple but specialized. The most important item is the padel racket—solid, perforated, and without strings. It’s made from carbon fiber or fiberglass with a foam core, and it varies in shape: round (control), diamond (power), or teardrop (hybrid). Players choose rackets based on their skill level and playing style. Padel balls resemble tennis balls but have slightly lower pressure for better control in enclosed courts. Footwear is also key: padel shoes offer lateral support and grip suitable for artificial turf and sand-filled surfaces. Apparel is similar to tennis—breathable clothes and wristbands are common. Safety gear like elbow or knee supports can help prevent injuries. Some players wear vibration-dampening gloves or wrist braces. Advanced gear might include smart sensors to track performance or custom-molded grips. While the setup cost is lower than other racket sports, choosing the right gear can greatly impact your game experience and performance.",

        "Who Are the Best Padel Players Today?: As of 2025, the top figures in padel dominate headlines in Spain, Argentina, and increasingly worldwide. On the men’s side, Alejandro Galán and Juan Lebrón have long held top rankings, known for their aggressive play and fluid teamwork. Arturo Coello, a rising Spanish star, has surged through the ranks with powerful smashes and clever tactics. In the women’s circuit, Alejandra Salazar and Gemma Triay form one of the strongest duos, known for their consistency and resilience under pressure. Paula Josemaría has also become a household name due to her speed and anticipation. The World Padel Tour (WPT) and Premier Padel Tour showcase elite talent, with tournaments held across Europe, South America, and the Middle East. Some tennis stars like Andy Murray and Serena Williams have also shown interest in the sport. Rankings evolve rapidly as younger players emerge, but Spain and Argentina remain the dominant forces in both talent and fanbase.",

        "The Global Rise of Padel: Padel has exploded in popularity over the last decade. Spain leads in player base and infrastructure, with over 20,000 courts and more than 5 million players. Argentina has long been a stronghold, with a deep-rooted padel culture. The sport has surged in Italy, Sweden, France, and the UAE, with courts popping up in urban areas and resorts. Padel's growth is driven by its social nature—it’s easy to learn, promotes teamwork, and doesn’t require advanced fitness to start. Major investments by sports clubs, ex-tennis pros, and celebrities (like Neymar and Beckham) have given it global exposure. In 2022, the International Padel Federation partnered with Qatar Sports Investments to launch the Premier Padel Tour, accelerating the sport’s international expansion. Padel clubs now exist in North America and Asia, and talks of Olympic inclusion are gaining momentum. Its combination of fun, accessibility, and fast-paced action continues to attract players of all ages and skill levels."
    ]
    
    # Add documents to database with unique IDs
    # ChromaDB needs unique identifiers for each document
    collection.add(
        documents=my_documents,
        ids=["padel1", "padel2", "padel3", "padel4", "padel5"]
    )
    
    return collection

def get_answer(collection, question):
    """
    This function searches documents and generates answers while minimizing hallucination
    """
    
    # STEP 1: Search for relevant documents in the database
    # We get 3 documents instead of 2 for better context coverage
    results = collection.query(
        query_texts=[question],    # The user's question
        n_results=3               # Get 3 most similar documents
    )
    
    # STEP 2: Extract search results
    # docs = the actual document text content
    # distances = how similar each document is to the question (lower = more similar)
    docs = results["documents"][0]
    distances = results["distances"][0]
    
    # STEP 3: Check if documents are actually relevant to the question
    # If no documents found OR all documents are too different from question
    # Return early to avoid hallucination
    if not docs or min(distances) > 1.5:  # 1.5 is similarity threshold - adjust as needed
        return "I don't have information about that topic in my documents."
    
    # STEP 4: Create structured context for the AI model
    # Format each document clearly with labels
    # This helps the AI understand document boundaries
    context = "\n\n".join([f"Document {i+1}: {doc}" for i, doc in enumerate(docs)])
    
    # STEP 5: Build improved prompt to reduce hallucination
    # Key changes from original:
    # - Separate context from instructions
    # - More explicit instructions about staying within context
    # - Clear format structure
    prompt = f"""Context information:
{context}

Question: {question}

Instructions: Answer ONLY using the information provided above. If the answer is not in the context, respond with "I don't know." Do not add information from outside the context.

Answer:"""
    
    # STEP 6: Generate answer with anti-hallucination parameters
    ai_model = pipeline("text2text-generation", model="google/flan-t5-small")
    response = ai_model(
        prompt, 
        max_length=75
    )
    
    # STEP 7: Extract and clean the generated answer
    answer = response[0]['generated_text'].strip()
    

    
    # STEP 8: Return the final answer
    return answer
            

# STREAMLIT BUILDING BLOCK 3: FUNCTION CALLS
# We call our function to set up the document database
# This happens every time someone uses the app
collection = setup_documents()

# STREAMLIT BUILDING BLOCK 4: TEXT INPUT BOX
# st.text_input() creates a box where users can type
# - First parameter: Label that appears above the box
# - The text users type gets stored in the 'question' variable
# - Users can click in this box and type their question
# 📝 Input label
question = st.text_input("Type your question here...")
# Placeholder text appears inside the box when empty

# STREAMLIT BUILDING BLOCK 5: BUTTON
# st.button() creates a clickable button
# - When clicked, all code inside the 'if' block runs
# - type="primary" makes the button blue and prominent
# - The button text appears on the button itself
# 🔍 Button
if st.button("🥎 **FIND OUT NOW!**"):
    
    # STREAMLIT BUILDING BLOCK 6: CONDITIONAL LOGIC
    # Check if user actually typed something (not empty)
    if question:
        
        # STREAMLIT BUILDING BLOCK 7: SPINNER (LOADING ANIMATION)
        # st.spinner() shows a rotating animation while code runs
        # - Text inside quotes appears next to the spinner
        # - Everything inside the 'with' block runs while spinner shows
        # - Spinner disappears when the code finishes
        with st.spinner("Serving up the perfect response..."):
            answer = get_answer(collection, question)
        
        # STREAMLIT BUILDING BLOCK 8: FORMATTED TEXT OUTPUT
        # st.write() can display different types of content
        # - **text** makes text bold (markdown formatting)
        # - First st.write() shows "Answer:" in bold
        # - Second st.write() shows the actual answer
        st.write("**Answer:**")
        st.write(answer)
    
    else:
        # STREAMLIT BUILDING BLOCK 9: SIMPLE MESSAGE
        # This runs if user didn't type a question
        # Reminds them to enter something before clicking
        st.write("Please enter a question!")


# STREAMLIT BUILDING BLOCK 6: SPINNER

# STREAMLIT BUILDING BLOCK 10: EXPANDABLE SECTION
# st.expander() creates a collapsible section
# - Users can click to show/hide the content inside
# - Great for help text, instructions, or extra information
# - Keeps the main interface clean
# ℹ️ Help section

st.markdown("""
<style>
    /* Change the expander header text color */
    .streamlit-expanderHeader {
        color: white !important;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)
with st.expander("ℹ️ **WHAT CAN I ASK ABOUT?**"):
    st.markdown("""
    <div style="background: rgba(245,245,245,0.65); border-radius: 20px; padding: 1.2rem 1.5rem; color: #333; font-family: 'Cal Sans', sans-serif;">
    🥎 <strong>THIS APP IS ALL ABOUT THE EXCITING WORLD OF PADEL</strong>!<br>
    YOU CAN ASK ABOUT:<br>
    - 📜 THE HISTORY AND ORIGINS OF PADEL<br>
    - 🏆 TOP PLAYERS AND WORLD RANKINGS<br>
    - 🧠 RULES AND COURT DYNAMICS<br>
    - 🎒 EQUIPMENT AND GEAR CHOICES<br>
    - 🌍 HOW PADEL IS GROWING GLOBALLY<br>
💡 <em>TRY QUESTIONS LIKE “WHO’S THE BEST PADEL PLAYER?” OR “HOW DO PADEL RULES WORK?”</em><br>
💡 <strong>TIP:</strong> <em>ASK ABOUT PLAYERS, RULES, OR COURT DESIGN TO GET SPECIFIC ANSWERS.</em>
    </div>
    """, unsafe_allow_html=True)




# Convert uploaded file to markdown text
def convert_to_markdown(file_path: str) -> str:
    path = Path(file_path)
    ext = path.suffix.lower()

    if ext == ".pdf":
        pdf_opts = PdfPipelineOptions(do_ocr=False)
        pdf_opts.accelerator_options = AcceleratorOptions(
            num_threads=4,
            device=AcceleratorDevice.CPU
        )
        converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(
                    pipeline_options=pdf_opts,
                    backend=DoclingParseV2DocumentBackend
                )
            }
        )
        doc = converter.convert(file_path).document
        return doc.export_to_markdown(image_mode="placeholder")

    if ext in [".doc", ".docx"]:
        converter = DocumentConverter()
        doc = converter.convert(file_path).document
        return doc.export_to_markdown(image_mode="placeholder")

    if ext == ".txt":
        try:
            return path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            return path.read_text(encoding="latin-1", errors="replace")

    raise ValueError(f"Unsupported extension: {ext}")


# Reset ChromaDB collection
def reset_collection(client, collection_name: str):
    try:
        client.delete_collection(name=collection_name)
    except Exception:
        pass
    return client.create_collection(name=collection_name)


# Add text chunks to ChromaDB
def add_text_to_chromadb(text: str, filename: str, collection_name: str = "documents"):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=700,
        chunk_overlap=100,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = splitter.split_text(text)

    if not hasattr(add_text_to_chromadb, 'client'):
        add_text_to_chromadb.client = chromadb.Client()
        add_text_to_chromadb.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        add_text_to_chromadb.collections = {}

    if collection_name not in add_text_to_chromadb.collections:
        try:
            collection = add_text_to_chromadb.client.get_collection(name=collection_name)
        except:
            collection = add_text_to_chromadb.client.create_collection(name=collection_name)
        add_text_to_chromadb.collections[collection_name] = collection

    collection = add_text_to_chromadb.collections[collection_name]

    for i, chunk in enumerate(chunks):
        embedding = add_text_to_chromadb.embedding_model.encode(chunk).tolist()

        metadata = {
            "filename": filename,
            "chunk_index": i,
            "chunk_size": len(chunk)
        }

        collection.add(
            embeddings=[embedding],
            documents=[chunk],
            metadatas=[metadata],
            ids=[f"{filename}_chunk_{i}"]
        )

    return collection



# Q&A function with source tracking
def get_answer_with_source(collection, question):
    results = collection.query(query_texts=[question], n_results=3)
    docs = results["documents"][0]
    distances = results["distances"][0]
    ids = results["ids"][0] if "ids" in results else ["unknown"] * len(docs)

    if not docs or min(distances) > 1.5:
        return "I don't have information about that topic in my documents.", "No source"

    context = "\n\n".join([f"Document {i+1}: {doc}" for i, doc in enumerate(docs)])
    prompt = f"""Context information:
{context}

Question: {question}

Instructions: Answer ONLY using the information provided above. If the answer is not in the context, respond with 'I don't know.' Do not add information from outside the context.

Answer:"""

    ai_model = pipeline("text2text-generation", model="google/flan-t5-small")
    response = ai_model(prompt, max_length=150)
    answer = response[0]['generated_text'].strip()
    # Extract source from best matching document
    best_source = ids[0].split('_chunk_')[0] if ids else "unknown"
    return answer, best_source

# Search history feature
def add_to_search_history(question, answer, source):
    if 'search_history' not in st.session_state:
        st.session_state.search_history = []
    st.session_state.search_history.insert(0, {
        'question': question,
        'answer': answer,
        'source': source,
        'timestamp': str(datetime.now().strftime("%H:%M:%S"))
    })
    if len(st.session_state.search_history) > 10:
        st.session_state.search_history = st.session_state.search_history[:10]

def show_search_history():
    st.subheader("🕒 Recent Searches")
    if 'search_history' not in st.session_state or not st.session_state.search_history:
        st.info("No searches yet.")
        return
    for i, search in enumerate(st.session_state.search_history):
        with st.expander(f"Q: {search['question'][:50]}... ({search['timestamp']})"):
            st.write("**Question:**", search['question'])
            st.write("**Answer:**", search['answer'])
            st.write("**Source:**", search['source'])

# Document manager with delete and preview
def show_document_manager():
    st.markdown(
        '<h2 style="color: white; font-family: Cal Sans, sans-serif;">📋 Manage documents</h2>',
        unsafe_allow_html=True
    )
    if not st.session_state.get('converted_docs', []):
        st.markdown(
            '<span style="color: white; font-family: Cal Sans, sans-serif;">No documents uploaded yet.</span>',
            unsafe_allow_html=True
        )
        return
    for i, doc in enumerate(st.session_state.converted_docs):
        col1, col2, col3 = st.columns([3, 1, 1])
        with col1:
            st.write(f"📄 {doc['filename']}")
            st.write(f"   Words: {len(doc['content'].split())}")
        with col2:
            if st.button("Preview", key=f"preview_{i}"):
                st.session_state[f'show_preview_{i}'] = True
        with col3:
            if st.button("Delete", key=f"delete_{i}"):
                st.session_state.converted_docs.pop(i)
                # Rebuild database
                st.session_state.collection = reset_collection(st.session_state.client, "documents")
                for d in st.session_state.converted_docs:
                    add_text_to_chromadb(d['content'], d['filename'], collection_name="documents")
                st.rerun()
        if st.session_state.get(f'show_preview_{i}', False):
            with st.expander(f"Preview: {doc['filename']}", expanded=True):
                st.text(doc['content'][:500] + "..." if len(doc['content']) > 500 else doc['content'])
                if st.button("Hide Preview", key=f"hide_{i}"):
                    st.session_state[f'show_preview_{i}'] = False
                    st.rerun()

# Document statistics
def show_document_stats():
    st.markdown(
        '<h2 style="color: white; font-family: Cal Sans, sans-serif;">📊 Document statistics</h2>',
        unsafe_allow_html=True
    )
    if not st.session_state.get('converted_docs', []):
        st.markdown('<span style="color: white; font-family: Cal Sans, sans-serif;">No documents to analyze.</span>', unsafe_allow_html=True)
        return
    total_docs = len(st.session_state.converted_docs)
    total_words = sum(len(doc['content'].split()) for doc in st.session_state.converted_docs)
    avg_words = total_words // total_docs if total_docs > 0 else 0
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Documents", total_docs)
    with col2:
        st.metric("Total Words", f"{total_words:,}")
    with col3:
        st.metric("Average Words/Doc", f"{avg_words:,}")
    file_types = {}
    for doc in st.session_state.converted_docs:
        ext = Path(doc['filename']).suffix.lower()
        file_types[ext] = file_types.get(ext, 0) + 1
    st.write("**File Types:**")
    for ext, count in file_types.items():
        st.write(f"• {ext}: {count} files")

# Helper: convert uploaded files to markdown and store in session
def convert_uploaded_files(uploaded_files):
    converted_docs = []
    for file in uploaded_files:
        suffix = Path(file.name).suffix
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
            temp_file.write(file.getvalue())
            temp_file_path = temp_file.name
        text = convert_to_markdown(temp_file_path)
        converted_docs.append({
            'filename': file.name,
            'content': text
        })
    return converted_docs

# Helper: add docs to database
def add_docs_to_database(collection, docs):
    count = 0
    for doc in docs:
        add_text_to_chromadb(doc['content'], doc['filename'], collection_name="documents")
        count += 1
    return count
    

def create_tabbed_interface():
    tab1, tab2, tab3, tab4 = st.tabs(["🎾 **UPLOAD**", "🔥 **QUESTIONS**", "📋 **MANAGE**", "📊 **STATS**"])

    with tab1:
        st.markdown(
            '<h2 style="color: white; font-family: Cal Sans, sans-serif;">🎾 Upload your padel documents</h2>',
            unsafe_allow_html=True
        )
        uploaded_files = st.file_uploader(
            "",
            type=["pdf", "doc", "docx", "txt"],
            accept_multiple_files=True,
            help="Supported formats: PDF, Word, and text files"
        )
        if st.button("💾 **ADD TO PADELMATE**", type="primary"):
            if uploaded_files:
                with st.spinner("Organizing your padel notes with passion..."):
                    converted_docs = convert_uploaded_files(uploaded_files)
                if 'converted_docs' not in st.session_state:
                    st.session_state.converted_docs = []
                if 'client' not in st.session_state:
                    st.session_state.client = chromadb.Client()
                if 'collection' not in st.session_state:
                    st.session_state.collection = reset_collection(st.session_state.client, "padel_documents")
                num_added = add_docs_to_database(st.session_state.collection, converted_docs)
                st.session_state.converted_docs.extend(converted_docs)
                st.markdown(
                    f"<span style='color: white; font-family: \"Cal Sans\", sans-serif;'>🎾 Added {num_added} padel notes to your Knowledge Base!</span>",
                    unsafe_allow_html=True
                )
            else:
                st.info("Please select your padel files to upload first.")

    with tab2:
        st.markdown('<h2 style="color: white; font-family: Cal Sans, sans-serif;">🔥 Ask anything about your padel docs</h2>', unsafe_allow_html=True)
        if st.session_state.get('converted_docs', []):
            question, search_button, clear_button = enhanced_question_interface()
            if search_button and question:
                with st.spinner("Searching your padel wisdom..."):
                    answer, source = get_answer_with_source(st.session_state.collection, question)
                st.markdown("### ✨ Your Padel-Powered Answer")
                st.write(answer)
                st.info(f"📄 Source: {source}")
                add_to_search_history(question, answer, source)
            if clear_button:
                st.session_state.search_history = []
                st.success("Search history cleared!")
            if st.session_state.get('search_history', []):
                show_search_history()
        else:
            st.markdown(
                "<span style='color: white; font-family: \"Cal Sans\", sans-serif;'>🎾 Upload some padel notes first to start your Q&A journey!</span>",
                unsafe_allow_html=True
            )

    with tab3:
        show_document_manager()

    with tab4:
        show_document_stats()




# MAIN APP
def enhanced_question_interface():
    with st.expander("💡 Example questions you can ask"):
        st.write("""
        🎾 What is the best padel training routine mentioned in my notes?  
        🏆 Who are the top-ranked players in padel history?  
        🛠️ What equipment recommendations are in my documents?
        """)
    question = st.text_input(
        "Type your question here:",
        placeholder="e.g., What is the best padel training routine mentioned in my notes?"
    )
    col1, col2 = st.columns([1, 1])
    # The following lines must NOT be indented further than this line
    with col1:
        search_button = st.button("🔍 **SEARCH**", type="primary")
    with col2:
        clear_button = st.button("🗑️ **CLEAR HISTORY**")
    return question, search_button, clear_button


def main():
    add_custom_css()
    # Removed the title and subtitle markdown lines here
    if 'converted_docs' not in st.session_state:
        st.session_state.converted_docs = []
    if 'client' not in st.session_state:
        st.session_state.client = chromadb.Client()
    if 'collection' not in st.session_state:
        st.session_state.collection = reset_collection(st.session_state.client, "documents")
    if 'search_history' not in st.session_state:
        st.session_state.search_history = []
    create_tabbed_interface()

def main():
    add_custom_css()
    
    if 'converted_docs' not in st.session_state:
        st.session_state.converted_docs = []
    if 'client' not in st.session_state:
        st.session_state.client = chromadb.Client()
    if 'collection' not in st.session_state:
        st.session_state.collection = reset_collection(st.session_state.client, "documents")
    if 'search_history' not in st.session_state:
        st.session_state.search_history = []

    with st.expander("**🎛️ ADVANCED FEATURES**"):
        create_tabbed_interface()

        st.markdown("""
        <style>
            /* Add black border to all expanders */
            .st-expander {
                border: 20px solid #000 !important;
                border-radius: 10px !important;
                box-shadow: none !important;
            }
        </style>
        """, unsafe_allow_html=True)
                    
        
    st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Cal+Sans&display=swap" rel="stylesheet">
<div style='text-align:center; margin-top:2.5rem; color:#fff; font-size:1.1rem; font-family: "Cal Sans", sans-serif;'>
    <span style="font-size:1.2rem; color:#fff;">
        <em>"Padel is like poison, it enters your body and you are infected forever."</em><br>
        <strong>~Paquito Navarro</strong>
    </span>
</div>
""", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
                
import streamlit as st

st.markdown("""
<style>
.fixed-button {
    position: fixed;
    bottom: 20px;
    right: 20px;
    z-index: 9999;
}
.fixed-button a {
    display: inline-block;
    background-color: #FFD700; /* Gold */
    color: black !important;
    padding: 0.75rem 1.5rem;
    border-radius: 30px; /* Pill shape */
    font-size: 1.3rem;
    font-weight: 600;
    text-decoration: none;
    cursor: pointer;
    box-shadow: 0 4px 8px rgba(255, 215, 0, 0.4);
    transition: background-color 0.3s ease, box-shadow 0.3s ease;
}
.fixed-button a:hover {
    background-color: #FFC107; /* Darker gold on hover */
    box-shadow: 0 6px 12px rgba(255, 193, 7, 0.6);
}
</style>
""", unsafe_allow_html=True)

st.markdown(f"""
<div class="fixed-button" style="font-family: 'Cal Sans', sans-serif;">
    <a href="https://padel-magazine.co.uk/category/news/international/" target="_blank" rel="noopener noreferrer" style="font-family: 'Cal Sans', sans-serif;">
        <strong>LATEST PADEL NEWS 📰</strong>
    </a>
</div>
""", unsafe_allow_html=True)

