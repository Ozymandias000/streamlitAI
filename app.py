import tempfile
from pathlib import Path
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.backend.docling_parse_v2_backend import DoclingParseV2DocumentBackend
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions, AcceleratorOptions, AcceleratorDevice


# Simple Q&A App using Streamlit
# Students: Replace the documents below with your own!

# IMPORTS - These are the libraries we need
import streamlit as st          # Creates web interface components
import chromadb                # Stores and searches through documents  
from transformers import pipeline  # AI model for generating answers

# Custom CSS for button styling 
st.markdown("""
    <style>
    div.stButton > button:first-child {
        background-color: #C1440E;
        color: white;
        border-radius: 8px;
        height: 3em;
        width: 100%;
        font-size: 16px;
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)

# 🌄 Set background image and style
st.markdown(
    """
    <style>
    .stApp {
        background-image: url("https://cdn.pixabay.com/photo/2021/06/04/06/54/racket-6308994_1280.jpg");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
        color: white;
    }

    .block-container {
        background-color: rgba(0, 0, 0, 0.5);
        padding: 2rem;
        border-radius: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Your app starts here

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

# MAIN APP STARTS HERE - This is where we build the user interface

# STREAMLIT BUILDING BLOCK 1: PAGE TITLE
# st.title() creates a large heading at the top of your web page
# The emoji 🤖 makes it more visually appealing
# This appears as the biggest text on your page
# Title
st.markdown("""
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Bitcount+Grid+Double:wght@355&display=swap" rel="stylesheet">
<link href="https://fonts.googleapis.com/css2?family=Cal+Sans&display=swap" rel="stylesheet">

<style>
  h1.custom-title {
    font-family: "Bitcount Grid Double", system-ui;
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
    margin-top: -10px; /* pull it closer */
    text-transform: uppercase;
    letter-spacing: 1.1px;
  }
</style>

<h1 class="custom-title">
  <span style="color:#C1440E;">PADEL</span><span style="color:#FFD700;">MATE</span>
</h1>
<p class="custom-subtitle">YOUR FRIENDLY PADEL CHATBOT</p>
""", unsafe_allow_html=True)


# STREAMLIT BUILDING BLOCK 2: DESCRIPTIVE TEXT  
# st.write() displays regular text on the page
# Use this for instructions, descriptions, or any text content
# It automatically formats the text nicely
# 💬 Intro
st.markdown("""
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=ZCOOL+QingKe+HuangYou&display=swap" rel="stylesheet">

<p style="
    font-family: 'ZCOOL QingKe HuangYou', sans-serif; 
    font-size: 20px; 
    color: white;
    text-align: right;
">
    Curious about padel? 🤔<br>
    Ask me anything about the sport!<br>
    I'm here to help you learn and discover everything about padel.
</p>
""", unsafe_allow_html=True)
# This text appears below the title and gives context to the app

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
if st.button("🥎 FIND OUT NOW!", type="primary"):
    
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
with st.expander("ℹ️ **WHAT CAN I ASK ABOUT?**"):
    st.markdown("""
    🥎 THIS APP IS ALL ABOUT THE EXCITING WORLD OF **PADEL**!
    
    YOU CAN ASK ABOUT:
    - 📜 THE HISTORY AND ORIGINS OF **PADEL**
    - 🏆 TOP PLAYERS AND WORLD RANKINGS
    - 🧠 RULES AND COURT DYNAMICS
    - 🎒 EQUIPMENT AND GEAR CHOICES
    - 🌍 HOW **PADEL** IS GROWING GLOBALLY
    
    💡 *TRY QUESTIONS LIKE “WHO’S THE BEST PADEL PLAYER?” OR “HOW DO PADEL RULES WORK?”*
    💡 **TIP:** *ASK ABOUT PLAYERS, RULES, OR COURT DESIGN TO GET SPECIFIC ANSWERS.*
    """)

# TO RUN: Save as app.py, then type: streamlit run app.py

# --- ADVANCED: DOCUMENT UPLOAD & KNOWLEDGE BASE EXTENSION ---


import streamlit as st
import tempfile
import chromadb
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

# Dummy convert_to_markdown for illustration
def convert_to_markdown(file_path: str) -> str:
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        return f.read()

def reset_collection(client, collection_name: str):
    try:
        client.delete_collection(name=collection_name)
    except Exception:
        pass
    return client.create_collection(name=collection_name)

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
    else:
        collection = add_text_to_chromadb.collections[collection_name]

    for i, chunk in enumerate(chunks):
        embedding = add_text_to_chromadb.embedding_model.encode(chunk).tolist()
        metadata = {"filename": filename, "chunk_index": i, "chunk_size": len(chunk)}
        collection.add(
            embeddings=[embedding],
            documents=[chunk],
            metadatas=[metadata],
            ids=[f"{filename}_chunk_{i}"]
        )
    return collection

def main():
    st.title("Padelmate with Document Upload")

    if 'upload_section_visible' not in st.session_state:
        st.session_state['upload_section_visible'] = False

    if st.button("Upload Documents"):
        st.session_state['upload_section_visible'] = not st.session_state['upload_section_visible']

    if st.session_state['upload_section_visible']:
        with st.expander("Upload your documents here", expanded=True):
            uploaded_files = st.file_uploader(
                "Choose files",
                type=["pdf", "doc", "docx", "txt"],
                accept_multiple_files=True
            )
            if st.button("Chunk and Store Documents"):
                if uploaded_files:
                    client = chromadb.Client()
                    collection = reset_collection(client, "documents")
                    for file in uploaded_files:
                        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                            temp_file.write(file.getvalue())
                            temp_file_path = temp_file.name
                        text = convert_to_markdown(temp_file_path)  # Replace with your actual function
                        collection = add_text_to_chromadb(text, file.name, collection_name="documents")
                    st.success("Documents processed and stored!")
                else:
                    st.error("Please upload at least one file.")

if __name__ == "__main__":
    main()


def reset_collection(client, collection_name: str):
    """Delete existing collection and create a new empty one"""
    try:
        client.delete_collection(name=collection_name)
    except Exception:
        pass
    new_collection = client.create_collection(name=collection_name)
    return new_collection

def add_text_to_chromadb(text: str, filename: str, collection_name: str = "documents"):
    """
    Add text to existing or new ChromaDB collection.
    Safe to call multiple times with same collection_name.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=700,
        chunk_overlap=100,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = splitter.split_text(text)
    # Initialize components (reuse if possible)
    if not hasattr(add_text_to_chromadb, 'client'):
        add_text_to_chromadb.client = chromadb.Client()
        add_text_to_chromadb.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        add_text_to_chromadb.collections = {}
    # Get or create collection
    if collection_name not in add_text_to_chromadb.collections:
        try:
            collection = add_text_to_chromadb.client.get_collection(name=collection_name)
        except:
            collection = add_text_to_chromadb.client.create_collection(name=collection_name)
        add_text_to_chromadb.collections[collection_name] = collection
    collection = add_text_to_chromadb.collections[collection_name]
    # Process chunks
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

with st.expander("📄 **UPLOAD YOUR OWN DOCUMENTS (PDF, DOCX, TXT)**"):
    uploaded_files = st.file_uploader(
        "Upload documents to extend the knowledge base",
        type=["pdf", "doc", "docx", "txt"],
        accept_multiple_files=True
    )
    if st.button("Chunk and Store Documents"):
        if uploaded_files:
            client = chromadb.Client()
            collection = reset_collection(client, "documents")
            for file in uploaded_files:
                with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                    temp_file.write(file.getvalue())
                    temp_file_path = temp_file.name
                # You must implement convert_to_markdown for your environment
                text = convert_to_markdown(temp_file_path)
                collection = add_text_to_chromadb(text, file.name, collection_name="documents")
                st.write(f"Stored {file.name} in ChromaDB")
        else:
            st.error("Upload files first!")

    # Q&A on uploaded documents
    question = st.text_input("Ask a question about your uploaded documents:")
    if st.button("Get Answer from Uploaded Docs"):
        if question:
            # Use the uploaded docs collection if available, else fallback
            try:
                client = chromadb.Client()
                collection = client.get_collection(name="documents")
                answer = get_answer(collection, question)
                st.write("**Answer:**")
                st.write(answer)
            except Exception:
                st.error("No uploaded documents found. Please upload and store documents first.")
        else:
            st.write("Please enter a question!")
