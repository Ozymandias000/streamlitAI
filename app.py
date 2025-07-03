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
    client = chromadb.Client()
    try:
        collection = client.get_collection(name="docs")
    except Exception:
        collection = client.create_collection(name="docs")

    doc_filenames = ['2-game-regulations.md', '2017_HISTORY-OF-PADEL_photo.md', 'Thinkpadelweb.md']
    documents = []
    for filename in doc_filenames:
        with open(filename, 'r', encoding='utf-8') as f:
            documents.append(f.read())

    collection.add(
        documents=documents,
        ids=["doc1", "doc2", "doc3"]
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

