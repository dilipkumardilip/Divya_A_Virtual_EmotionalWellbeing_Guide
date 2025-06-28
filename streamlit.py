import streamlit as st
import os
from langchain.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import tempfile
from PIL import Image
import pytesseract
import io
import base64

# Page configuration
st.set_page_config(
    page_title="Divya - Content Generator",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
<style>
   .main-header {
        background: linear-gradient(90deg, #FF6B6B, #4ECDC4);
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        text-align: center;
        color: black;
    }
    
    .content-type-card {
        background: #3082d4;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #FF6B6B;
        margin: 0.5rem 0;
    }
    
    .generated-content {
        background: #3eaed4;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        border: 1px solid #4ECDC4;
    }
    
    .stButton > button {
        background: linear-gradient(90deg, #FF6B6B, #4ECDC4);
        color: rgb(212, 151, 151);
        border: none;
        padding: 0.5rem 2rem;
        border-radius: 25px;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Initialize Ollama model
@st.cache_resource
def load_ollama_model(model_name="llama3.1:latest"):
    """Initialize Ollama model"""
    try:
        llm = Ollama(model=model_name, base_url="http://localhost:11434")
        return llm
    except Exception as e:
        st.error(f"Error loading Ollama model: {e}")
        st.info("Make sure Ollama is running locally with: `ollama serve`")
        return None

# Content type prompts
CONTENT_PROMPTS = {
    "Quote Reflection": {
        "description": "Deep dive into wisdom quotes with Gen Z perspective",
        "prompt": """
        You are Divya, a 22-year-old Indian emotional wisdom coach and influencer. Create a 120-150 word Instagram post reflecting on this quote/wisdom:

        Quote/Wisdom: {input_text}
        Additional Context: {context}

        Writing Style:
        - Authentic Gen Z voice (use "bestie", "lowkey", "literally")
        - Mix English with occasional Hindi words naturally
        - Personal, relatable examples
        - End with engaging question for comments
        - Include relevant emojis
        - Focus on emotional intelligence and relationships

        Structure:
        1. Hook (attention-grabbing opening)
        2. Personal insight/story
        3. Practical wisdom for Gen Z
        4. Call-to-action question

        Make it feel like advice from a wise friend, not a lecture.
        """
    },
    
    "Story/Scenario": {
        "description": "Real-life emotional scenarios and solutions",
        "prompt": """
        You are Divya, an emotional wisdom coach. Create a 120-150 word story-based Instagram post about:

        Topic/Scenario: {input_text}
        Additional Context: {context}

        Writing Style:
        - Start with "Story time..." or similar hook
        - Use real-life relatable scenarios
        - Show both the problem and healthy solution
        - Gen Z language and examples
        - Include emotional validation
        - End with community engagement question
        - Use emojis appropriately

        Structure:
        1. Story hook
        2. The scenario/conflict
        3. Healthy resolution/lesson
        4. Universal wisdom
        5. Engagement question

        Make it feel like sharing with close friends while teaching emotional intelligence.
        """
    },
    
    "Psychology Bite": {
        "description": "Simple psychology concepts for emotional growth",
        "prompt": """
        You are Divya, breaking down psychology concepts for Gen Z. Create a 120-150 word educational Instagram post about:

        Psychology Topic: {input_text}
        Additional Context: {context}

        Writing Style:
        - Start with "Psychology time!" or similar
        - Explain complex concepts simply
        - Use analogies Gen Z understands (social media, apps, etc.)
        - Include practical tips
        - Make it actionable
        - End with reflection question
        - Use brain/lightbulb emojis

        Structure:
        1. Engaging psychology hook
        2. Simple explanation of concept
        3. Real-life application
        4. Practical tip/technique
        5. Community question

        Avoid jargon. Make psychology accessible and interesting for young adults.
        """
    },
    
    "Pop Culture Critique": {
        "description": "Analyzing relationship tropes in movies/shows",
        "prompt": """
        You are Divya, analyzing pop culture through an emotional intelligence lens. Create a 120-150 word Instagram post critiquing:

        Movie/Show/Trend: {input_text}
        Additional Context: {context}

        Writing Style:
        - Start with "Can we talk about..." or "Unpopular opinion..."
        - Point out toxic vs healthy relationship patterns
        - Reference specific scenes/examples
        - Offer healthier alternatives
        - Use Gen Z slang ("red flags", "toxic", "normalize")
        - Include thinking/movie emojis
        - End with discussion starter

        Structure:
        1. Pop culture reference hook
        2. Problematic pattern identification
        3. Why it's harmful
        4. Healthier alternative
        5. Discussion question

        Be critical but constructive. Help followers develop media literacy around relationships.
        """
    },
    
    "Emotional Practice": {
        "description": "Practical exercises for emotional wellbeing",
        "prompt": """
        You are Divya, teaching practical emotional wellness techniques. Create a 120-150 word Instagram post about:

        Practice/Technique: {input_text}
        Additional Context: {context}

        Writing Style:
        - Start with "Let's practice..." or "Try this with me..."
        - Give step-by-step instructions
        - Explain the 'why' behind the practice
        - Make it feel like doing it together
        - Include encouraging language
        - Use calming emojis (✨🧘‍♀️💕)
        - End with experience-sharing question

        Structure:
        1. Practice introduction
        2. Step-by-step guide
        3. Benefits explanation
        4. Personal encouragement
        5. Community sharing prompt

        Make it feel interactive and supportive, like a friend guiding through self-care.
        """
    }
}

# Text extraction functions
def extract_text_from_image(image):
    """Extract text from image using OCR"""
    try:
        text = pytesseract.image_to_string(image)
        return text.strip()
    except Exception as e:
        st.error(f"Error extracting text from image: {e}")
        return ""

def extract_text_from_pdf(pdf_file):
    """Extract text from PDF file"""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            tmp_file.write(pdf_file.getvalue())
            tmp_file_path = tmp_file.name
        
        loader = PyPDFLoader(tmp_file_path)
        documents = loader.load()
        
        # Combine all pages
        full_text = "\n".join([doc.page_content for doc in documents])
        
        # Clean up temp file
        os.unlink(tmp_file_path)
        
        return full_text
    except Exception as e:
        st.error(f"Error extracting text from PDF: {e}")
        return ""

def chunk_text(text, chunk_size=1000):
    """Split long text into chunks"""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=200,
        length_function=len,
    )
    chunks = text_splitter.split_text(text)
    return chunks

def generate_content(llm, content_type, input_text, context=""):
    """Generate content using Ollama model"""
    try:
        prompt_template = PromptTemplate(
            input_variables=["input_text", "context"],
            template=CONTENT_PROMPTS[content_type]["prompt"]
        )
        
        chain = LLMChain(llm=llm, prompt=prompt_template)
        
        result = chain.run(input_text=input_text, context=context)
        return result.strip()
    
    except Exception as e:
        st.error(f"Error generating content: {e}")
        return None

# Main UI
def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>✨ Divya - Emotional Wisdom Content Generator</h1>
        <p>AI-Powered Content Creation for Gen Z Mental Health & Relationships</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar for model selection
    with st.sidebar:
        st.header("🤖 Model Configuration")
        
        # Model selection
        available_models = ["llama2", "llama3.1:latest", "llama2:7b", "mistral", "codellama", "neural-chat"]
        selected_model = st.selectbox(
            "Select Ollama Model:",
            available_models,
            help="Make sure the model is downloaded in Ollama"
        )
        
        # Load model
        llm = load_ollama_model(selected_model)
        
        if llm:
            st.success(f"✅ {selected_model} - {llm} loaded successfully!")
        else:
            st.error("❌ Model not available")
            st.stop()
        
        st.markdown("---")
        st.markdown("""
        **💡 Quick Setup:**
        1. Install Ollama
        2. Run `ollama serve`
        3. Download model: `ollama pull llama2`
        """)
    
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📝 Content Input")
        
        # Content type selection
        st.subheader("1. Choose Content Type")
        content_type = st.radio(
            "Select the type of content you want to create:",
            list(CONTENT_PROMPTS.keys()),
            help="Each type has a specialized prompt for different content styles"
        )
        
        # Display content type description
        st.markdown(f"""
        <div class="content-type-card">
            <strong>{content_type}</strong><br>
            {CONTENT_PROMPTS[content_type]['description']}
        </div>
        """, unsafe_allow_html=True)
        
        st.subheader("2. Primary Input")
        
        # Input method selection
        input_method = st.radio(
            "How do you want to provide the main content?",
            ["Text Input", "Upload File", "Image with Text"]
        )
        
        primary_input = ""
        
        if input_method == "Text Input":
            primary_input = st.text_area(
                "Enter your quote, topic, or main content:",
                placeholder="e.g., 'You have a right to your actions, but not to the fruits of your actions' - Bhagavad Gita",
                height=100
            )
        
        elif input_method == "Upload File":
            uploaded_file = st.file_uploader(
                "Upload PDF or Text file:",
                type=['pdf', 'txt'],
                help="The content will be extracted automatically"
            )
            
            if uploaded_file:
                if uploaded_file.type == "application/pdf":
                    with st.spinner("Extracting text from PDF..."):
                        primary_input = extract_text_from_pdf(uploaded_file)
                else:
                    primary_input = str(uploaded_file.read(), "utf-8")
                
                if primary_input:
                    st.success(f"✅ Extracted {len(primary_input)} characters")
                    with st.expander("Preview extracted text"):
                        st.text(primary_input[:500] + "..." if len(primary_input) > 500 else primary_input)
        
        elif input_method == "Image with Text":
            uploaded_image = st.file_uploader(
                "Upload image with text:",
                type=['png', 'jpg', 'jpeg'],
                help="We'll extract text using OCR"
            )
            
            if uploaded_image:
                image = Image.open(uploaded_image)
                st.image(image, caption="Uploaded Image", use_column_width=True)
                
                with st.spinner("Extracting text from image..."):
                    primary_input = extract_text_from_image(image)
                
                if primary_input:
                    st.success("✅ Text extracted successfully")
                    st.text_area("Extracted text:", value=primary_input, height=100)
        
        st.subheader("3. Additional Context (Optional)")
        additional_context = st.text_area(
            "Provide additional context, background, or specific guidance:",
            placeholder="e.g., Focus on heartbreak recovery, target audience is college students, mention Delhi University course...",
            height=80
        )
        
        # Generate button
        generate_button = st.button(
            "🚀 Generate Content",
            disabled=not primary_input,
            help="Make sure to provide primary input first"
        )
    
    with col2:
        st.header("📱 Generated Content")
        
        if generate_button and primary_input:
            with st.spinner("Generating content with AI..."):
                # If text is too long, chunk it
                if len(primary_input) > 2000:
                    chunks = chunk_text(primary_input, 1500)
                    primary_input = chunks[0]  # Use first chunk
                    st.info(f"Text was long, using first {len(primary_input)} characters")
                
                generated_content = generate_content(
                    llm=llm,
                    content_type=content_type,
                    input_text=primary_input,
                    context=additional_context
                )
                
                if generated_content:
                    st.markdown("""
                    <div class="generated-content">
                    """, unsafe_allow_html=True)
                    
                    st.subheader(f"✨ {content_type} Content")
                    st.write(generated_content)
                    
                    # Word count
                    word_count = len(generated_content.split())
                    st.caption(f"📊 Word count: {word_count} words")
                    
                    # Copy button simulation
                    st.text_area(
                        "Copy this content:",
                        value=generated_content,
                        height=200,
                        help="Select all and copy this content"
                    )
                    
                    st.markdown("</div>", unsafe_allow_html=True)
                    
                    # Additional features
                    st.subheader("📋 Content Actions")
                    
                    col_a, col_b, col_c = st.columns(3)
                    
                    with col_a:
                        if st.button("🔄 Regenerate"):
                            st.experimental_rerun()
                    
                    with col_b:
                        if st.button("✂️ Make Shorter"):
                            st.info("Feature coming soon!")
                    
                    with col_c:
                        if st.button("📈 Make Longer"):
                            st.info("Feature coming soon!")
        
        else:
            st.info("👆 Configure your content settings and click 'Generate Content' to create your post!")
            
            # Show example
            st.subheader("📖 Example Output")
            st.markdown("""
            <div class="generated-content">
            <strong>Sample Quote Reflection:</strong><br><br>
            Bestie, can we talk about this Gita wisdom for a sec? 💭 "You have a right to your actions, but not to the fruits" literally changed how I see everything!

            Like, we're so obsessed with outcomes - will they text back? Will I get the job? Will this relationship work? But here's the thing: when we're constantly worried about results, we're not even present for the process.

            I started applying this when my situationship was giving me anxiety. Instead of spiraling about "where is this going," I focused on - am I being authentic? Am I communicating well? Am I respecting my boundaries? That's literally all I can control!

            The universe has its own timing, and our job is to show up fully without attachment to specific outcomes. It's actually so freeing! ✨

            What's one situation where you struggle with letting go of outcomes? Drop it below! 👇

            #EmotionalWisdom #BhagavadGita #GenZWisdom #LetGo
            </div>
            """, unsafe_allow_html=True)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; margin-top: 2rem;">
        <p>Built with ❤️ for Divya's Emotional Wisdom Platform</p>
        <p>Powered by Ollama + Langchain + Streamlit</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()