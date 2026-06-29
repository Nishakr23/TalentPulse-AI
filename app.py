import streamlit as st
import json, time, re, io
from datetime import datetime
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from docx import Document

st.set_page_config(page_title="TalentPulse AI", layout="wide", page_icon="⚡")

# Futuristic CSS + Animation
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Rajdhani:wght@300;500;700&display=swap');

   .stApp {
        background: linear-gradient(135deg, #0a0e27 0%, #1a1a2e 50%, #16213e 100%);
    }

    h1 {
        font-family: 'Orbitron', sans-serif;
        background: linear-gradient(90deg, #00f5ff, #7b2ff7, #f72585);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 900;
        font-size: 3rem!important;
        text-align: center;
        animation: glow 2s ease-in-out infinite alternate;
        margin-bottom: 0.5rem;
    }

    @keyframes glow {
        from { filter: drop-shadow(0 0 10px #00f5ff); }
        to { filter: drop-shadow(0 0 20px #7b2ff7); }
    }

    /* NEW: Neural Scanner Animation */
   .neural-scanner {
        width: 100%;
        height: 80px;
        position: relative;
        margin: 1rem 0 2rem 0;
        overflow: hidden;
    }

   .scan-line {
        position: absolute;
        width: 100%;
        height: 2px;
        background: linear-gradient(90deg, transparent, #00f5ff, #7b2ff7, #f72585, transparent);
        box-shadow: 0 0 20px #00f5ff;
        animation: scan 3s linear infinite;
    }

    @keyframes scan {
        0% { top: 0%; opacity: 0; }
        10% { opacity: 1; }
        90% { opacity: 1; }
        100% { top: 100%; opacity: 0; }
    }

   .neural-nodes {
        position: absolute;
        width: 100%;
        height: 100%;
        display: flex;
        justify-content: space-around;
        align-items: center;
    }

   .node {
        width: 8px;
        height: 8px;
        background: #00f5ff;
        border-radius: 50%;
        box-shadow: 0 0 15px #00f5ff;
        animation: pulse 2s ease-in-out infinite;
    }

   .node:nth-child(2) { animation-delay: 0.3s; background: #7b2ff7; box-shadow: 0 0 15px #7b2ff7; }
   .node:nth-child(3) { animation-delay: 0.6s; background: #f72585; box-shadow: 0 0 15px #f72585; }
   .node:nth-child(4) { animation-delay: 0.9s; }
   .node:nth-child(5) { animation-delay: 1.2s; background: #7b2ff7; box-shadow: 0 0 15px #7b2ff7; }

    @keyframes pulse {
        0%, 100% { transform: scale(1); opacity: 0.3; }
        50% { transform: scale(1.5); opacity: 1; }
    }

   .scanner-text {
        position: absolute;
        width: 100%;
        text-align: center;
        top: 50%;
        transform: translateY(-50%);
        font-family: 'Rajdhani', sans-serif;
        font-size: 0.9rem;
        color: #00f5ff;
        letter-spacing: 3px;
        text-transform: uppercase;
        animation: flicker 3s infinite;
    }

    @keyframes flicker {
        0%, 100% { opacity: 0.4; }
        50% { opacity: 1; }
    }

   .stButton>button {
        background: linear-gradient(90deg, #7b2ff7, #f72585);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.75rem 2rem;
        font-family: 'Rajdhani', sans-serif;
        font-weight: 700;
        font-size: 1.1rem;
        box-shadow: 0 0 20px rgba(123, 47, 247, 0.5);
        transition: all 0.3s;
    }

   .stButton>button:hover {
        box-shadow: 0 0 30px rgba(247, 37, 133, 0.8);
        transform: translateY(-2px);
    }

   .upload-box {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        border: 2px solid rgba(123, 47, 247, 0.3);
        border-radius: 16px;
        padding: 2rem;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
    }

   .metric-card {
        background: linear-gradient(135deg, rgba(123, 47, 247, 0.1), rgba(0, 245, 255, 0.1));
        border: 1px solid rgba(123, 47, 247, 0.3);
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
    }

   .stDataFrame {
        background: rgba(255, 255, 255, 0.03);
        border-radius: 12px;
    }

    p, div, span {
        font-family: 'Rajdhani', sans-serif;
        color: #e0e0e0;
    }

   .rank-badge {
        background: linear-gradient(90deg, #f72585, #7b2ff7);
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-weight: 700;
        display: inline-block;
    }



</style>
""", unsafe_allow_html=True)

st.markdown("<h1>⚡ TALENT PULSE AI ⚡</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; font-size: 1.2rem; color: #00f5ff;'>Neural Candidate Discovery & Ranking Engine</p>", unsafe_allow_html=True)

# NEW: Animation Block
st.markdown("""
<div class="neural-scanner">
    <div class="neural-nodes">
        <div class="node"></div>
        <div class="node"></div>
        <div class="node"></div>
        <div class="node"></div>
        <div class="node"></div>
    </div>
    <div class="scan-line"></div>
    <div class="scanner-text">◢ SCANNING NEURAL PATHWAYS ◣</div>
</div>
""", unsafe_allow_html=True)

# Baaki ka code same rahega...
@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

def extract_text_from_docx(file):
    doc = Document(file)
    return '\n'.join([para.text for para in doc.paragraphs])

def parse_candidates(file):
    content = file.read().decode('utf-8')
    try:
        data = json.loads(content)
        if isinstance(data, list):
            return data
        elif isinstance(data, dict) and 'candidates' in data:
            return data['candidates']
    except:
        pass
    file.seek(0)
    return [json.loads(line) for line in file if line.strip()]

# ... baaki functions same as before ...

def extract_from_jd(jd):
    jd_lower = jd.lower()
    yoe_min, yoe_max = 5, 99
    yoe_match = re.search(r'(\d+)\s*[-to]+\s*(\d+)?\s*years?', jd_lower)
    if yoe_match:
        yoe_min = int(yoe_match.group(1))
        yoe_max = int(yoe_match.group(2)) if yoe_match.group(2) else 99
    elif re.search(r'(\d+)\+\s*years?', jd_lower):
        yoe_min = int(re.search(r'(\d+)\+\s*years?', jd_lower).group(1))

    bad_titles = ['marketing manager', 'hr manager', 'sales executive', 'graphic designer',
                  'accountant', 'project manager', 'customer support', 'civil engineer',
                  'mechanical engineer', 'electrical engineer', 'frontend engineer', 'backend engineer']
    bad_sections = re.findall(r'(?:not want|disqualify|reject|not a fit|we will not move forward).*?([A-Za-z\s]+(?:Manager|Engineer|Scientist|Analyst|Designer|Developer|Recruiter|Executive|Specialist))', jd, re.IGNORECASE)
    bad_titles.extend([b.lower().strip() for b in bad_sections])
    bad_titles = list(dict.fromkeys(bad_titles))

    good_titles = ['ai engineer', 'ml engineer', 'machine learning engineer', 'data scientist',
                   'nlp engineer', 'computer vision engineer', 'research scientist', 'applied scientist']
    jd_title_match = re.search(r'Job Description:\s*([A-Za-z\s]+)', jd, re.IGNORECASE)
    if jd_title_match:
        jd_title = jd_title_match.group(1).lower().strip()
        if any(kw in jd_title for kw in ['ai', 'ml', 'machine learning', 'data', 'nlp', 'research']):
            good_titles.insert(0, jd_title)
    good_titles = list(dict.fromkeys(good_titles))

    stop_words = {'the','and','for','with','years','experience','role','job','work','team','company','business','strong','good','excellent','ability','must','should','required','open','type','time','series','engineer'}
    words = re.findall(r'\b[a-z]{4,}\b', jd_lower)
    keywords = list(dict.fromkeys([w for w in words if w not in stop_words]))[:20]
    keywords.extend(['rag', 'recommendation', 'vector', 'embedding', 'llm', 'machine', 'learning'])
    keywords = list(dict.fromkeys(keywords))
    return yoe_min, yoe_max, bad_titles, good_titles, keywords

def build_candidate_text(c):
    prof = c.get('profile', {})
    skills = ', '.join([s['name'] for s in c.get('skills', [])[:5]])
    career = []
    for job in c.get('career_history', [])[:3]:
        career.append(f"{job.get('title','')} at {job.get('company_name','')}: {job.get('description','')[:150]}")
    career_text = '. '.join(career)
    return f"Title: {prof.get('current_title','')}. Summary: {prof.get('summary','')}. Career: {career_text}. Skills: {skills}."

def days_since(date_str):
    try: return (datetime.now() - datetime.fromisoformat(date_str.replace('Z', ''))).days
    except: return 999

def is_honeypot(cand):
    prof = cand.get('profile', {})
    yoe = prof.get('years_of_experience', 0)
    companies = cand.get('career_history', [])
    skills = cand.get('skills', [])
    if yoe > 7 and len(companies) == 1 and companies[0].get('duration_months', 0) < 36: return True
    if len([s for s in skills if s.get('proficiency') == 'expert']) > 8 and yoe < 2: return True
    return False

def score_redrob_signals(sig):
    score = 0.0
    if sig.get('open_to_work_flag', False): score += 0.15
    if days_since(sig.get('last_active_date', '2000-01-01')) < 14: score += 0.20
    else: score -= 0.30
    resp_rate = sig.get('recruiter_response_rate', 0)
    if resp_rate > 0.6: score += 0.25
    elif resp_rate < 0.2: score -= 0.40
    elif resp_rate < 0.05: score -= 0.80
    if sig.get('interview_completion_rate', 0) > 0.8: score += 0.15
    if sig.get('offer_acceptance_rate', -1) > 0.5: score += 0.10
    if sig.get('notice_period_days', 180) <= 30: score += 0.15
    return max(-0.8, min(score, 1.0))

def is_disqualified(cand, sig, YOE_MIN, YOE_MAX, BAD_TITLES, GOOD_TITLES):
    if is_honeypot(cand): return True
    prof = cand.get('profile', {})
    title = prof.get('current_title', '').lower()
    yoe = prof.get('years_of_experience', 0)
    if yoe < YOE_MIN or yoe > YOE_MAX: return True
    if any(bad in title for bad in BAD_TITLES): return True
    if 'engineer' in GOOD_TITLES[0] and not any(eng in title for eng in ['engineer', 'developer', 'scientist', 'architect']):
        return True
    companies = [c.get('company_name', '').lower() for c in cand.get('career_history', [])]
    consulting_firms = ['tcs', 'infosys', 'wipro', 'accenture', 'cognizant', 'capgemini']
    if len(companies) > 0 and all(any(cf in comp for cf in consulting_firms) for comp in companies):
        return True
    if days_since(sig.get('last_active_date', '2000-01-01')) > 180: return True
    if sig.get('recruiter_response_rate', 1) < 0.05: return True
    return False

def run_ranking(candidates, jd_text, model, top_n=50):
    YOE_MIN, YOE_MAX, BAD_TITLES, GOOD_TITLES, KEYWORDS = extract_from_jd(jd_text)
    jd_emb = model.encode(jd_text)

    texts = [build_candidate_text(c) for c in candidates]
    cand_emb = model.encode(texts, batch_size=128, show_progress_bar=False)
    semantic_scores = np.dot(cand_emb, jd_emb)
    results = []

    progress = st.progress(0, text="🧠 Neural scoring in progress...")
    for i, cand in enumerate(candidates):
        sig = cand.get('redrob_signals', {})
        if is_disqualified(cand, sig, YOE_MIN, YOE_MAX, BAD_TITLES, GOOD_TITLES):
            progress.progress((i+1)/len(candidates))
            continue

        prof = cand.get('profile', {})
        title = prof.get('current_title', '').lower()
        title_boost = 0.15 if any(good in title for good in GOOD_TITLES[:4]) else 0.05 if any(good in title for good in GOOD_TITLES) else 0

        career_text = ' '.join([j.get('description','').lower() + ' ' + j.get('company_name','').lower() for j in cand.get('career_history', [])[:3]])
        skills_text = ' '.join([s['name'].lower() for s in cand.get('skills', [])])
        combined_text = career_text + ' ' + skills_text + ' ' + title

        product_companies = ['google', 'meta', 'amazon', 'microsoft', 'flipkart', 'swiggy', 'zomato', 'paytm', 'uber', 'ola', 'cred', 'razorpay', 'zerodha', 'freshworks', 'product']
        has_product_scale = 0.15 if any(kw in career_text for kw in ['production', 'scaled', 'million users', 'billion', 'shipped']) else 0
        has_product_exp = 0.25 if any(pc in career_text for pc in product_companies) else has_product_scale

        ml_keywords = ['rag', 'retrieval', 'vector', 'embedding', 'recommendation', 'search', 'llm', 'machine learning', 'deep learning', 'neural', 'nlp', 'computer vision']
        has_rag_work = 0.30 if any(kw in combined_text for kw in ml_keywords) else 0

        title_penalty = -0.35 if any(t in title for t in ['civil', 'mechanical', 'electrical', 'frontend', 'backend']) else -0.20 if 'software engineer' in title and not has_rag_work else 0

        keyword_matches = sum(1 for kw in KEYWORDS if kw in combined_text)
        keyword_boost = min(keyword_matches * 0.05, 0.20)

        relevance_boost = has_product_exp + has_rag_work + title_penalty + keyword_boost
        final = semantic_scores[i] * 0.25 + score_redrob_signals(sig) * 0.25 + relevance_boost * 0.5 + title_boost

        results.append({
            'candidate_id': cand['candidate_id'],
            'score': round(float(final), 6),
            'cand_obj': cand,
            'sig': sig
        })
        progress.progress((i+1)/len(candidates))

    results.sort(key=lambda x: (-x['score'], x['candidate_id']))
    if len(results) < top_n:
        st.warning(f"⚠️ Only {len(results)} candidates passed filters. Showing all.")
        top_n = len(results)

    top_results = results[:top_n]
    final_results = []
    for rank, r in enumerate(top_results, 1):
        cand = r['cand_obj']
        sig = r['sig']
        prof = cand.get('profile', {})
        title = prof.get('current_title', '')
        yoe = prof.get('years_of_experience', 0)
        career = cand.get('career_history', [])

        reason_parts = []
        career_text = ' '.join([j.get('description','').lower() + ' ' + j.get('company_name','').lower() for j in career[:2]])
        skills_text = ' '.join([s['name'].lower() for s in cand.get('skills', [])])
        combined = career_text + ' ' + skills_text

        product_companies = ['google', 'meta', 'amazon', 'microsoft', 'flipkart', 'swiggy', 'zomato', 'paytm']
        has_product_exp = any(pc in career_text for pc in product_companies)
        has_rag = any(kw in combined for kw in ['rag', 'retrieval', 'vector', 'embedding', 'llm', 'recommendation', 'search'])
        has_ml = any(kw in combined for kw in ['machine learning', 'deep learning', 'neural', 'ml', 'ai'])

        if has_rag and has_product_exp:
            reason_parts.append(f"{yoe} YOE {title} building RAG systems at product companies")
        elif has_rag:
            reason_parts.append(f"{yoe} YOE {title} with RAG/vector search experience")
        elif has_ml and has_product_exp:
            reason_parts.append(f"{yoe} YOE {title} building ML systems at product companies")
        elif has_ml:
            reason_parts.append(f"{yoe} YOE {title}; matches 'applied ML' profile in JD")
        else:
            reason_parts.append(f"{yoe} YOE {title}")

        if sig.get('recruiter_response_rate', 0) > 0.75:
            reason_parts.append("strong recent engagement")
        if 0 < sig.get('notice_period_days', 999) <= 30:
            reason_parts.append(f"can join in {sig['notice_period_days']}d")
        if prof.get('location','').lower() in ['bangalore', 'bengaluru', 'hyderabad', 'pune', 'mumbai']:
            reason_parts.append(f"{prof.get('location')}-based")

        concerns = []
        if sig.get('notice_period_days', 0) > 90:
            concerns.append(f"notice period ({sig['notice_period_days']} days)")
        if days_since(sig.get('last_active_date', '2000-01-01')) > 120:
            concerns.append("low recent activity")
        if sig.get('recruiter_response_rate', 1) < 0.3:
            concerns.append(f"low response rate ({int(sig['recruiter_response_rate']*100)}%)")

        if rank > 90 and not has_rag and not has_ml:
            reason_parts.append("adjacent skills only - included as final filler given engagement")
        elif concerns:
            reason_parts.append(f"some concern on {' & '.join(concerns)}")

        reason = '; '.join(reason_parts) + '.'
        final_results.append({
            'candidate_id': r['candidate_id'],
            'rank': rank,
            'score': r['score'],
            'reasoning': reason[:195]
        })
    return pd.DataFrame(final_results), YOE_MIN, YOE_MAX, GOOD_TITLES[0], len(results)

# UI Layout
st.markdown('## <div class="">', unsafe_allow_html=True)
col1, col2 = st.columns(2)
with col1:
    st.markdown(" <div style='text-align:center; font-size: 2rem; color: #00f5ff;'>📄 Job Description</div>", unsafe_allow_html=True)
    jd_file = st.file_uploader("Upload JD (.docx)", type=['docx'], label_visibility="collapsed")
with col2:
    st.markdown("<div style='text-align:center; font-size: 2rem; color: #00f5ff;'>👥 Candidate Database</div>", unsafe_allow_html=True)
    cand_file = st.file_uploader("Upload Candidates (.json)", type=['json', 'jsonl'], label_visibility="collapsed")
st.markdown('</div>', unsafe_allow_html=True)

if jd_file and cand_file:
    model = load_model()
    jd_text = extract_text_from_docx(jd_file)
    candidates = parse_candidates(cand_file)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"<div class='metric-card'><h3 style='text-align:center; font-size: 2rem; color: #00f5ff;'>{len(candidates)}</h3><p>Total Candidates</p></div>", unsafe_allow_html=True)
    with col2:
        st.markdown(f"<div class='metric-card'><h3 style='text-align:center; font-size: 2rem; color: #00f5ff;'>{len(jd_text)}</h3><p>JD Characters</p></div>", unsafe_allow_html=True)
    with col3:
        st.markdown(f"<div class='metric-card'><h3 style='text-align:center; font-size: 2rem; color: #00f5ff;'>50</h3><p>Top Results</p></div>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("🚀 INITIATE NEURAL RANKING", type="primary", use_container_width=True):
        start = time.time()
        with st.spinner("🧠 AI Engine Processing..."):
            df, yoe_min, yoe_max, jd_title, total_passed = run_ranking(candidates, jd_text, model, top_n=50)

        st.success(f"✅ Complete in {time.time() - start:.1f}s | Filtered: {total_passed}/{len(candidates)}")

        st.markdown("<div style=' font-size: 2rem; color: #00f5ff;'> 🏆 Top Ranked Candidates</div>", unsafe_allow_html=True)
        st.dataframe(df, use_container_width=True, height=600)

        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("📥 DOWNLOAD TALENTPULSE.CSV", csv, "TalentPulse.csv", "text/csv", use_container_width=True)

        st.markdown("<div style=' font-size: 2rem; color: #00f5ff;'> 🌟 Top Breakdown</div>", unsafe_allow_html=True)
        for _, row in df.head(5).iterrows():
            st.markdown(f"""
            <div style='background: rgba(123, 47, 247, 0.1); border-left: 4px solid #7b2ff7; padding: 1rem; margin: 0.5rem 0; border-radius: 8px;'>
                <span class='rank-badge'>RANK {row['rank']}</span>
                <strong style='color: #00f5ff; margin-left: 1rem;'>{row['candidate_id']}</strong>
                <span style='float: right; color: #f72585;'>Score: {row['score']}</span>
                <br><span style='color: #b0b0b0; font-size: 0.9rem;'>{row['reasoning']}</span>
            </div>
            """, unsafe_allow_html=True)
else:
    st.info("🎯 Upload Job Description + Candidate JSON to activate the AI ranking engine")