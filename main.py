import json, time, re
from datetime import datetime
import pandas as pd
from sentence_transformers import SentenceTransformer
import numpy as np
from tqdm import tqdm
import pickle, os

TEAM_ID = "TalentPulse"
INPUT_FILE = "candidates.jsonl"
JD_FILE = "job_description.md"
OUTPUT_FILE = f"{TEAM_ID}.csv"
EMB_FILE = "candidate_embeddings.pkl"

print("Loading model...")
model = SentenceTransformer('all-MiniLM-L6-v2')

print("Loading candidates...")
with open(INPUT_FILE, "r", encoding="utf-8") as f:
    candidates = [json.loads(line) for line in f]
print(f"Loaded {len(candidates)} candidates")

print("Loading JD...")
with open(JD_FILE, 'r') as f:
    jd_text = f.read()

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

    # UPDATED: AI/ML titles ko priority
    good_titles = ['ai engineer', 'ml engineer', 'machine learning engineer', 'data scientist',
                   'nlp engineer', 'computer vision engineer', 'research scientist', 'applied scientist']
    jd_title_match = re.search(r'Job Description:\s*([A-Za-z\s]+)', jd, re.IGNORECASE)
    if jd_title_match:
        jd_title = jd_title_match.group(1).lower().strip()
        if any(kw in jd_title for kw in ['ai', 'ml', 'machine learning', 'data', 'nlp', 'research']):
            good_titles.insert(0, jd_title)
    good_titles = list(dict.fromkeys(good_titles))

    # UPDATED: Better keywords for ML/RAG
    stop_words = {'the','and','for','with','years','experience','role','job','work','team','company','business','strong','good','excellent','ability','must','should','required','open','type','time','series','engineer'}
    words = re.findall(r'\b[a-z]{4,}\b', jd_lower)
    keywords = list(dict.fromkeys([w for w in words if w not in stop_words]))[:20]
    # Force add critical ML keywords
    keywords.extend(['rag', 'recommendation', 'vector', 'embedding', 'llm', 'machine', 'learning'])
    keywords = list(dict.fromkeys(keywords))
    return yoe_min, yoe_max, bad_titles, good_titles, keywords

YOE_MIN, YOE_MAX, BAD_TITLES, GOOD_TITLES, KEYWORDS = extract_from_jd(jd_text)
jd_emb = model.encode(jd_text)
print(f"JD Parsed: YOE {YOE_MIN}-{YOE_MAX} | Bad Titles: {len(BAD_TITLES)} | Good: {GOOD_TITLES[0]} | Keywords: {len(KEYWORDS)}")

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

def is_disqualified(cand, sig):
    if is_honeypot(cand): return True
    prof = cand.get('profile', {})
    title = prof.get('current_title', '').lower()
    yoe = prof.get('years_of_experience', 0)
    if yoe < YOE_MIN or yoe > YOE_MAX: return True
    if any(bad in title for bad in BAD_TITLES): return True

    # UPDATED: Soft disqualify for non-engineers when JD wants engineer
    if 'engineer' in GOOD_TITLES[0] and not any(eng in title for eng in ['engineer', 'developer', 'scientist', 'architect']):
        return True

    companies = [c.get('company_name', '').lower() for c in cand.get('career_history', [])]
    consulting_firms = ['tcs', 'infosys', 'wipro', 'accenture', 'cognizant', 'capgemini']
    if len(companies) > 0 and all(any(cf in comp for cf in consulting_firms) for comp in companies):
        return True
    if days_since(sig.get('last_active_date', '2000-01-01')) > 180: return True
    if sig.get('recruiter_response_rate', 1) < 0.05: return True # Changed from 0.1 to 0.05
    return False

if os.path.exists(EMB_FILE):
    print("Loading cached embeddings... ⚡")
    with open(EMB_FILE, "rb") as f:
        cand_emb = pickle.load(f)
else:
    print("Encoding all candidates...")
    texts = [build_candidate_text(c) for c in candidates]
    cand_emb = model.encode(texts, batch_size=512, show_progress_bar=True)
    with open(EMB_FILE, "wb") as f:
        pickle.dump(cand_emb, f)

print("Ranking candidates...")
start_time = time.time()
semantic_scores = np.dot(cand_emb, jd_emb)
results = []

# PASS 1: Score all candidates - UPDATED WEIGHTS
for i, cand in enumerate(tqdm(candidates)):
    sig = cand.get('redrob_signals', {})
    if is_disqualified(cand, sig): continue

    prof = cand.get('profile', {})
    title = prof.get('current_title', '').lower()

    # UPDATED: Title boost - AI/ML titles get more
    title_boost = 0.15 if any(good in title for good in GOOD_TITLES[:4]) else 0.05 if any(good in title for good in GOOD_TITLES) else 0

    # UPDATED: Relevance scoring - MAJOR CHANGES
    career_text = ' '.join([j.get('description','').lower() + ' ' + j.get('company_name','').lower() for j in cand.get('career_history', [])[:3]])
    skills_text = ' '.join([s['name'].lower() for s in cand.get('skills', [])])
    combined_text = career_text + ' ' + skills_text + ' ' + title

    # Product Experience Boost - INCREASED
    product_companies = ['google', 'meta', 'amazon', 'microsoft', 'flipkart', 'swiggy', 'zomato', 'paytm', 'uber', 'ola', 'cred', 'product']
    has_product_exp = 0.25 if any(pc in career_text for pc in product_companies) else 0

    # RAG/ML Work Boost - INCREASED + MORE KEYWORDS
    ml_keywords = ['rag', 'retrieval', 'vector', 'embedding', 'recommendation', 'search', 'llm', 'machine learning', 'deep learning', 'neural', 'nlp', 'computer vision']
    has_rag_work = 0.30 if any(kw in combined_text for kw in ml_keywords) else 0

    # Title Penalty - Software Engineer without ML gets penalized
    title_penalty = -0.35 if any(t in title for t in ['civil', 'mechanical', 'electrical', 'frontend', 'backend']) else -0.20 if 'software engineer' in title and not has_rag_work else 0

    # NEW: JD Keyword match bonus
    keyword_matches = sum(1 for kw in KEYWORDS if kw in combined_text)
    keyword_boost = min(keyword_matches * 0.05, 0.20) # Max 0.20 boost

    relevance_boost = has_product_exp + has_rag_work + title_penalty + keyword_boost

    # UPDATED WEIGHTS: Semantic 25% + Signals 25% + Relevance 50%
    final = semantic_scores[i] * 0.25 + score_redrob_signals(sig) * 0.25 + relevance_boost * 0.5 + title_boost

    results.append({
        'candidate_id': cand['candidate_id'],
        'score': round(float(final), 6),
        'cand_obj': cand,
        'sig': sig
    })

results.sort(key=lambda x: (-x['score'], x['candidate_id']))
if len(results) < 100:
    raise ValueError(f"Only {len(results)} candidates passed filters. Need 100.")

top_100 = results[:100]

# PASS 2: Generate reasoning - UPDATED
final_results = []
for rank, r in enumerate(top_100, 1):
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

assert len(final_results) == 100
assert all(final_results[i]['score'] >= final_results[i+1]['score'] for i in range(99))
assert len(set(r['candidate_id'] for r in final_results)) == 100
assert final_results[0]['rank'] == 1 and final_results[-1]['rank'] == 100

df = pd.DataFrame(final_results)[['candidate_id', 'rank', 'score', 'reasoning']]
df.to_csv(OUTPUT_FILE, index=False, encoding='utf-8')
print(f"Done in {time.time() - start_time:.1f}s | Saved {OUTPUT_FILE}")
print(f"Top 3: {df.head(3)[['candidate_id', 'rank', 'score']].to_dict('records')}")