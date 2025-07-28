import streamlit as st
import pandas as pd
import requests
from urllib.parse import quote
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer, util
from transformers import pipeline
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import plotly.express as px
from datetime import datetime
from io import BytesIO
from fpdf import FPDF
from collections import Counter
import re
import nltk

nltk.download('punkt')

# 🛠️ 모델 로딩
@st.cache_resource
def load_models():
    summarizer = SentenceTransformer("jhgan/ko-sroberta-multitask")
    sentiment_ko = pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment")
    return summarizer, sentiment_ko

summarizer, sentiment_ko = load_models()

# 📋 이모지 매핑
SENTI_EMOJI = {"긍정": "🟢", "부정": "🔴", "중립": "🟡"}
TONE_EMOJI = {"정보성": "ℹ️", "감정적": "💬", "분석적": "🧐"}

# 🌍 페이지 설정
st.set_page_config(page_title="🧠 AI 뉴스 요약 대시보드", layout="wide")
st.title("📰 AI/로봇 뉴스 요약 대시보드")

# 🎛️ 사이드바 입력
st.sidebar.header("🔎 뉴스 수집 조건")
KEYWORDS = ["AI", "로봇", "로봇감정", "로봇성격", "IT", "산업데이터", "데이터시스템"]
selected_keywords = st.sidebar.multiselect("💡 키워드 선택", KEYWORDS, default=KEYWORDS)
extra_kw = st.sidebar.text_input("➕ 추가 키워드 (쉼표 구분)")
if extra_kw:
    selected_keywords += [kw.strip() for kw in extra_kw.split(",") if kw.strip()]
lang_option = st.sidebar.radio("🌐 뉴스 언어", ["한국어", "영어"])
max_items = st.sidebar.slider("📰 키워드별 뉴스 수", 1, 15, 5)
start_date = st.sidebar.date_input("📅 시작일", None)
end_date = st.sidebar.date_input("📅 종료일", None)

# 📑 뉴스 본문 정제
def clean_text(html):
    return BeautifulSoup(html, "html.parser").get_text(separator=" ").strip()

# ✂️ 임베딩 기반 요약
def summarize(text, num_sent=3):
    if not text or len(text.strip()) < 30:
        return "요약 불가 (본문 부족)"
    sentences = [s.strip() for s in text.replace("!", ".").split(". ") if len(s.strip()) > 10]
    if len(sentences) <= num_sent:
        return text
    emb = summarizer.encode(sentences, convert_to_tensor=True)
    center = emb.mean(dim=0)
    scores = [util.pytorch_cos_sim(center, e)[0][0].item() for e in emb]
    top_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:num_sent]
    return ". ".join([sentences[i] for i in sorted(top_idx)])

# 🏷️ 한글 키워드 추출 (순수 파이썬, 불용어 포함)
KOREAN_STOPWORDS = {'있다', '하다', '수', '등', '및', '에서', '으로', '이번', '관한', '하여',
                   '대한', '관련', '한', '더', '있으며', '따라', '등의'}

def extract_keywords(text, n=5):
    words = re.findall(r"[가-힣]{2,}", text)
    words = [w for w in words if w not in KOREAN_STOPWORDS]
    freq = Counter(words)
    return ", ".join([w for w, _ in freq.most_common(n)]) if freq else "키워드 없음"

# 😶 감성 분석
def get_sentiment(text):
    try:
        result = sentiment_ko(text[:512])
        label = result[0]["label"]
        if "1" in label or "2" in label:
            return "부정"
        elif "4" in label or "5" in label:
            return "긍정"
        else:
            return "중립"
    except:
        return "중립"

# 🧐 콘텐츠 톤 분석
def analyze_tone(text):
    if len(text) > 1000 and any(word in text for word in ["우려", "논란", "환영", "논의", "과제"]):
        return "분석적"
    elif any(word in text for word in ["슬프다", "기쁘다", "충격", "감동", "분노"]):
        return "감정적"
    else:
        return "정보성"

# 🏷️ 태그 생성
def generate_tags(text):
    tags = []
    if "기술" in text:
        tags.append("#기술동향")
    if "시장" in text or "수요" in text:
        tags.append("#시장분석")
    if "논란" in text or "문제" in text:
        tags.append("#이슈")
    return " ".join(tags) if tags else "#일반"

# 💡 한줄평 생성 (이모지 포함)
def generate_opinion(sentiment, tone):
    senti_txt = {
        "긍정": "🟢 긍정적인 관점",
        "부정": "🔴 비판적인 관점",
        "중립": "🟡 중립적인 관점"
    }.get(sentiment, "🟡 중립적인 관점")
    tone_txt = {
        "정보성": "ℹ️ 정보 전달",
        "감정적": "💬 감정 표현",
        "분석적": "🧐 분석적 접근"
    }.get(tone, "ℹ️ 정보 전달")
    return f"{senti_txt} + {tone_txt}의 뉴스입니다."

# 📰 뉴스 본문 가져오기 (Pure Python: requests + bs4 + html5lib)
def get_article_text(url, lang="ko"):
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
        }
        resp = requests.get(url, headers=headers, timeout=5)
        resp.encoding = resp.apparent_encoding
        if resp.status_code != 200: return ""
        soup = BeautifulSoup(resp.text, "html5lib")

        # 대표 언론사 주요 본문 영역 시도
        body = ""
        selectors = [
            ("div", {"id": "newsct_article"}),
            ("div", {"id": "dic_area"}),
            ("div", {"class": "article_body"}),
            ("div", {"class": "newsEndContents"}),
            ("div", {"class": "art_body"}),
            ("div", {"class": "article-content"}),
            ("div", {"class": "content"}),
        ]
        for tag, attr in selectors:
            el = soup.find(tag, attrs=attr)
            if el:
                body = el.get_text(separator=" ", strip=True)
                break
        if not body:
            art = soup.find("article")
            if art:
                body = art.get_text(separator=" ", strip=True)
        if not body:
            p_tags = soup.find_all("p")
            body = " ".join([p.get_text(separator=" ", strip=True) for p in p_tags])
        if len(body) < 50:
            body = soup.get_text(separator=" ", strip=True)
        return " ".join(body.split())
    except Exception:
        return ""

import feedparser
# 🌐 뉴스 크롤링
@st.cache_data(show_spinner=True)
def fetch_news(keyword, lang="ko", max_items=5):
    q = quote(keyword)
    lang_code = "ko" if lang == "한국어" or lang == "ko" else "en"
    rss_url = f"https://news.google.com/rss/search?q={q}&hl={lang_code}&gl=KR&ceid=KR:{lang_code}"
    feed = requests.get(rss_url)
    if feed.status_code != 200:
        return pd.DataFrame()
    feed = feedparser.parse(feed.text)
    articles = []
    for entry in feed.entries[:max_items]:
        try:
            title = entry.title
            link = entry.link
            date = entry.get("published", datetime.now().strftime("%Y-%m-%d"))
            preview = clean_text(entry.summary if hasattr(entry, "summary") else "")
            fulltext = get_article_text(link, lang)
            if not fulltext or len(fulltext.strip()) < 50:
                fulltext = preview
            summary = summarize(fulltext)
            keywords = extract_keywords(fulltext)
            sentiment = get_sentiment(fulltext)
            tone = analyze_tone(fulltext)
            tags = generate_tags(fulltext)
            opinion = generate_opinion(sentiment, tone)
            articles.append({
                "키워드": keyword, "제목": title, "링크": link, "날짜": date,
                "본문": fulltext, "요약": summary, "키워드추출": keywords,
                "감성": sentiment, "콘텐츠톤": tone, "태그": tags, "한줄평": opinion
            })
        except Exception:
            continue
    return pd.DataFrame(articles)

# 🚀 뉴스 데이터 수집 및 필터링
lang_code = "ko" if lang_option == "한국어" else "en"
df_list = [fetch_news(k, lang=lang_code, max_items=max_items) for k in selected_keywords]
news_df = pd.concat(df_list).drop_duplicates(subset=["링크"]) if df_list else pd.DataFrame()

if not news_df.empty:
    news_df["날짜"] = pd.to_datetime(news_df["날짜"], errors="coerce")
    if start_date:
        news_df = news_df[news_df["날짜"] >= pd.to_datetime(start_date)]
    if end_date:
        news_df = news_df[news_df["날짜"] <= pd.to_datetime(end_date)]

# 🗂️ 탭 UI
tab1, tab2, tab3 = st.tabs(["📰 뉴스 목록", "📊 통계·워드클라우드", "📁 북마크/PDF"])

# 📰 탭1: 뉴스 목록
with tab1:
    st.subheader("📰 최신 뉴스 목록")
    if not news_df.empty:
        if "bookmarks" not in st.session_state:
            st.session_state["bookmarks"] = []
        for i, row in news_df.iterrows():
            senti_emo = SENTI_EMOJI.get(row["감성"], "🟡")
            tone_emo = TONE_EMOJI.get(row["콘텐츠톤"], "ℹ️")
            st.markdown(
                f"### {senti_emo}{tone_emo} [{row['제목']}]({row['링크']})"
            )
            st.caption(
                f"📅 {row['날짜'].date()} | {senti_emo} 감성: `{row['감성']}` | {tone_emo} 톤: `{row['콘텐츠톤']}` | {row['태그']}"
            )
            st.markdown(f"🧾 **요약:** {row['요약']}")
            st.markdown(f"💡 **한줄평:** {row['한줄평']}")
            st.markdown(f"🏷️ `{row['키워드추출']}`")
            if st.button("⭐ 북마크", key=f"bookmark_{hash(row['링크'])}"):
                if row["링크"] not in st.session_state["bookmarks"]:
                    st.session_state["bookmarks"].append(row["링크"])
            st.divider()
    else:
        st.info("뉴스 데이터가 없습니다.")

# 📊 탭2: 통계 및 워드클라우드
with tab2:
    st.subheader("📊 뉴스 통계 및 워드클라우드")
    if not news_df.empty:
        col1, col2 = st.columns([2, 1])
        with col1:
            st.markdown("#### 🔢 키워드별 뉴스 수")
            st.bar_chart(news_df["키워드"].value_counts())

            st.markdown("#### 😶 감성 분포")
            senti_bar = news_df["감성"].value_counts().rename_axis('감성').reset_index(name='건수')
            senti_bar["이모지"] = senti_bar["감성"].map(SENTI_EMOJI)
            st.dataframe(senti_bar, hide_index=True)
            st.bar_chart(news_df["감성"].value_counts())

            st.markdown("#### 🧐 콘텐츠 톤 분포")
            st.plotly_chart(px.histogram(news_df, x="키워드", color="콘텐츠톤", barmode="group"))

        with col2:
            st.markdown("#### ☁️ 워드클라우드")
            all_kws = ", ".join(news_df["키워드추출"].dropna())
            try:
                wc = WordCloud(width=400, height=300, background_color='white', font_path=None).generate(all_kws)
                fig, ax = plt.subplots()
                ax.imshow(wc, interpolation='bilinear')
                ax.axis("off")
                st.pyplot(fig)
            except Exception:
                st.warning("⚠️ 워드클라우드 생성 실패. 폰트 경로를 확인해주세요.")
    else:
        st.info("시각화할 데이터가 없습니다.")

# 📁 탭3: 북마크 & PDF
with tab3:
    st.subheader("📁 북마크 및 PDF 저장")
    bm_df = news_df[news_df["링크"].isin(st.session_state.get("bookmarks", []))]
    if not bm_df.empty:
        for _, row in bm_df.iterrows():
            st.markdown(
                f"- {SENTI_EMOJI.get(row['감성'],'🟡')}{TONE_EMOJI.get(row['콘텐츠톤'],'ℹ️')} "
                f"[{row['제목']}]({row['링크']})"
            )
        if st.button("⬇️ PDF 다운로드"):
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font('Arial', '', 12)
            pdf.cell(200, 10, "⭐ 북마크 뉴스 요약", 0, 1, 'C')
            for _, row in bm_df.iterrows():
                entry = (
                    f"{SENTI_EMOJI.get(row['감성'],'🟡')}{TONE_EMOJI.get(row['콘텐츠톤'],'ℹ️')} 제목: {row['제목']}\n"
                    f"요약: {row['요약']}\n"
                    f"감성: {row['감성']}, 톤: {row['콘텐츠톤']}\n"
                    f"한줄평: {row['한줄평']}\n"
                    f"키워드: {row['키워드추출']}\n"
                    f"태그: {row['태그']}\n"
                    f"링크: {row['링크']}\n"
                    + "-" * 40 + "\n"
                )
                pdf.multi_cell(0, 10, entry)
            temp = BytesIO()
            pdf.output(temp)
            temp.seek(0)
            st.download_button("📄 PDF 저장", data=temp.read(), file_name="bookmarked_news.pdf")
    else:
        st.info("⭐ 북마크된 뉴스가 없습니다.")
