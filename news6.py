import streamlit as st
import pandas as pd
import requests
from urllib.parse import quote, urlparse, parse_qs
from bs4 import BeautifulSoup
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import plotly.express as px
from datetime import datetime
from io import BytesIO
from collections import Counter
import re
import feedparser

# 페이지 설정
st.set_page_config(page_title="🧠 AI 뉴스 요약 대시보드", layout="wide")
st.title("📰 AI/로봇 뉴스 요약 대시보드")

# 사이드바 설정
st.sidebar.header("🔎 뉴스 수집 조건")
KEYWORDS = ["AI", "로봇", "로봇감정", "로봇성격", "IT", "산업데이터", "데이터시스템"]
selected_keywords = st.sidebar.multiselect("💡 키워드 선택", KEYWORDS, default=KEYWORDS[:3])
extra_kw = st.sidebar.text_input("➕ 추가 키워드 (쉼표 구분)")
if extra_kw:
    selected_keywords += [kw.strip() for kw in extra_kw.split(",") if kw.strip()]
lang_option = st.sidebar.radio("🌐 뉴스 언어", ["한국어", "영어"])
max_items = st.sidebar.slider("📰 키워드별 뉴스 수", 1, 10, 3)
start_date = st.sidebar.date_input("📅 시작일", None)
end_date = st.sidebar.date_input("📅 종료일", None)

# 이모지 설정
SENTI_EMOJI = {"긍정": "🟢", "부정": "🔴", "중립": "🟡"}
TONE_EMOJI = {"정보성": "ℹ️", "감정적": "💬", "분석적": "🧐"}

def clean_text(html):
    """HTML 태그 제거"""
    return BeautifulSoup(html, "html.parser").get_text(separator=" ").strip()

def simple_summarize(text, num_sent=2):
    """간단한 요약 (AI 모델 없이)"""
    if not text or len(text.strip()) < 30:
        return "요약 불가 (본문 부족)"
    
    # 문장 분리
    sentences = [s.strip() for s in text.replace("!", ".").split(". ") if len(s.strip()) > 15]
    
    if len(sentences) <= num_sent:
        return text[:300] + "..." if len(text) > 300 else text
    
    # 첫 번째와 중간 문장 선택
    selected = [sentences[0]]
    if len(sentences) > 2:
        selected.append(sentences[len(sentences)//2])
    
    return ". ".join(selected)

def extract_keywords(text, n=5):
    """키워드 추출 (간단한 빈도 기반)"""
    KOREAN_STOPWORDS = {'있다', '하다', '수', '등', '및', '에서', '으로', '이번',
                        '관한', '하여', '대한', '관련', '한', '더', '있으며', '따라', '등의'}
    
    words = re.findall(r"[가-힣]{2,}", text)
    words = [w for w in words if w not in KOREAN_STOPWORDS and len(w) > 1]
    freq = Counter(words)
    return ", ".join([w for w, _ in freq.most_common(n)]) if freq else "키워드 없음"

def simple_sentiment(text):
    """간단한 감성 분석 (키워드 기반)"""
    positive_words = ['좋다', '훌륭', '성공', '발전', '혁신', '개선', '증가', '상승', '긍정']
    negative_words = ['나쁘다', '문제', '실패', '우려', '논란', '감소', '하락', '부정', '위험']
    
    pos_count = sum(1 for word in positive_words if word in text)
    neg_count = sum(1 for word in negative_words if word in text)
    
    if pos_count > neg_count:
        return "긍정"
    elif neg_count > pos_count:
        return "부정"
    else:
        return "중립"

def analyze_tone(text):
    """콘텐츠 톤 분석"""
    if any(word in text for word in ["분석", "연구", "조사", "데이터", "통계"]):
        return "분석적"
    elif any(word in text for word in ["놀라", "충격", "감동", "기쁘", "슬프"]):
        return "감정적"
    else:
        return "정보성"

def generate_tags(text):
    """태그 생성"""
    tags = []
    if "기술" in text or "AI" in text:
        tags.append("#기술동향")
    if "시장" in text or "수요" in text:
        tags.append("#시장분석")
    if "논란" in text or "문제" in text:
        tags.append("#이슈")
    return " ".join(tags) if tags else "#일반"

def generate_opinion(sentiment, tone):
    """한줄평 생성"""
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

@st.cache_data(show_spinner=True, ttl=3600)
def fetch_news(keyword, lang="ko", max_items=3):
    """뉴스 수집"""
    try:
        q = quote(keyword)
        lang_code = "ko" if lang == "한국어" or lang == "ko" else "en"
        rss_url = f"https://news.google.com/rss/search?q={q}&hl={lang_code}&gl=KR&ceid=KR:{lang_code}"
        
        response = requests.get(rss_url, timeout=10)
        if response.status_code != 200:
            return pd.DataFrame()
        
        feed = feedparser.parse(response.text)
        articles = []
        
        for entry in feed.entries[:max_items]:
            try:
                title = entry.title
                # Google News 리다이렉트 링크 처리
                original_link = entry.link
                
                # Google News 링크에서 실제 URL 추출 시도
                if 'news.google.com' in original_link and 'url=' in original_link:
                    try:
                        # URL 파라미터에서 실제 링크 추출
                        parsed = urlparse(original_link)
                        query_params = parse_qs(parsed.query)
                        if 'url' in query_params:
                            link = query_params['url'][0]
                        else:
                            link = original_link
                    except:
                        link = original_link
                else:
                    link = original_link
                
                date = entry.get("published", datetime.now().strftime("%Y-%m-%d"))
                preview = clean_text(entry.summary if hasattr(entry, "summary") else "")
                
                # 본문 대신 요약문 사용 (속도 향상)
                fulltext = preview if preview else title
                
                summary = simple_summarize(fulltext)
                keywords = extract_keywords(fulltext)
                sentiment = simple_sentiment(fulltext)
                tone = analyze_tone(fulltext)
                tags = generate_tags(fulltext)
                opinion = generate_opinion(sentiment, tone)
                
                articles.append({
                    "키워드": keyword,
                    "제목": title,
                    "링크": link,
                    "날짜": date,
                    "본문": fulltext,
                    "요약": summary,
                    "키워드추출": keywords,
                    "감성": sentiment,
                    "콘텐츠톤": tone,
                    "태그": tags,
                    "한줄평": opinion
                })
            except Exception as e:
                st.warning(f"뉴스 처리 중 오류: {e}")
                continue
                
        return pd.DataFrame(articles)
    except Exception as e:
        st.error(f"뉴스 수집 실패: {e}")
        return pd.DataFrame()

# 메인 실행 부분
if st.button("🔍 뉴스 수집 시작", type="primary"):
    if not selected_keywords:
        st.warning("키워드를 선택해주세요!")
    else:
        lang_code = "ko" if lang_option == "한국어" else "en"
        
        with st.spinner("뉴스를 수집하고 있습니다..."):
            df_list = []
            progress_bar = st.progress(0)
            
            for i, keyword in enumerate(selected_keywords):
                df = fetch_news(keyword, lang=lang_code, max_items=max_items)
                if not df.empty:
                    df_list.append(df)
                progress_bar.progress((i + 1) / len(selected_keywords))
            
            progress_bar.empty()
            
            if df_list:
                news_df = pd.concat(df_list, ignore_index=True).drop_duplicates(subset=["링크"])
                st.session_state['news_df'] = news_df
                st.success(f"총 {len(news_df)}개의 뉴스를 수집했습니다!")
            else:
                st.warning("수집된 뉴스가 없습니다.")
                st.session_state['news_df'] = pd.DataFrame()

# 세션에서 뉴스 데이터 가져오기
news_df = st.session_state.get('news_df', pd.DataFrame())

# 날짜 필터링
if not news_df.empty:
    news_df["날짜"] = pd.to_datetime(news_df["날짜"], errors="coerce")
    if start_date:
        news_df = news_df[news_df["날짜"] >= pd.to_datetime(start_date)]
    if end_date:
        news_df = news_df[news_df["날짜"] <= pd.to_datetime(end_date)]

# 탭 생성
tab1, tab2, tab3 = st.tabs(["📰 뉴스 목록", "📊 통계·워드클라우드", "📁 북마크/PDF"])

with tab1:
    st.subheader("📰 최신 뉴스 목록")
    if not news_df.empty:
        if "bookmarks" not in st.session_state:
            st.session_state["bookmarks"] = []
            
        for i, row in news_df.iterrows():
            senti_emo = SENTI_EMOJI.get(row["감성"], "🟡")
            tone_emo = TONE_EMOJI.get(row["콘텐츠톤"], "ℹ️")
            
            # 제목과 링크를 안전하게 표시
            st.markdown(f"### {senti_emo}{tone_emo} {row['제목']}")
            st.markdown(f"🔗 **링크:** {row['링크']}")
            st.caption(f"📅 {row['날짜'].date()} | {senti_emo} 감성: `{row['감성']}` | {tone_emo} 톤: `{row['콘텐츠톤']}` | {row['태그']}")
            st.markdown(f"🧾 **요약:** {row['요약']}")
            st.markdown(f"💡 **한줄평:** {row['한줄평']}")
            st.markdown(f"🏷️ `{row['키워드추출']}`")
            
            if st.button("⭐ 북마크", key=f"bookmark_{i}"):
                if row["링크"] not in st.session_state["bookmarks"]:
                    st.session_state["bookmarks"].append(row["링크"])
                    st.success("북마크에 추가되었습니다!")
            st.divider()
    else:
        st.info("🔍 뉴스 수집 시작 버튼을 클릭하여 뉴스를 가져오세요!")

with tab2:
    st.subheader("📊 뉴스 통계 및 워드클라우드")
    if not news_df.empty:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("#### 🔢 키워드별 뉴스 수")
            keyword_counts = news_df["키워드"].value_counts()
            st.bar_chart(keyword_counts)
            
            st.markdown("#### 😶 감성 분포")
            sentiment_counts = news_df["감성"].value_counts()
            st.bar_chart(sentiment_counts)
            
            st.markdown("#### 🧐 콘텐츠 톤 분포")
            fig = px.histogram(news_df, x="키워드", color="콘텐츠톤", barmode="group")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("#### ☁️ 워드클라우드")
            all_keywords = ", ".join(news_df["키워드추출"].dropna())
            
            if all_keywords:
                try:
                    wc = WordCloud(
                        width=400, 
                        height=300, 
                        background_color='white',
                        max_words=50
                    ).generate(all_keywords)
                    
                    fig, ax = plt.subplots(figsize=(6, 4))
                    ax.imshow(wc, interpolation='bilinear')
                    ax.axis("off")
                    st.pyplot(fig)
                except Exception as e:
                    st.warning(f"워드클라우드 생성 실패: {e}")
            else:
                st.info("워드클라우드를 생성할 데이터가 없습니다.")
    else:
        st.info("통계를 표시할 데이터가 없습니다.")

with tab3:
    st.subheader("📁 북마크 및 PDF 저장")
    
    bookmarked_links = st.session_state.get("bookmarks", [])
    if bookmarked_links and not news_df.empty:
        bm_df = news_df[news_df["링크"].isin(bookmarked_links)]
        
        if not bm_df.empty:
            st.write(f"📌 북마크된 뉴스: {len(bm_df)}개")
            
            for _, row in bm_df.iterrows():
                senti_emo = SENTI_EMOJI.get(row['감성'], '🟡')
                tone_emo = TONE_EMOJI.get(row['콘텐츠톤'], 'ℹ️')
                st.markdown(f"- {senti_emo}{tone_emo} **{row['제목']}**")
                st.markdown(f"  🔗 {row['링크']}")
                st.markdown("---")
            
            # CSV 다운로드로 변경 (한글 지원)
            if st.button("⬇️ CSV 다운로드"):
                try:
                    # CSV 형태로 데이터 준비
                    csv_data = bm_df[['키워드', '제목', '요약', '감성', '콘텐츠톤', '키워드추출', '태그', '한줄평', '링크']].copy()
                    csv_string = csv_data.to_csv(index=False, encoding='utf-8-sig')
                    
                    st.download_button(
                        "📄 CSV 파일 저장",
                        data=csv_string.encode('utf-8-sig'),
                        file_name=f"bookmarked_news_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
                    st.success("CSV 파일이 준비되었습니다!")
                except Exception as e:
                    st.error(f"CSV 생성 실패: {e}")
            
            # 텍스트 파일 다운로드 추가
            if st.button("📝 텍스트 파일 다운로드"):
                try:
                    text_content = "=== 북마크된 뉴스 요약 ===\n\n"
                    for i, row in bm_df.iterrows():
                        text_content += f"""
📰 제목: {row['제목']}
🔗 링크: {row['링크']}
📅 날짜: {row['날짜']}
🧾 요약: {row['요약']}
💭 한줄평: {row['한줄평']}
😶 감성: {row['감성']} | 🧐 톤: {row['콘텐츠톤']}
🏷️ 키워드: {row['키워드추출']}
🏷️ 태그: {row['태그']}

{'='*60}

"""
                    
                    st.download_button(
                        "📝 텍스트 파일 저장",
                        data=text_content.encode('utf-8'),
                        file_name=f"bookmarked_news_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                        mime="text/plain"
                    )
                    st.success("텍스트 파일이 준비되었습니다!")
                except Exception as e:
                    st.error(f"텍스트 파일 생성 실패: {e}")
        else:
            st.info("북마크된 뉴스가 현재 데이터에 없습니다.")
    else:
        st.info("⭐ 북마크된 뉴스가 없습니다.")