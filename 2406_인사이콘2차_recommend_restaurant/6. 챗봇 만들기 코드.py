#!/usr/bin/env python
# coding: utf-8

# <h1>Table of Contents<span class="tocSkip"></span></h1>
# <div class="toc"><ul class="toc-item"></ul></div>

# In[58]:


from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st
import numpy as np
import ast
import pandas as pd
import requests
import io

# 개별 파일 불러오기
df = pd.read_csv('2406_인사이콘2차_recommend_restaurant/final_df_0108.csv')

# 중복 제거
rests = df.drop_duplicates(subset=['restaurant_name', 'second_count_review', 'distance'])

# OpenAI API 키 설정
client = OpenAI(api_key = st.secrets["default"]["api_key"])

# 상단 제목
st.title("~~~UNMAGO~~~")

# 데이터프레임 초기화
if 'initial_data' not in st.session_state:
    st.session_state.initial_data = rests.copy()
if 'filtered_data' not in st.session_state:
    st.session_state.filtered_data = rests.copy()
if 'categories' not in st.session_state:
    st.session_state.categories = set()
if 'selected_categories' not in st.session_state:
    st.session_state.selected_categories = []

# 초기 데이터로 설정
st.session_state.filtered_data = st.session_state.initial_data.copy()

# 상황 선택
st.write("### 1. 무슨 상황인가요?")
situations = ["혼밥", '데이트 및 소개팅',  "회식", "카공", "맛집탐방"]
situation_choice = st.radio("TPO를 선택해 주세요", situations, key="situation_choice")

st.write(f"선택한 상황: {situation_choice}")

if situation_choice:
    df = st.session_state.initial_data.copy()

    if situation_choice == '카공':
        concentrate = df[df['집중하기 좋아요'] > 60]
        comfort = df[df['좌석이 편해요'] > 60]
        kagong = pd.concat([concentrate, comfort]).drop_duplicates()
        st.session_state.filtered_data = kagong[kagong['new_category'].isin(['커피/차', '디저트'])]
    
    elif situation_choice == '회식':
        drop_conditions = ['커피/차', '디저트', '간식', '도시락']
        df = df[~df['new_category'].isin(drop_conditions)]
        
        kokkiribab = df[df['양이 많아요'] > 40]
        partyppl = df[df['단체모임 하기 좋아요'] > 50]
        bigplace = df[df['매장이 넓어요'] > 50]

        intersection = pd.merge(kokkiribab, partyppl, how='inner')
        filtered = pd.merge(intersection, bigplace, how='inner')
        
        st.write("")
        st.write("### 1.5 몇명이서 회식하시나요?")
        meal_count = st.text_input('숫자를 적어주세요.', key="meal_count")
        if meal_count:
            meal_count = int(meal_count)
            st.write(f"{meal_count}명 이상 수용 가능한 식당으로 추천해드립니다.")
            filtered = filtered[filtered['max_seats'] >= meal_count]
        
        st.session_state.filtered_data = filtered
    
    elif situation_choice == '혼밥':
        price = df[df['가성비가 좋아요'] > 50]
        fast = df[df['음식이 빨리 나와요'] > 60]

        # 'price'와 'fast'의 합집합 생성
        combined = pd.concat([price, fast]).drop_duplicates()

        # 'alone' 데이터프레임 생성
        alone = df[df['혼밥하기 좋아요'] > 80]

        # 'combined'와 'alone'의 교집합 생성
        st.session_state.filtered_data = pd.merge(combined, alone, how='inner')
    
    elif situation_choice == '데이트 및 소개팅':
        special = df[df['특별한 메뉴가 있어요'] > 20]
        interior = df[df['인테리어가 멋져요'] > 40]
        goodday = df[df['특별한 날 가기 좋아요'] > 60]
        conversation = df[df['대화하기 좋아요'] > 60]

        date = pd.concat([special, interior, goodday])
        date = date.drop_duplicates()
        st.session_state.filtered_data = date[date['restaurant_name'].isin(conversation['restaurant_name'])]
                     
    elif situation_choice == '맛집탐방':
        mat = df[df['음식이 맛있어요'] > 50]
        special_menu = df[df['특별한 메뉴가 있어요'] > 25]
        meat = df[df['고기 질이 좋아요'] > 75]
        local = df[df['현지 맛에 가까워요'] > 85]

        taste = pd.concat([special_menu, mat, meat, local])
        taste = taste.drop_duplicates()
        st.session_state.filtered_data = taste[taste['restaurant_name'].isin(mat['restaurant_name'])]

# HTML을 사용하여 민트색 배경 박스 생성
html_code1 = f"""
<div style="background-color: #aaf0d1; padding: 10px; border-radius: 5px; position: -webkit-sticky; position: sticky; top: 0; z-index: 1000;">
    <p style="font-size: 14px; font-weight: normal; margin: 0; color: black;">현재 후보가 되는 가게의 갯수는 {len(st.session_state.filtered_data)}개 입니다.</p>
</div>
"""

# HTML 코드 표시
st.markdown(html_code1, unsafe_allow_html=True)


st.write("")
# 지하철역 선택
st.write("### 2. 어떤 역 근처가 좋으신가요?")
df = st.session_state.filtered_data
subways = set(df['station'])

subway_choices = st.multiselect("역을 선택해 주세요", subways, key="subway_choices")

if subway_choices:
    st.write(f"선택한 역: {', '.join(subway_choices)}")
    st.session_state.filtered_data = df[df['station'].isin(subway_choices)]

# HTML을 사용하여 민트색 배경 박스 생성
html_code4 = f"""
<div style="background-color: #aaf0d1; padding: 10px; border-radius: 5px; position: -webkit-sticky; position: sticky; top: 0; z-index: 1000;">
    <p style="font-size: 14px; font-weight: normal; margin: 0; color: black;">현재 후보가 되는 가게의 갯수는 {len(st.session_state.filtered_data)}개 입니다.</p>
</div>
"""

# HTML 코드 표시
st.markdown(html_code4, unsafe_allow_html=True)

st.write("")

# 서비스 선택
st.write("### 3. 원하는 서비스가 있으신가요?")
df = st.session_state.filtered_data
services = ["주차 가능 여부", "포장 가능 여부", "장애인 시설 유무", "무한 리필", "단체 이용 가능", "예약", "대기공간",'비건 메뉴 제공',"할랄푸드 제공","글루텐프리 메뉴 제공"]

service_choices = st.multiselect("서비스를 선택해 주세요", services, key="service_choices")

if service_choices:
    st.write(f"선택한 서비스: {', '.join(service_choices)}")
    condition = df[service_choices].eq(1).all(axis=1)
    st.session_state.filtered_data = df[condition]

# HTML을 사용하여 민트색 배경 박스 생성
html_code2 = f"""
<div style="background-color: #aaf0d1; padding: 10px; border-radius: 5px; position: -webkit-sticky; position: sticky; top: 0; z-index: 1000;">
    <p style="font-size: 14px; font-weight: normal; margin: 0; color: black;">현재 후보가 되는 가게의 갯수는 {len(st.session_state.filtered_data)}개 입니다.</p>
</div>
"""

# HTML 코드 표시
st.markdown(html_code2, unsafe_allow_html=True)


st.write("")

# 카테고리 선택
st.write('### 4. 원하는 카테고리가 있으신가요?')
df = st.session_state.filtered_data
st.session_state.categories = set(df['new_category'])
category_choices = st.multiselect("카테고리를 선택해 주세요", st.session_state.categories, key="category_choices")

if category_choices:
    st.write(f"선택한 카테고리: {', '.join(category_choices)}")
    st.session_state.selected_categories = category_choices
    st.session_state.filtered_data = df[df['new_category'].isin(st.session_state.selected_categories)]

# HTML을 사용하여 민트색 배경 박스 생성
html_code3 = f"""
<div style="background-color: #aaf0d1; padding: 10px; border-radius: 5px; position: -webkit-sticky; position: sticky; top: 0; z-index: 1000;">
    <p style="font-size: 14px; font-weight: normal; margin: 0; color: black;">현재 후보가 되는 가게의 갯수는 {len(st.session_state.filtered_data)}개 입니다.</p>
</div>
"""

# HTML 코드 표시
st.markdown(html_code3, unsafe_allow_html=True)

st.write("")
# 좋아하는 특징 적기
st.write('### 좋아하는 특징 적기')
user_input = st.text_input('(필수)좋아하는 음식점의 특징을 자유롭게 적어주세요. ex)매장이 깨끗함, 메뉴가 다양함', key="user_input")
if user_input:
    st.write('감사합니다!')

# 임베딩 함수 정의
def get_text_embedding(text, model="text-embedding-3-small"):
    response = client.embeddings.create(
        input=[text],
        model=model
    )
    embedding = response.data[0].embedding
    return embedding

## 추천 함수 정의
def recommend_documents(prompt, df, top_n=3):
    # 입력 텍스트 임베딩
    input_embedding = get_text_embedding(prompt)

    # 코사인 유사도 계산
    embeddings = np.vstack(df['임베딩결과값'].apply(lambda x: np.array(ast.literal_eval(x)) if isinstance(x, str) else x).values)
    similarities = cosine_similarity([input_embedding], embeddings)[0]

    # 유사도 높은 순으로 정렬하여 상위 n개 추천
    top_indices = similarities.argsort()[-top_n:][::-1]  # 유사도가 높은 순으로 인덱스 정렬

    # 상위 n개 추천 음식점 이름 리스트와 유사도
    recommended_restaurants = []
    for idx in top_indices:
        restaurant_info = {
            'restaurant_name': df.iloc[idx]['restaurant_name'],
            'new_category': df.iloc[idx]['new_category'],
            'review_summarized': df.iloc[idx]['review_summarized'],
            'info': df.iloc[idx]['info'],
            'menu_img': df.iloc[idx]['menu_img'].strip("[]").strip("'"),
            'menu_info': df.iloc[idx]['menu_info'],
            'business_hours': df.iloc[idx]['business_hours'],
            'phone_number': df.iloc[idx]['phone_number'],
            'station': df.iloc[idx]['station'],
            'meter': df.iloc[idx]['meter']
        }
        recommended_restaurants.append(restaurant_info)

    return recommended_restaurants

# 이미지 URL을 열어서 이미지를 반환하는 함수
def load_image(url):
    if url:  # URL이 빈 문자열이 아닌 경우에만 요청
        response = requests.get(url)
        return io.BytesIO(response.content)
    else:
        return None

# 추천 실행 버튼
if st.button("선택 완료(추천 받기 시작)", key="recommend_button"):
    if user_input:
        st.write('입력해주신 내용을 바탕으로 추천을 진행합니다...')
        # '임베딩결과값' 열이 있는지 확인
        if '임베딩결과값' not in st.session_state.filtered_data.columns:
            st.session_state.filtered_data['임베딩결과값'] = st.session_state.filtered_data['restaurant_name'].apply(lambda x: get_text_embedding(x))

        # 임베딩 결과값이 문자열로 저장된 경우 변환
        st.session_state.filtered_data['임베딩결과값'] = st.session_state.filtered_data['임베딩결과값'].apply(
            lambda x: np.array(ast.literal_eval(x)) if isinstance(x, str) else x)

        recommendation = recommend_documents(user_input, st.session_state.filtered_data)

        # 추천 결과 출력
        st.write("### 추천 음식점")
        top_n = min(3, len(recommendation))
        for i, rec in enumerate(recommendation[:top_n], 1):
            with st.expander(f"{rec['restaurant_name']} ({rec['new_category']})"):
                # 리뷰 요약 부분 처리
                review_summarized = rec['review_summarized']

                # 1., 2., 3.으로 나누기
                positive_review = ""
                negative_review = ""
                overall_review = ""

                # 1. 부터 2. 까지 긍정 부분, 2. 부터 3. 까지 부정 부분, 3. 이후 전체 요약 부분 추출
                if "1." in review_summarized:
                    positive_part = review_summarized.split("1.")[1]
                    if "2." in positive_part:
                        positive_review = positive_part.split("2.")[0].split(":", 1)[1].strip() if ":" in positive_part.split("2.")[0] else positive_part.split("2.")[0].strip()
                        negative_part = positive_part.split("2.")[1]
                        if "3." in negative_part:
                            negative_review = negative_part.split("3.")[0].split(":", 1)[1].strip() if ":" in negative_part.split("3.")[0] else negative_part.split("3.")[0].strip()
                            overall_part = negative_part.split("3.")[1]
                            overall_review = overall_part.split(":", 1)[1].strip() if ":" in overall_part else overall_part.strip()
                        else:
                            negative_review = negative_part.split(":", 1)[1].strip() if ":" in negative_part else negative_part.strip()
                    else:
                        positive_review = positive_part.split(":", 1)[1].strip() if ":" in positive_part else positive_part.strip()

                st.write(f"1. 긍정부분 요약:\n{positive_review}")
                st.write(f"2. 부정부분 요약:\n{negative_review}")
                st.write(f"3. 전체 요약:\n{overall_review}")

                st.write(f"전화번호:\n{rec['phone_number']}")
                st.write(f"{rec['station']}에서 {rec['meter']}m 거리에 있음")

            # 3개의 열을 생성
            col1, col2, col3 = st.columns(3)

            # 사장님의 한마디
            with col1:
                if not pd.isna(rec['info']):
                    with st.expander("사장님의 한마디"):
                        st.write(f"{rec['info']}")

            # 영업시간
            with col2:
                if not pd.isna(rec['business_hours']):
                    with st.expander(f"영업시간 - {rec['restaurant_name']}"):
                        business_hours = rec['business_hours']

                        # business_hours가 float 타입일 경우 빈 문자열로 처리
                        if isinstance(business_hours, float):
                            business_hours = ""

                        business_hours = business_hours.replace('접기', '').strip()

                        # '영업 종료', '영업 전', '영업 중', '영업 시작'으로 시작하는 경우
                        if any(business_hours.startswith(keyword) for keyword in ['영업 종료', '영업 전', '영업 중', '영업 시작']):
                            # '분에 브레이크타임', '분에 영업 시작', '분에 라스트오더', '분에 영업 종료' 위치 찾기
                            break_time_idx = business_hours.find('분에 브레이크타임')
                            end_time_idx = business_hours.find('분에 영업 종료')
                            start_time_idx = business_hours.find('분에 영업 시작')
                            last_order_idx = business_hours.find('분에 라스트오더')

                            # 유효한 위치 중 가장 작은 값을 찾음
                            if break_time_idx != -1:
                                business_hours = business_hours[break_time_idx + len('분에 브레이크타임'):].strip()
                            elif start_time_idx != -1:
                                business_hours = business_hours[start_time_idx + len('분에 영업 시작'):].strip()
                            elif last_order_idx != -1:
                                business_hours = business_hours[last_order_idx + len('분에 라스트오더'):].strip()
                            elif end_time_idx != -1:
                                business_hours = business_hours[end_time_idx + len('분에 영업 종료'):].strip()
                            else:
                                # 유효한 인덱스가 없는 경우 '영업 종료', '영업 전', '영업 중', '영업 시작' 제거
                                business_hours = business_hours.replace('영업 종료', '').replace('영업 전', '').replace('영업 중', '').replace('영업 시작', '').strip()

                        # 요일별로 각 줄에 하나씩 출력
                        days = ['월', '화', '수', '목', '금', '토', '일']
                        hours_by_day = {day: '' for day in days}
                        processed_days = set()  # 이미 처리된 요일을 저장할 집합

                        found_day = False
                        for day in days:
                            if day in business_hours and day not in processed_days:
                                start_idx = business_hours.find(day)

                                if day == '일':
                                    # '일'인 경우, '일요일'이나 '매일'을 피하기 위해 추가 검사를 수행
                                    while start_idx != -1:
                                        if start_idx == 0 or (business_hours[start_idx - 1] != '요' and business_hours[start_idx - 1] != '매'):
                                            break
                                        start_idx = business_hours.find(day, start_idx + 1)

                                if start_idx != -1:
                                    next_day_idx = len(business_hours)
                                    for next_day in days:
                                        if next_day != day and next_day not in processed_days:
                                            idx = business_hours.find(next_day, start_idx + len(day))
                                            # next_day가 '일'인 경우, 앞에 '요'나 '매'가 오지 않는지 확인
                                            while next_day == '일' and idx != -1:
                                                if idx == 0 or (business_hours[idx - 1] != '요' and business_hours[idx - 1] != '매'):
                                                    break
                                                idx = business_hours.find(next_day, idx + 1)
                                            if idx != -1:
                                                next_day_idx = min(next_day_idx, idx)
                                    hours_by_day[day] = business_hours[start_idx:next_day_idx].strip()
                                    processed_days.add(day)
                                    found_day = True

                        if found_day:
                            for day in days:
                                if hours_by_day[day]:
                                    st.write(hours_by_day[day])
                        else:
                            st.write(business_hours)





           # 메뉴판 보기
            with col3:
                with st.expander(f"메뉴판 보기 - {rec['restaurant_name']}"):
                    menu_items = ast.literal_eval(rec['menu_info'])
                    for item in menu_items:
                        if " 대표 " in item:
                            item = item.replace(" 대표 ", " ")
                            st.write(f"[대표] {item}")
                        else:
                            st.write(item)
                    image = load_image(rec['menu_img'])
                    if image:
                        st.image(image, caption="메뉴판", use_column_width=True)
    else:
        st.write("추천을 받기 위해 좋아하는 음식점의 특징을 입력해주세요.")


# In[ ]:




