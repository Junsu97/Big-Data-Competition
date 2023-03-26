import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
import requests
from urllib.parse import quote
import json

# 한글 폰트 경로
font_path = 'C:/Windows/Fonts/malgun.ttf'
# 폰트 설정
font_name = fm.FontProperties(fname=font_path).get_name()
plt.rc('font', family=font_name)

#csv 파일 불러오기
df = pd.read_csv('C:\DB\주민등록인구(연령별_동별)_20230317170259.csv', encoding='utf-8-sig')

#필요한 항목 추출
filtered_df = df[df['항목'] == '계']
selected_columns = ["동별(3)", "2020", "2020.1"]
data = filtered_df[selected_columns]
row_data = data.iloc[0:24]

#컬럼 명 변경
row_data = row_data.rename(columns={"2020": "10~14세", "2020.1": "15~19세","동별(3)":"동별"})
row_data['10~14세'] = row_data['10~14세'].astype(int)
row_data['15~19세'] = row_data['15~19세'].astype(int)

#새로운 컬럼을 만들어 값을 합침
row_data['10~19세'] = row_data['10~14세'] + row_data['15~19세']



#x,y축 값 설정 및 폰트 크기
bars = row_data.plot(x = '동별',y ='10~19세',kind='bar',fontsize=10)
plt.xticks(rotation = 0)
plt.xlabel('행정동')
plt.ylabel('인구수')

# 막대 위에 숫자 출력
for i, v in enumerate(row_data['10~19세']):
    bars.text(i, v, str(v), ha='center', va='bottom', fontsize=10)

plt.show()

##########################################################################

df_2 = pd.read_excel("C:\DB\서울특별시 강서구_기초생활수급자 연령별, 성별 현황_20200531.xls")#,encoding='utf-8-sig')
selected_columns = ["Unnamed: 2","Unnamed: 8","Unnamed: 9","Unnamed: 10"]
data = df_2[selected_columns]

#13~29세 합침(성별 통합)
data = data.rename(columns={"Unnamed: 2": "동별","Unnamed: 8":"13~15세","Unnamed: 9":"16~18세","Unnamed: 10":"19~29세"})
data['13~15세'] = data['13~15세']
data['16~18세'] = data['16~18세']
data['19~29세'] = data['19~29세']
# 새로운 컬럼 생성
data['13~29세 합계'] = data['13~15세'] + data['16~18세'] + data['19~29세']

data['동별'] = data['동별'].replace(method='ffill')
data = data.fillna(0)

# '합계' 열을 선택하여 성별을 기준으로 합계를 계산하여 그래프에 표시
data.reset_index(drop=True, inplace=True)

# 8행부터 160행까지의 데이터만 선택
selected_data = data.iloc[8:160]

# 동별 기준으로 그룹화하여 합계 계산
grouped_data = selected_data.groupby(['동별'], as_index=False).sum()

# 막대 그래프 그리기
fig, ax = plt.subplots(figsize=(10, 8))
bars_2 = grouped_data.plot(x='동별', y='13~29세 합계', kind='bar', ax=ax)
plt.xticks(rotation = 0)

# 막대 그래프 위에 수치 표현
for bar in bars_2.containers:
    ax.bar_label(bar, label_type='edge')
    
plt.show()

###############################################################################



#Kakao API 키
KAKAO_REST_API_KEY = 'db32ff4274e4fd4b838b7d0e01bd05c1'

# 주소를 좌표로 변환하는 함수
def get_location(address):
    url = 'https://dapi.kakao.com/v2/local/search/address.json'
    headers = {'Authorization': 'KakaoAK {api_key}'.format(api_key = KAKAO_REST_API_KEY)}
    params = {'query': address}
    response = requests.get(url, headers=headers, params=params)
    response.raise_for_status()

    result = response.json()['documents'][0] if response.json()['documents'] else None
    if result:
        return result['y'], result['x'], result['address']['region_3depth_h_name']
    else:
        print(f"No result for {address}")
        return None


#Kakao API를 이용해 좌표를 행정동으로 변환하는 함수
def get_region(y, x):
    url = 'https://dapi.kakao.com/v2/local/geo/coord2regioncode.json'
    headers = {'Authorization': 'KakaoAK {api_key}'.format(api_key = KAKAO_REST_API_KEY)}
    params = {'y': y, 'x': x}
    response = requests.get(url, headers=headers, params=params)
    response.raise_for_status()

    result = response.json().get('documents', [])
    if result:
        return result[0]['region_3depth_name']
    else:
        return None


# 데이터 불러오기
df_3 = pd.read_excel("C:\DB\서울시강서구학원교습소정보.xls")

# '교습계열명'이 '보통교와'인 데이터만 추출하여 새로운 데이터프레임 생성
df_boto = df_3[df_3['교습계열명'] == '보통교과']
# xx로 기준으로 학원 수 세기
counts = df_boto.groupby(['도로명주소', '학원명']).size().reset_index(name='count')

addresses = counts['도로명주소'].tolist()

# 주소를 좌표로 변환하여 행정동 정보 가져오기
regions = []
for address in addresses:
    result = get_location(address)
    if result is not None:
        y, x, region_h = result
        if region_h is not None:
            region = region_h
        else:
            region = get_region(y, x)
            if region is None:
                region = "-"
        regions.append(region)
    else:
        regions.append("-")

# 변환된 행정동 정보로 데이터프레임 생성
df2 = pd.DataFrame({'address': addresses, 'region': regions, 'count': counts['count']})
df2['region'] = df2['region'].fillna('-')
# '가양2동' 추가
df2.loc[len(df2)] = ['address_value', '가양2동', 0]

total_count = df2['count'].sum()
print('전체 학원 수:', total_count)

############### 순위####################
# 데이터프레임 그룹화하여 각 행정동별 학원 수 구하기
df2 = df2.groupby('region')['count'].sum().reset_index()

# 학원 수에 따른 오름차순 정렬
df2 = df2.sort_values(by=['count'], ascending=False)

# 순위 계산하여 rank 컬럼에 저장
df2['rank'] = df2['count'].rank(ascending=False, method='dense').astype(int)

# 순위 출력
row = []
for idx, row in df2.iterrows():
    if row['count'] == 14 :
        print(f"공동{row['rank']}위: {row['region']} ({row['count']}개)")
    else:
        print(f"{row['rank']}위: {row['region']} ({row['count']}개)")
        

##################################################################


# 시각화하기
fig, ax = plt.subplots(figsize=(20,5))
df2 = df2.groupby('region')['count'].sum().reset_index()
bars = plt.bar(df2['region'], df2['count'])

plt.xticks(rotation=0)
plt.xlabel('행정동')
plt.ylabel('학원 수')

print(df2.columns)

# y축 범위 설정
max_count = df2['count'].max()
plt.ylim(0, max_count)

for bar in bars:
    ax.annotate(format(bar.get_height()), 
                 xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()), 
                 xytext=(0, 3),
                 textcoords="offset points",
                 ha='center', va='bottom')

plt.show()





#######################################################################
# 기초생활 + 학생수 그래프


fig, ax = plt.subplots(figsize=(15,5))
ax2 = ax.twinx()

# 막대 그래프
bars = row_data.plot(x='동별', y='10~19세', kind='bar', fontsize=10, ax=ax)

# 선 그래프
line = grouped_data.plot(x='동별', y='13~29세 합계', kind='line', ax=ax2, color='red')

# y 축 레이블 설정
ax.set_ylabel('10~19세')
ax2.set_ylabel('13~29세 합계')

# 범례 설정
lines, labels = ax.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax.legend(lines + lines2, labels + labels2, loc='upper left')

plt.xticks(rotation = 0)
plt.show()

######################################################################

#10~19세 동별 통계 순위

# csv 파일 불러오기
df = pd.read_csv('C:\DB\주민등록인구(연령별_동별)_20230317170259.csv', encoding='utf-8-sig')

# 필요한 항목 추출
filtered_df = df[df['항목'] == '계']
selected_columns = ["동별(3)", "2020", "2020.1"]
data = filtered_df[selected_columns]
row_data = data.iloc[0:24]

# 컬럼 명 변경
row_data = row_data.rename(columns={"2020": "10~14세", "2020.1": "15~19세", "동별(3)":"동별"})
row_data['10~14세'] = row_data['10~14세'].astype(int)
row_data['15~19세'] = row_data['15~19세'].astype(int)

# 새로운 컬럼을 만들어 값을 합침
row_data['10~19세'] = row_data['10~14세'] + row_data['15~19세']

# 인구수 기준 오름차순 정렬
sorted_data = row_data.sort_values(by='10~19세', ascending=True)

# 인덱스 리셋
sorted_data = sorted_data.reset_index(drop=True)

# 인구수가 높은 순위 표 만들기
rank = []
for i in range(len(sorted_data)):
    rank.append(i+1)
sorted_data.insert(0, "순위", rank)
table = sorted_data[["순위", "동별", "10~19세"]]


# 출력
print(table)
##########################################################
#기초생활 수급자

grouped_data = grouped_data.sort_values(by=['13~29세 합계'], ascending=True)

# '순위' 컬럼 추가
grouped_data['순위'] = range(1, len(grouped_data) + 1)

# '순위', '동별', '13~29세 합계' 컬럼만 선택하여 출력
result = grouped_data[['순위', '동별', '13~29세 합계']]
print(result)


#################################################################################

df2['rank'] = df2['count'].rank(ascending=False, method='dense').astype(int)
df2= df2.rename(columns={"region":"동별","rank":"순위"})
print(df2.columns)
# '동별'로 묶어 데이터 결합
merged_df = pd.merge(table[['동별', '순위']], result[['동별', '순위']], on='동별')
merged_df = pd.merge(merged_df, df2[['동별', '순위']], on='동별')

# score 컬럼 생성
merged_df['score'] = merged_df['순위_x'] + merged_df['순위_y'] + merged_df['순위']

# 오름차순 정렬
merged_df = merged_df.sort_values('score')

# 인덱스초기화
merged_df = merged_df.reset_index(drop=True)

# 막대그래프
plt.bar(merged_df['동별'], merged_df['score'])

# 라벨과 타이틀 설정
plt.xlabel('동별')
plt.ylabel('종합점수')
plt.title('각 동별 종합점수')

# 막대그래프에 수치표현
for i, score in enumerate(merged_df['score']):
    plt.text(i, score+1, str(score), ha='center', fontsize=10)


plt.show()
# 동별로 점수를 기준으로 순위 출력 & 공동순위 표시
rank = 1
prev_score = None
merged_df = merged_df.sort_values('score', ascending=False).reset_index(drop=True)
merged_df['순위'] = merged_df.index + 1

for i, row in merged_df.iterrows():
    if prev_score is None or prev_score != row['score']:
        rank = i + 1
    if prev_score == row['score']:
        print(f"공동{rank}위: {row['동별']} ({row['score']})")
    else:
        print(f"{rank}위: {row['동별']} ({row['score']})")
    prev_score = row['score']


