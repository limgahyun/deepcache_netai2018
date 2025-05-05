"""
                           DeepCache
               
DeepCache는 다음 BSD 3-Clause License에 따라 배포됩니다:

Copyright(c) 2019
                University of Minensota - Twin Cities
        Authors: Arvind Narayanan, Saurabh Verma, Eman Ramadan, Pariya Babaie, and Zhi-Li Zhang

모든 권리 보유.

소스 및 바이너리 형태로 재배포 및 사용은 다음 조건을 충족하는 경우에 허용됩니다:

1. 소스 코드의 재배포는 위의 저작권 고지, 이 조건 목록 및 다음 면책 조항을 유지해야 합니다.

2. 바이너리 형태로의 재배포는 위의 저작권 고지, 이 조건 목록 및 다음 면책 조항을 문서 및/또는 배포와 함께 제공되는 기타 자료에 포함해야 합니다.

3. 저작권 보유자의 이름이나 기여자의 이름은 사전 서면 허가 없이 이 소프트웨어에서 파생된 제품을 홍보하거나 보증하는 데 사용될 수 없습니다.

이 소프트웨어는 저작권 보유자 및 기여자에 의해 "있는 그대로" 제공되며, 상품성 및 특정 목적에 대한 적합성에 대한 묵시적 보증을 포함하되 이에 국한되지 않는 명시적 또는 묵시적 보증 없이 제공됩니다. 저작권 보유자 또는 기여자는 이 소프트웨어의 사용으로 인해 발생하는 직접적, 간접적, 부수적, 특별, 모범적 또는 결과적 손해(대체 상품 또는 서비스의 조달, 사용 손실, 데이터 또는 이익 손실, 또는 비즈니스 중단 포함)에 대해 책임을 지지 않습니다.

@author: Pariya Babaie (babai008@umn.edu) & Eman Ramadan (eman@cs.umn.edu)


설명:
    이 코드는 분석된 요청 파일에 대해 다음과 같은 그래프를 생성합니다:
    1- {RequestFile}_ActiveObjects: 매 시간 요청된 고유 활성 객체 수
    2- {RequestFile}_Frequency: 생성된 객체의 빈도 분포
    3- {RequestFile}_Lifespan: 각 수명 값에 대해 생성된 객체의 비율
    4- {RequestFile}_ObjectIntroduction: 매일 도입된 객체의 비율
    5- {RequestFile}_HourlyRequestRatio: 시간별 객체 요청 비율 (MediSyn 데이터셋용)

전처리 스크립트:
    plotObjectProperties.py 스크립트를 실행하기 전에 requestAnalysis.py 스크립트를 실행해야 합니다.
    requestAnalysis.py: 생성된 요청을 분석하고 여기서 사용되는 _bin, _properties 파일을 생성합니다.

입력:
    입력 디렉토리는 '../Datasets'입니다:
    1- REQUESTFILENAME: 분석할 요청 파일 (Line 36에서 설정)
    2- PLOT_EXT: 생성된 그래프의 확장자, 기본값은 pdf (대안으로 png 사용 가능)

출력:
    출력 그래프는 '../Datasets' 디렉토리에 생성됩니다.
"""

from __future__ import print_function
import pandas as pd
import matplotlib.pyplot as plt
import collections


REQDIR = '../Datasets/'  # 요청 파일이 위치한 디렉토리
REQUESTFILENAME = 'mediSynDataset_x2_O3488.csv'  # 분석할 요청 파일 이름
BINFILENAME = REQDIR + REQUESTFILENAME[:-4] + '_bins.csv'  # bin 파일 경로
PROPERTIES_FILENAME = REQDIR + REQUESTFILENAME[:-4] + '_properties.csv'  # 속성 파일 경로
HOURLY_REQUEST_RATIO_FILE = 'hourly_request_ratio.csv'  # 시간별 요청 비율 파일

PLOT_PATH_PREFIX = '{}{}_'.format(REQDIR, REQUESTFILENAME[:-4])  # 그래프 저장 경로 접두사
PLOT_EXT = 'pdf'  # 그래프 확장자


"""####################  각 시간대의 활성 객체 수 그래프 생성  ######################################"""
print('활성 객체 수 그래프 생성 중...')
bindf = pd.read_csv(BINFILENAME)  # bin 파일 읽기
bindf['binID'] = bindf['binID'] - bindf.iloc[0]['binID']  # binID를 0부터 시작하도록 조정
bindf['binID'] = bindf['binID'].astype('int')  # binID를 정수형으로 변환
plt.close('all')
plt.figure(figsize=(3, 2))
bindf.plot('binID', 'uniqueObjNum', legend=None, color='blue')  # binID와 고유 객체 수를 그래프로 그림
plt.xlabel('시간별 Bin')  # x축 레이블
plt.ylabel('활성 객체 수')  # y축 레이블
plt.title('시간당 활성 객체 수')  # 그래프 제목
plt.grid()  # 격자 추가
plt.tight_layout()  # 레이아웃 조정
plt.savefig('{}ActiveObjects.{}'.format(PLOT_PATH_PREFIX, PLOT_EXT), format=PLOT_EXT, facecolor='white', dpi=300)  # 그래프 저장


"""####################  객체 빈도 분포 그래프 생성  #########################################################"""
print('객체 빈도 분포 그래프 생성 중...')
objdf = pd.read_csv(PROPERTIES_FILENAME)  # 속성 파일 읽기
objdf.sort_values('frequency', ascending=False, inplace=True)  # 빈도 기준으로 내림차순 정렬
plt.close('all')
plt.figure(figsize=(4, 3))
plt.loglog(range(1, len(objdf.frequency) + 1), objdf.frequency, color='blue')  # 로그-로그 스케일로 그래프 그림
plt.yscale('log')  # y축 로그 스케일 설정
plt.xlabel('객체 순위')  # x축 레이블
plt.ylabel('객체 빈도')  # y축 레이블
plt.title('객체 빈도 분포')  # 그래프 제목
plt.grid()  # 격자 추가
plt.tight_layout()  # 레이아웃 조정
plt.savefig('{}Frequency.{}'.format(PLOT_PATH_PREFIX, PLOT_EXT), format=PLOT_EXT, facecolor='white', dpi=300)  # 그래프 저장


"""####################  객체 수명 비율 그래프 생성  #########################################"""
print('객체 수명 비율 그래프 생성 중...')
plt.close('all')
plt.figure(figsize=(4, 3))
objdf.sort_values('lifeSpan', ascending=True, inplace=True)  # 수명 기준으로 오름차순 정렬
lifeSpans = objdf["lifeSpan"].apply(int)  # 수명을 정수형으로 변환
max_days = max(lifeSpans.unique())  # 최대 수명 값 계산
days = lifeSpans.unique()  # 고유 수명 값 가져오기
counts = collections.Counter(lifeSpans)  # 각 수명 값의 빈도 계산
plt.grid(zorder=0)  # 격자 추가
if len(days) <= 5:  # 수명 값이 5개 이하인 경우
    fig, ax = plt.subplots(1, 1)
    ax.bar(days, [float(counts[key]) / objdf.shape[0] for key in counts], width=0.2, zorder=3, color='blue')  # 막대 그래프 생성
    ax.set_xlim(0, max_days)  # x축 범위 설정
else:  # 수명 값이 5개 초과인 경우
    plt.bar(days, [float(counts[key]) / objdf.shape[0] for key in counts], zorder=3, color='blue')  # 막대 그래프 생성
plt.xlabel('객체 수명 (일 단위)')  # x축 레이블
plt.ylabel('비율 (전체 객체 대비)')  # y축 레이블
plt.title('객체 수명 비율')  # 그래프 제목
plt.tight_layout()  # 레이아웃 조정
plt.savefig('{}Lifespan.{}'.format(PLOT_PATH_PREFIX, PLOT_EXT), format=PLOT_EXT, facecolor='white', dpi=300)  # 그래프 저장


"""####################  매일 도입된 객체 비율 그래프 생성  #############################################"""
print('매일 도입된 객체 비율 그래프 생성 중...')
plt.close('all')
plt.figure(figsize=(4, 3))
plt.grid(zorder=0)  # 격자 추가
objdf.sort_values('start_day', ascending=True, inplace=True)  # 시작 날짜 기준으로 정렬
start_days = objdf["start_day"].apply(int)  # 시작 날짜를 정수형으로 변환
days = start_days.unique()  # 고유 시작 날짜 가져오기
counts = collections.Counter(start_days)  # 각 시작 날짜의 빈도 계산
if len(days) <= 5:  # 시작 날짜가 5개 이하인 경우
    fig, ax = plt.subplots(1, 1)
    ax.bar(days, [float(counts[key]) / objdf.shape[0] for key in counts], width=0.2, zorder=3, color='blue')  # 막대 그래프 생성
    ax.set_xlim(0, max_days)  # x축 범위 설정
else:  # 시작 날짜가 5개 초과인 경우
    plt.bar(days, [float(counts[key]) / objdf.shape[0] for key in counts], zorder=3, color='blue')  # 막대 그래프 생성
plt.title('매일 도입된 객체 비율')  # 그래프 제목
plt.xlabel('시뮬레이션 시간 (일 단위)')  # x축 레이블
plt.ylabel('비율 (전체 객체 대비)')  # y축 레이블
plt.tight_layout()  # 레이아웃 조정
plt.savefig('{}ObjectIntroduction.{}'.format(PLOT_PATH_PREFIX, PLOT_EXT), format=PLOT_EXT, facecolor='white', dpi=300)  # 그래프 저장

del bindf  # 메모리 정리
del objdf  # 메모리 정리


"""####################  시간별 요청 비율 그래프 생성  ##################################################"""
print('시간별 요청 비율 그래프 생성 중...')
if 'mediSynDataset' in REQUESTFILENAME:  # MediSyn 데이터셋인 경우
    hour_date_df = pd.read_csv(HOURLY_REQUEST_RATIO_FILE, header=None, skiprows=[0, 0], names=['hour', 'ratio'])  # 시간별 요청 비율 파일 읽기
    plt.close('all')
    plt.figure(figsize=(4, 3))
    plt.bar(hour_date_df.hour, hour_date_df.ratio, color='blue')  # 막대 그래프 생성
    plt.title('시간별 접근 비율')  # 그래프 제목
    plt.ylabel('접근 비율')  # y축 레이블
    plt.xlabel('시간')  # x축 레이블
    plt.tight_layout()  # 레이아웃 조정
    plt.savefig('{}HourlyRequestRatio.{}'.format(PLOT_PATH_PREFIX, PLOT_EXT), format=PLOT_EXT,
                facecolor='white', dpi=300)  # 그래프 저장
    del hour_date_df  # 메모리 정리


print('완료')
